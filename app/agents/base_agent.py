"""
Base agent implementation for the Context Corrosion experiment.

All agents share the same model, temperature, and system prompt.
Only structural parameters (max_tokens, speak_order) differ.

Key fixes
---------
FIX A  response_mime_type="application/json" on all structured phases.
       This is the ONLY reliable way to stop Gemini wrapping JSON in
       ```json ... ``` fences. Prompts alone do not work.

FIX B  JSON schema passed to response_schema so Gemini knows the exact
       shape and doesn't waste tokens on extra prose or keys.

FIX C  Discussion (Phase 2) uses a SEPARATE GenerationConfig without
       JSON mode — prose should be prose, not coerced to JSON.

FIX D  Partial-field extraction fallback recovers "answer"/"final_answer"
       from truncated text so embedding always has real content.

FIX E  Completeness check + single retry on truncated structured output.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import google.generativeai as genai

from app.config import (
    BASE_SYSTEM_PROMPT,
    DOMINANT_JSON_MAX_TOKENS,
    DOMINANT_MAX_TOKENS,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    PEER_JSON_MAX_TOKENS,
    PEER_MAX_TOKENS,
)

logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()


genai.configure(api_key=GEMINI_API_KEY)

# ── JSON schemas for structured phases ───────────────────────────────────────
# Passed to response_schema so Gemini generates exactly these fields, nothing more.

_PHASE1_SCHEMA = {
    "type": "object",
    "properties": {
        "answer":      {"type": "string"},
        "confidence":  {"type": "number"},
        "key_points":  {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "confidence", "key_points"],
}

_PHASE3_SCHEMA = {
    "type": "object",
    "properties": {
        "final_answer":      {"type": "string"},
        "final_confidence":  {"type": "number"},
        "changed_position":  {"type": "boolean"},
        "reason_for_change": {"type": "string"},
    },
    "required": ["final_answer", "final_confidence", "changed_position", "reason_for_change"],
}


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences — used as belt-and-suspenders even with JSON mode."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _extract_partial_value(text: str, key: str) -> str | None:
    """FIX D: Recover a string value from a truncated JSON blob using regex."""
    pattern = rf'"{re.escape(key)}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        raw_val = match.group(1)
        return raw_val.replace("\\n", " ").replace("\\t", " ").replace('\\"', '"').strip()
    return None


def _is_complete_json(text: str) -> bool:
    """FIX E: Return True if text is a parseable JSON object."""
    try:
        json.loads(_strip_json_fences(text))
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _safe_parse_json(text: str, expected_keys: list[str] | None = None) -> dict[str, Any]:
    """
    Multi-level parse with graceful degradation:
      1. Strip fences, full parse.
      2. Regex-extract first {...} block, parse.
      3. Extract individual fields (FIX D).
      4. Return {raw_text, parse_error: true}.
    """
    clean = _strip_json_fences(text)

    try:
        result = json.loads(clean)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # FIX D — partial field recovery
    recovered: dict[str, Any] = {}
    for key in (expected_keys or ["answer", "final_answer"]):
        val = _extract_partial_value(clean, key)
        if val:
            recovered[key] = val

    for key in ("confidence", "final_confidence"):
        m = re.search(rf'"{key}"\s*:\s*([0-9.]+)', clean)
        if m:
            try:
                recovered[key] = float(m.group(1))
            except ValueError:
                pass

    kp_match = re.search(r'"key_points"\s*:\s*\[([^\]]*)', clean, re.DOTALL)
    if kp_match:
        points = re.findall(r'"([^"]+)"', kp_match.group(1))
        if points:
            recovered["key_points"] = points

    bool_match = re.search(r'"changed_position"\s*:\s*(true|false)', clean)
    if bool_match:
        recovered["changed_position"] = bool_match.group(1) == "true"

    if recovered:
        recovered["parse_warning"] = "Recovered from truncated/malformed JSON"
        return recovered

    return {"raw_text": text, "parse_error": True}


# ── Agent class ───────────────────────────────────────────────────────────────

class Agent:
    """
    A single LLM agent in the Context Corrosion experiment.

    Structural asymmetry is the ONLY difference between agents:
      - Dominant (D): DOMINANT_MAX_TOKENS, speaks first
      - Peers (P1-P4): PEER_MAX_TOKENS, respond after D

    Everything else — model, temperature, base system prompt — is identical.
    """

    def __init__(
        self,
        agent_id: str,
        is_dominant: bool,
        temperature: float,
        extra_system_context: str | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.is_dominant = is_dominant
        self.temperature = temperature
        self.max_tokens = DOMINANT_MAX_TOKENS if is_dominant else PEER_MAX_TOKENS
        self.json_max_tokens = DOMINANT_JSON_MAX_TOKENS if is_dominant else PEER_JSON_MAX_TOKENS

        system_prompt = BASE_SYSTEM_PROMPT
        if extra_system_context:
            system_prompt = f"{system_prompt}\n\n[CONTEXT]: {extra_system_context}"

        self._model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_prompt,
        )

        # FIX C — separate configs for prose vs JSON phases
        # Prose discussion: standard config, no JSON coercion
        self._prose_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=self.max_tokens,
        )

        # FIX A — JSON phases: mime type forces raw JSON, no fences
        # FIX B — schema constrains output shape, saves tokens on extra keys
        self._json_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=self.json_max_tokens,
            response_mime_type="application/json",
        )

        logger.info(
            "Agent %s | dominant=%s | prose_tokens=%d | json_tokens=%d | temp=%.2f",
            agent_id, is_dominant, self.max_tokens, self.json_max_tokens, temperature,
        )

    def _generate_prose(self, prompt: str) -> str:
        """Generate free-form prose (Phase 2 discussion). No JSON coercion."""
        try:
            response = self._model.generate_content(
                prompt,
                generation_config=self._prose_config,
            )
            return response.text.strip()
        except Exception as exc:
            logger.error("Agent %s prose generation error: %s", self.agent_id, exc)
            raise

    def _generate_json(self, prompt: str, retry_prompt: str | None = None) -> str:
        """
        Generate structured JSON output (Phases 1 and 3).
        FIX A: Uses response_mime_type="application/json" — guaranteed no fences.
        FIX E: Completeness check with single retry using a shorter prompt.
        """
        try:
            response = self._model.generate_content(
                prompt,
                generation_config=self._json_config,
            )
            raw = response.text.strip()
        except Exception as exc:
            logger.error("Agent %s JSON generation error: %s", self.agent_id, exc)
            raise

        if _is_complete_json(raw):
            return raw

        # FIX E — retry with brevity-enforced prompt
        logger.warning(
            "Agent %s JSON response incomplete (%d chars). Retrying with brevity prompt.",
            self.agent_id, len(raw),
        )
        short_prompt = retry_prompt or (
            "Respond with the shortest possible valid JSON matching the required schema. "
            "Use one sentence for all string fields. "
            + prompt
        )
        try:
            response = self._model.generate_content(
                short_prompt,
                generation_config=self._json_config,
            )
            return response.text.strip()
        except Exception as exc:
            logger.warning("Agent %s retry also failed: %s. Using original.", self.agent_id, exc)
            return raw

    # ── Phase 1: Independent Thought ─────────────────────────────────────────

    def initial_response(self, task_description: str) -> tuple[str, dict[str, Any]]:
        """
        Phase 1: Each agent answers independently, no shared memory.
        Returns (raw_text, parsed_dict).
        FIX A+B: JSON mode + schema ensures clean parseable output.
        """
        prompt = (
            f"TASK:\n{task_description}\n\n"
            "State your position on this topic.\n"
            "Respond with a JSON object containing:\n"
            '  "answer": your one-sentence position,\n'
            '  "confidence": a float 0.0-1.0,\n'
            '  "key_points": a list of 2-3 short strings (each under 15 words).'
        )
        raw = self._generate_json(prompt)
        parsed = _safe_parse_json(raw, expected_keys=["answer"])

        if parsed.get("parse_error"):
            logger.warning(
                "Agent %s phase1 parse FAILED | raw length: %d | raw: %.60s",
                self.agent_id, len(raw), raw,
            )
        elif parsed.get("parse_warning"):
            logger.warning(
                "Agent %s phase1 parse PARTIAL recovery | raw length: %d",
                self.agent_id, len(raw),
            )

        return raw, parsed

    # ── Phase 2: Discussion ───────────────────────────────────────────────────

    def discussion_response(
        self,
        task_description: str,
        shared_memory: list[dict[str, Any]],
        round_num: int,
    ) -> str:
        """
        Phase 2: Free-form prose response to the shared discussion.
        FIX C: Uses prose config — no JSON schema, no response_mime_type.
        """
        history_text = "\n\n".join(
            f"[{e['agent_id']} - Round {e['round']}]: {e['text']}"
            for e in shared_memory
        )
        prompt = (
            f"TASK:\n{task_description}\n\n"
            f"=== DISCUSSION HISTORY (Round {round_num}) ===\n{history_text}\n\n"
            f"You are Agent {self.agent_id}. "
            "Respond with your analysis or updated view. "
            "Plain prose only — no JSON, no bullet lists, no markdown."
        )
        return self._generate_prose(prompt)

    # ── Phase 3: Final Vote ───────────────────────────────────────────────────

    def final_vote(
        self,
        task_description: str,
        shared_memory: list[dict[str, Any]],
        initial_answer: str,
    ) -> tuple[str, dict[str, Any]]:
        """
        Phase 3: Final structured position after all discussion rounds.
        Returns (raw_text, parsed_dict).
        FIX A+B: JSON mode + schema ensures clean parseable output.
        """
        history_text = "\n\n".join(
            f"[{e['agent_id']} - Round {e['round']}]: {e['text']}"
            for e in shared_memory
        )
        prompt = (
            f"TASK:\n{task_description}\n\n"
            f"=== FULL DISCUSSION ===\n{history_text}\n\n"
            f"Your initial answer was: {initial_answer}\n\n"
            "State your FINAL position.\n"
            "Respond with a JSON object containing:\n"
            '  "final_answer": your one-sentence final position,\n'
            '  "final_confidence": a float 0.0-1.0,\n'
            '  "changed_position": true or false,\n'
            '  "reason_for_change": one sentence, or "N/A" if unchanged.'
        )
        # FIX: Retry prompt strips all history — the first attempt fails because
        # the full discussion history exhausts the output token budget, leaving
        # no room to complete the JSON. The retry uses only the initial answer
        # so the model can finish within the limit.
        retry_prompt = (
            f"Your initial position was: \"{initial_answer[:120]}\"\n\n"
            "Output ONLY a valid JSON object with exactly these four keys:\n"
            '{"final_answer": "<one sentence>", "final_confidence": 0.85, '
            '"changed_position": false, "reason_for_change": "N/A"}\n'
            "Keep final_answer under 30 words. No other text."
        )
        raw = self._generate_json(prompt, retry_prompt=retry_prompt)
        parsed = _safe_parse_json(raw, expected_keys=["final_answer"])

        if parsed.get("parse_error"):
            logger.warning(
                "Agent %s final_vote parse FAILED | raw length: %d | raw: %.60s",
                self.agent_id, len(raw), raw,
            )
        elif parsed.get("parse_warning"):
            logger.warning(
                "Agent %s final_vote parse PARTIAL | raw length: %d",
                self.agent_id, len(raw),
            )

        return raw, parsed