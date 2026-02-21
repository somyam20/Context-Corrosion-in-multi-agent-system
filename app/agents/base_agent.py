"""
Base agent implementation for the Context Corrosion experiment.
All agents share the same model, temperature, and system prompt.
Only structural parameters (max_tokens, speak_order) differ.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import google.genai as genai

from app.config import (
    BASE_SYSTEM_PROMPT,
    DOMINANT_MAX_TOKENS,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    PEER_MAX_TOKENS,
)
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences from model output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _safe_parse_json(text: str) -> dict[str, Any]:
    """Attempt to parse JSON, returning raw text on failure."""
    clean = _strip_json_fences(text)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Try extracting first JSON block
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"raw_text": text, "parse_error": True}


class Agent:
    """
    A single LLM agent.

    Parameters
    ----------
    agent_id : str
        Unique identifier (e.g., "D", "P1").
    is_dominant : bool
        If True, uses DOMINANT_MAX_TOKENS; otherwise PEER_MAX_TOKENS.
    temperature : float
        Sampling temperature (same for all agents per experiment).
    extra_system_context : str | None
        Additional context injected only for the dominant agent when biased.
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

        system_prompt = BASE_SYSTEM_PROMPT
        if extra_system_context:
            system_prompt = f"{system_prompt}\n\n[CONTEXT]: {extra_system_context}"

        self._model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_prompt,
        )
        self._generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=self.max_tokens,
        )

        logger.info(
            "Agent %s initialized | dominant=%s | max_tokens=%d | temperature=%.2f",
            agent_id,
            is_dominant,
            self.max_tokens,
            temperature,
        )

    def generate(self, prompt: str) -> str:
        """
        Send a prompt and return raw text response.

        Parameters
        ----------
        prompt : str
            The full prompt to send.

        Returns
        -------
        str
            Raw model response text.
        """
        try:
            response = self._model.generate_content(
                prompt,
                generation_config=self._generation_config,
            )
            return response.text.strip()
        except Exception as exc:
            logger.error("Agent %s generation error: %s", self.agent_id, exc)
            raise

    def initial_response(self, task_description: str) -> tuple[str, dict[str, Any]]:
        """
        Phase 1: Independent thought — no shared memory.

        Returns
        -------
        tuple[str, dict]
            (raw_text, parsed_json)
        """
        prompt = (
            f"TASK:\n{task_description}\n\n"
            "Respond ONLY with a valid JSON object in this exact format:\n"
            '{"answer": "your position", "confidence": 0.0-1.0, "key_points": ["point1", "point2"]}\n'
            "Do NOT include any text outside the JSON."
        )
        raw = self.generate(prompt)
        parsed = _safe_parse_json(raw)
        return raw, parsed

    def discussion_response(
        self,
        task_description: str,
        shared_memory: list[dict[str, Any]],
        round_num: int,
    ) -> str:
        """
        Phase 2: Discussion round response given shared memory.

        Parameters
        ----------
        task_description : str
        shared_memory : list[dict]
            Chronological list of all prior statements.
        round_num : int
            Current discussion round (1-indexed).

        Returns
        -------
        str
            Agent's discussion statement.
        """
        history_text = "\n\n".join(
            f"[{entry['agent_id']} - Round {entry['round']}]: {entry['text']}"
            for entry in shared_memory
        )
        prompt = (
            f"TASK:\n{task_description}\n\n"
            f"=== DISCUSSION HISTORY (Round {round_num}) ===\n{history_text}\n\n"
            f"You are Agent {self.agent_id}. "
            "Provide your analysis or updated view based on the discussion so far. "
            "Be concise and focused."
        )
        return self.generate(prompt)

    def final_vote(
        self,
        task_description: str,
        shared_memory: list[dict[str, Any]],
        initial_answer: str,
    ) -> tuple[str, dict[str, Any]]:
        """
        Phase 3: Final position after all discussion rounds.

        Returns
        -------
        tuple[str, dict]
            (raw_text, parsed_json)
        """
        history_text = "\n\n".join(
            f"[{entry['agent_id']} - Round {entry['round']}]: {entry['text']}"
            for entry in shared_memory
        )
        prompt = (
            f"TASK:\n{task_description}\n\n"
            f"=== FULL DISCUSSION ===\n{history_text}\n\n"
            f"Your initial answer was: {initial_answer}\n\n"
            "Now provide your FINAL position as a JSON object:\n"
            '{"final_answer": "...", "final_confidence": 0.0-1.0, '
            '"changed_position": true/false, "reason_for_change": "..."}\n'
            "Do NOT include any text outside the JSON."
        )
        raw = self.generate(prompt)
        parsed = _safe_parse_json(raw)
        return raw, parsed
