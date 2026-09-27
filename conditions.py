"""
conditions.py — The three experimental tiers for the ICL dialect study.

Baseline-0  (Identity Control)
    Returns the raw dialect string as the "hypothesis".
    No model calls. Establishes the mathematical floor of lexical distance.
    Expected: ~57% WER, ~24% CER.

Condition-A  (Zero-Shot LLM)
    Single instruction: "Translate this dialect into Standard German."
    No examples. Tests the LLM's intrinsic sociolinguistic competence.

Condition-B  (Few-Shot LLM / In-Context Learning)
    Prepends N_FEW_SHOT_EXAMPLES dialect->standard pairs from the SAME REGION
    before each test sentence. Tests whether in-context examples guide the
    model toward the reference distribution.

LLM Provider:
    Qwen (qwen/qwen3.8-27b) via Groq API — https://console.groq.com
    Free tier: 30 req/min, 14,400 req/day.
    Install: pip install groq
"""

from __future__ import annotations

import logging
import re
import time
from typing import List

import pandas as pd

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_REQUEST_DELAY,
    GEN_TEMPERATURE,
    N_FEW_SHOT_EXAMPLES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared prompt components
# ---------------------------------------------------------------------------

_ZERO_SHOT_SYSTEM = (
    "You are an expert German linguist. "
    "Your only task is to normalise regional Bavarian dialect sentences "
    "into Standard High German (Hochdeutsch).\n\n"
    "STRICT RULES:\n"
    "1. Output EXACTLY ONE sentence — the normalised Standard German sentence.\n"
    "2. Do NOT explain, annotate, correct, or add any commentary.\n"
    "3. Do NOT use markdown, bullet points, arrows, or corrections (e.g. no '* ->', no bold).\n"
    "4. Do NOT add prefixes like 'Standard German:', 'Translation:', 'Answer:'.\n"
    "5. Preserve the full meaning — do not add or remove information.\n"
    "6. If the input is already Standard German, return it unchanged."
)

_FEW_SHOT_SYSTEM_HEADER = (
    "You are an expert German linguist. "
    "Your only task is to normalise regional Bavarian dialect sentences "
    "into Standard High German (Hochdeutsch).\n\n"
    "Study these examples of the expected output format:\n\n"
    "{examples}\n"
    "STRICT RULES:\n"
    "1. Output EXACTLY ONE sentence — the normalised Standard German sentence.\n"
    "2. Do NOT explain, annotate, correct, or add any commentary.\n"
    "3. Do NOT use markdown, bullet points, arrows, or corrections (e.g. no '* ->', no bold).\n"
    "4. Do NOT add prefixes like 'Standard German:', 'Translation:', 'Answer:'.\n"
    "5. Preserve the full meaning exactly — follow the style of the examples above."
)

_EXAMPLE_TEMPLATE = "Dialect: {dial}\nStandard German: {deu}"

# User message: just the dialect sentence — system prompt handles all formatting
_USER_TEMPLATE = "{dial}"


def _format_examples(example_df: pd.DataFrame) -> str:
    """Format a small DataFrame of examples into a prompt block."""
    lines = []
    for _, row in example_df.iterrows():
        lines.append(
            _EXAMPLE_TEMPLATE.format(dial=row["dialect"], deu=row["reference"])
        )
    return "\n\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Groq client helper
# ---------------------------------------------------------------------------

def _get_groq_client():
    """Return an initialised Groq client."""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError(
            "Run:  pip install groq\n"
            "Then set GROQ_API_KEY in config.py or as an environment variable."
        )
    if not GROQ_API_KEY:
        raise ValueError(
            "Groq API key not set!\n"
            "  1. Go to https://console.groq.com\n"
            "  2. Sign in -> 'API Keys' -> 'Create API Key'\n"
            "  3. Set GROQ_API_KEY in config.py or via environment variable."
        )
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)


def _clean_output(text: str, fallback: str = "") -> str:
    """
    Strip non-sentence artifacts from LLM output.

    Removes:
    - Qwen3 <think>...</think> reasoning blocks
    - Markdown bold/italic markers  (**text**, *text*)
    - Annotation/correction patterns  ("* -> ...", "Correction: ...")
    - Prefixes like "Standard German:", "Translation:", "Draft Output:"
    - Reasoning markers like "Let's refine:", "Let's think:"
    - Leading/trailing quotes and whitespace
    - Multiline responses (keep only first non-empty line)
    """
    if not text:
        return fallback

    # Strip Qwen3 <think>...</think> reasoning blocks (can be multiline)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        return fallback

    # Remove markdown bold/italic
    text = re.sub(r"[*_]{1,3}(.*?)[*_]{1,3}", r"\1", text)

    # Drop annotation/correction lines and reasoning artifact lines
    lines = text.splitlines()
    clean_lines = [
        ln for ln in lines
        if ln.strip()
        and not re.match(r"^\s*[*\-]\s*(-\x3e|\u2192)", ln)           # arrow annotations
        and not re.match(r"^\s*(Correction|Note|Anmerkung):", ln, re.I)  # meta-commentary
        and not re.match(r"^\s*(Standard German|Translation|Output|Answer):", ln, re.I)
        and not re.match(r"^\s*(Draft Output|Draft|My output|Here is|Normalized):", ln, re.I)
        and not re.match(r"^\s*Let'?s\s+(refine|think|review|check|verify)", ln, re.I)
    ]

    if not clean_lines:
        return fallback

    # Take only the first substantive line
    result = clean_lines[0].strip()

    # Strip surrounding quotes
    result = re.sub(r'^["\u201c\u201e]|["\u201d\u201e]$', "", result).strip()

    # Strip common prefixes that sneak through (case-insensitive)
    result = re.sub(
        r"^(Standard German|Hochdeutsch|Translation|Output|Answer|Result"
        r"|Draft Output|Draft|Normalized|Here is)[:\s]+",
        "", result, flags=re.I
    ).strip()

    # Drop if the line still starts with a reasoning marker
    if re.match(r"^Let'?s\s+(refine|think|review|check|verify)", result, re.I):
        return fallback

    return result if result else fallback


def _call_groq(
    client,
    system_prompt: str,
    user_message: str,
    model: str = GROQ_MODEL,
    temperature: float = GEN_TEMPERATURE,
    max_retries: int = 4,
    base_delay: float = 5.0,
) -> str:
    """Single Groq chat-completion call with exponential-backoff retry."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                temperature=temperature,
                max_tokens=512,
            )
            raw    = (response.choices[0].message.content or "").strip()
            result = _clean_output(raw)

            # Validate: reject if output is suspiciously short vs input
            input_words  = len(user_message.split())
            output_words = len(result.split()) if result else 0
            if result and output_words >= max(2, input_words * 0.3):
                return result

            logger.warning(
                "Output suspiciously short (%d words vs %d input words) on attempt %d: %r",
                output_words, input_words, attempt + 1, raw[:80],
            )

        except Exception as exc:
            err = str(exc)
            if "429" in err or "rate_limit" in err.lower() or "503" in err:
                wait = base_delay * (2 ** attempt)   # 5s, 10s, 20s, 40s
                logger.warning(
                    "Groq transient error (attempt %d/%d), retrying in %.0fs: %s",
                    attempt + 1, max_retries, wait, exc,
                )
                time.sleep(wait)
            else:
                raise   # non-retryable — surface immediately

    logger.error("Groq failed after %d retries. Returning empty string.", max_retries)
    return ""


# ---------------------------------------------------------------------------
# Tier 0 — Identity baseline
# ---------------------------------------------------------------------------

def run_identity(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline-0: return the raw dialect text as the hypothesis.

    No model calls. Computes the mathematical floor of WER/CER/BLEU
    caused purely by the lexical distance between dialect and standard German.
    """
    result = test_df.copy()
    result["condition"]  = "Baseline-0 (Identity)"
    result["hypothesis"] = result["dialect"]
    logger.info("Baseline-0 complete: %d sentences (no API calls).", len(result))
    return result


# ---------------------------------------------------------------------------
# Tier A — Zero-Shot LLM  (via Groq)
# ---------------------------------------------------------------------------

def run_zero_shot(
    test_df: pd.DataFrame,
    request_delay: float = GROQ_REQUEST_DELAY,
) -> pd.DataFrame:
    """
    Condition-A: zero-shot normalisation with a single system instruction.

    One API call per test sentence via Groq (llama-3.3-70b-versatile).
    """
    client = _get_groq_client()
    hypotheses: List[str] = []

    total = len(test_df)
    for idx, (_, row) in enumerate(test_df.iterrows()):
        user_msg = _USER_TEMPLATE.format(dial=row["dialect"])
        try:
            hyp = _call_groq(client, _ZERO_SHOT_SYSTEM, user_msg)
        except Exception as exc:
            logger.warning("[Zero-shot] Error on row %d: %s", idx, exc)
            hyp = ""
        hypotheses.append(hyp)

        if (idx + 1) % 10 == 0:
            logger.info("[Zero-shot] %d/%d done.", idx + 1, total)

        if request_delay > 0:
            time.sleep(request_delay)

    result = test_df.copy()
    result["condition"]  = "Condition-A (Zero-Shot)"
    result["hypothesis"] = hypotheses
    logger.info("[Zero-shot] Complete: %d sentences.", total)
    return result


# ---------------------------------------------------------------------------
# Tier B — Few-Shot LLM (In-Context Learning)  (via Groq)
# ---------------------------------------------------------------------------

def run_few_shot(
    test_df: pd.DataFrame,
    examples_df: pd.DataFrame,
    n_examples: int = N_FEW_SHOT_EXAMPLES,
    request_delay: float = GROQ_REQUEST_DELAY,
) -> pd.DataFrame:
    """
    Condition-B: few-shot normalisation with in-region dialect->standard examples.

    For each test sentence, the system prompt is augmented with N_FEW_SHOT_EXAMPLES
    parallel pairs drawn from the SAME REGION (from the held-out example pool).
    This tests whether the model can learn from in-context demonstrations.

    Parameters
    ----------
    test_df     : Test sentences (must not overlap with examples_df).
    examples_df : Few-shot pool (first N rows per region from data_loader).
    n_examples  : Number of examples to include per region (default 5).
    request_delay: Seconds between API calls.
    """
    client = _get_groq_client()

    # Pre-build one system prompt per region (reused across all sentences in that region)
    region_prompts: dict[str, str] = {}
    for region in test_df["region"].unique():
        region_examples = examples_df[examples_df["region"] == region].head(n_examples)
        if region_examples.empty:
            logger.warning("No few-shot examples for region '%s' — falling back to zero-shot.", region)
            region_prompts[region] = _ZERO_SHOT_SYSTEM
        else:
            example_block = _format_examples(region_examples)
            region_prompts[region] = _FEW_SHOT_SYSTEM_HEADER.format(
                examples=example_block
            )
            logger.debug(
                "Region '%s': %d examples loaded into few-shot prompt.",
                region, len(region_examples),
            )

    hypotheses: List[str] = []
    total = len(test_df)

    for idx, (_, row) in enumerate(test_df.iterrows()):
        system_prompt = region_prompts[row["region"]]
        user_msg      = _USER_TEMPLATE.format(dial=row["dialect"])
        try:
            hyp = _call_groq(client, system_prompt, user_msg)
        except Exception as exc:
            logger.warning("[Few-shot] Error on row %d: %s", idx, exc)
            hyp = ""
        hypotheses.append(hyp)

        if (idx + 1) % 10 == 0:
            logger.info("[Few-shot] %d/%d done.", idx + 1, total)

        if request_delay > 0:
            time.sleep(request_delay)

    result = test_df.copy()
    result["condition"]  = "Condition-B (Few-Shot)"
    result["hypothesis"] = hypotheses
    logger.info("[Few-shot] Complete: %d sentences.", total)
    return result
