"""
config.py — Configuration for the Betthupferl ICL Dialect Normalisation Study.

Research Question:
    How effectively do autoregressive LLMs normalise regional German dialects
    into Standard German, and how much does few-shot in-context learning reduce
    Word Error Rate compared to zero-shot extraction?

Experimental Design (3 tiers):
    Baseline-0  : Identity control  — raw dialect vs. reference (no LLM)
    Condition-A  : Zero-shot LLM    — single instruction, no examples
    Condition-B  : Few-shot LLM     — 3-5 in-context dialect->standard pairs

# LLM Provider (All LLM calls - Generation & LLM Judge):
#     Groq API -> https://console.groq.com
#     Model: qwen/qwen3.8-27b
#     Free tier: 30 req/min, 14,400 req/day
#     Install  : pip install groq
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Groq API  (Generation & LLM-as-a-Judge)
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# Model used for dialect normalisation and judging
GROQ_MODEL: str = "qwen/qwen3.8-27b"

# Groq free tier request delay
GROQ_REQUEST_DELAY: float = 2.0   # seconds between API calls

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATA_ROOT: str = os.getenv(
    "BETTHUPFERL_DATA_ROOT",
    r"c:\Users\Rahul\Documents\Poster Nlp\pure_text",
)

REGIONS: list[str] = [
    "mittelfranken",   # Middle Franconia
    "niederbayern",    # Lower Bavaria
    "oberbayern",      # Upper Bavaria
    "oberfranken",     # Upper Franconia
    "oberpfalz",       # Upper Palatinate
    "schwaben",        # Swabia
    "unterfranken",    # Lower Franconia
]

# ---------------------------------------------------------------------------
# Experimental parameters
# ---------------------------------------------------------------------------
# Few-shot example pool: first N sentences per region (never evaluated)
N_FEW_SHOT_EXAMPLES: int = 5

# Test set size per region (sentences evaluated in all 3 conditions)
MAX_SENTENCES_PER_REGION: int = 30   # 30x7 = 210 test sentences per condition

# Kept for backward compat with metrics.py / evaluate.py judge calls
GEMINI_REQUEST_DELAY: float = 4.5

# Judge scoring (Groq judge — deterministic)
JUDGE_TEMPERATURE: float  = 0.0

# Generation temperature — 0.0 for fully deterministic academic evaluation
GEN_TEMPERATURE: float    = 0.0

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
RESULTS_DIR: str = r"c:\Users\Rahul\Documents\Poster Nlp\results"
FIGURES_DIR: str = r"c:\Users\Rahul\Documents\Poster Nlp\figures"

FIGURE_DPI: int = 200   # >= 150 PPI required by examination constraints
