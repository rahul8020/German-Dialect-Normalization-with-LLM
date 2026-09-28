import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# Model used for dialect normalisation and judging
GROQ_MODEL: str = "qwen/qwen3.8-27b"

# Groq free tier request delay
GROQ_REQUEST_DELAY: float = 2.0   


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

GEN_TEMPERATURE: float    = 0.0


# Output
RESULTS_DIR: str = r"c:\Users\Rahul\Documents\Poster Nlp\results"
FIGURES_DIR: str = r"c:\Users\Rahul\Documents\Poster Nlp\figures"

FIGURE_DPI: int = 200 
