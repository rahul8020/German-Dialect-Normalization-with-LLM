"""
End-to-end smoke test: runs all 3 conditions on 3 real Betthupferl sentences.
No judge calls. Shows WER reduction from Baseline-0 -> Zero-Shot -> Few-Shot.
"""
import logging
import sys
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# Force UTF-8 output so German characters don't crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from data_loader import load_all_regions
from conditions import run_identity, run_zero_shot, run_few_shot
from metrics import compute_metrics_from_df

# Load 3 test sentences from oberbayern only (fast)
examples, test = load_all_regions(regions=["oberbayern"])
test3 = test.head(3)

print("=" * 70)
print("INPUT (Bavarian dialect):")
for i, row in test3.iterrows():
    print(f"  [{i+1}] {row['dialect']}")

print("\nREFERENCE (Standard German):")
for i, row in test3.iterrows():
    print(f"  [{i+1}] {row['reference']}")

print("\n--- Baseline-0 (Identity) ---")
b0 = run_identity(test3)
m0 = compute_metrics_from_df(b0)
print(f"  WER={m0['wer']:.3f}  CER={m0['cer']:.3f}  BLEU={m0['bleu']:.1f}")

print("\n--- Condition-A (Zero-Shot) ---")
ca = run_zero_shot(test3, request_delay=5)
for i, (_, row) in enumerate(ca.iterrows(), 1):
    print(f"  [{i}] {row['hypothesis']}")
ma = compute_metrics_from_df(ca)
print(f"  WER={ma['wer']:.3f}  CER={ma['cer']:.3f}  BLEU={ma['bleu']:.1f}")

print("\n--- Condition-B (Few-Shot) ---")
cb = run_few_shot(test3, examples, request_delay=5)
for i, (_, row) in enumerate(cb.iterrows(), 1):
    print(f"  [{i}] {row['hypothesis']}")
mb = compute_metrics_from_df(cb)
print(f"  WER={mb['wer']:.3f}  CER={mb['cer']:.3f}  BLEU={mb['bleu']:.1f}")

print("\n" + "=" * 70)
print(f"WER: {m0['wer']:.3f} -> {ma['wer']:.3f} -> {mb['wer']:.3f}")
print(f"Total WER reduction: {(m0['wer'] - mb['wer'])*100:.1f} percentage points")
