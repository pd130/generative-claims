"""Test exact claim rate and synthetic data uniqueness."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from src.generator.synthetic_generator import SyntheticGenerator
from src.utils.config import get_settings

gen = SyntheticGenerator()
settings = get_settings()

# Test 1: Exact claim rate
print("=" * 60)
print("TEST 1: Exact claim rate")
print("=" * 60)
for target in [0.02, 0.06, 0.10, 0.15]:
    for n in [456, 1000, 100]:
        df = gen.generate(n_rows=n, claim_rate=target, seed=42)
        actual = df["claim_status"].mean()
        expected_count = int(round(n * target))
        actual_count = df["claim_status"].sum()
        status = "PASS" if actual_count == expected_count else "FAIL"
        print(f"  [{status}] n={n}, target={target:.2%}, "
              f"expected_claims={expected_count}, actual_claims={actual_count}, "
              f"actual_rate={actual:.4%}")

# Test 2: Prompt with "2% claim rate"
print()
print("=" * 60)
print("TEST 2: Prompt-based claim rate")
print("=" * 60)
df2 = gen.generate_from_prompt("generate 456 rows data with 2% claim rate", seed=42)
n_claims = df2["claim_status"].sum()
print(f"  Rows: {len(df2)}")
print(f"  Claims: {n_claims}")
print(f"  Rate: {df2['claim_status'].mean():.4%}")
print(f"  Expected claims: {int(round(456 * 0.02))} = {round(456*0.02)}")

# Test 3: Synthetic data is NOT same as real data
print()
print("=" * 60)
print("TEST 3: Data is truly synthetic (not copied)")
print("=" * 60)
real_df = pd.read_csv(settings.raw_data_path)
synth = gen.generate(n_rows=500, seed=123)

# Check: no synthetic row should exactly match any real row
# Compare on key columns that vary
compare_cols = ["subscription_length", "vehicle_age", "customer_age",
                "region_code", "model", "airbags", "displacement"]
real_tuples = set(real_df[compare_cols].apply(tuple, axis=1))
synth_tuples = set(synth[compare_cols].apply(tuple, axis=1))
overlap = real_tuples & synth_tuples
print(f"  Real unique rows: {len(real_tuples)}")
print(f"  Synth unique rows: {len(synth_tuples)}")
print(f"  Exact matches: {len(overlap)}")
print(f"  Match rate: {len(overlap)/len(synth_tuples):.2%}")
print(f"  {'PASS' if len(overlap) / len(synth_tuples) < 0.05 else 'FAIL'}: "
      f"< 5% overlap means data is truly synthetic")

# Also check that FD-determined numerical columns have been perturbed
print()
print("  Perturbation check (airbags, displacement columns):")
real_m4 = real_df[real_df["model"] == "M4"].iloc[0]
synth_m4 = synth[synth["model"] == "M4"]
if len(synth_m4) > 0:
    s = synth_m4.iloc[0]
    for col in ["displacement", "gross_weight", "length", "width"]:
        if col in synth_m4.columns:
            real_val = real_m4[col]
            synth_vals = synth_m4[col].unique()
            print(f"    {col}: real={real_val}, synth_unique_vals={synth_vals[:5]}")

print()
print("All tests done!")
