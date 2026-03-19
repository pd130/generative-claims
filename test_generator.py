"""Quick test of the synthetic generator."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.generator.synthetic_generator import SyntheticGenerator

gen = SyntheticGenerator()

# Basic generation
print("=== Basic generation (100 rows) ===")
df = gen.generate(n_rows=100)
print(df.head(5).to_string())
print(f"\nShape: {df.shape}")
print(f"Claim rate: {df['claim_status'].mean():.4f}")
print(f"Models: {df['model'].unique()}")
print(f"Fuel types: {df['fuel_type'].unique()}")

# Prompt-based generation
print("\n=== Prompt: 'Generate 50 diesel vehicles' ===")
df2 = gen.generate_from_prompt("Generate 50 diesel vehicles", seed=123)
print(f"Shape: {df2.shape}")
print(f"Fuel types: {df2['fuel_type'].unique()}")
print(f"Claim rate: {df2['claim_status'].mean():.4f}")

print("\n=== Prompt: 'young drivers high risk 30 rows' ===")
df3 = gen.generate_from_prompt("young drivers high risk 30 rows", seed=456)
print(f"Shape: {df3.shape}")
print(f"Age range: {df3['customer_age'].min()} - {df3['customer_age'].max()}")
print(f"Claim rate: {df3['claim_status'].mean():.4f}")

print("\nGenerator test complete!")
