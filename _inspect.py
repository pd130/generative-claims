"""Quick helper to inspect profiled statistics."""
import json, sys
sys.path.insert(0, ".")

with open("data/processed/statistics.json") as f:
    s = json.load(f)

print("=== NUMERICAL COLUMNS ===")
for col, info in s["columns"].items():
    if info.get("type") == "numerical":
        d = info.get("distribution", {})
        print(f"  {col}: min={info['min']}, max={info['max']}, mean={info['mean']:.2f}, dist={d.get('name','?')}")

print("\n=== CATEGORICAL COLUMNS ===")
for col, info in s["columns"].items():
    if info.get("type") == "categorical":
        card = info.get("cardinality", "?")
        binary = info.get("is_binary", False)
        print(f"  {col}: cardinality={card}, binary={binary}")

print("\n=== FD SUMMARY ===")
with open("data/processed/functional_dependencies.json") as f:
    fd = json.load(f)

# group FDs by LHS  
from collections import defaultdict
fd_map = defaultdict(list)
for fdr in fd["exact"][:50]:
    lhs_key = ", ".join(fdr["lhs"])
    fd_map[lhs_key].append(fdr["rhs"])

for lhs, rhs_list in fd_map.items():
    print(f"  {lhs} -> {rhs_list}")
