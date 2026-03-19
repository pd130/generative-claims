"""Comprehensive test suite for dynamic prompt parsing & generation.

Covers: model allocation, percentages, exclusions, fuel, segment,
transmission, steering, brakes, safety, NCAP, age ranges, vehicle age
presets, subscription, region, claim rate edge cases, shorthand row
counts, multi-fuel, combined prompts, and adversarial inputs.

Run:  python test_dynamic.py
"""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(__file__))

from src.generator.synthetic_generator import SyntheticGenerator

gen = SyntheticGenerator()
PASS = 0
FAIL = 0


def run(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL += 1
        print(f"  FAIL  {name}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
# 1. MODEL ALLOCATION
# ═══════════════════════════════════════════════════════════════════════

def test_exact_model_count():
    df = gen.generate(n_rows=2000, prompt="2000 rows with 34 M3 model cars and rest divided among others", seed=42)
    m3 = (df["model"] == "M3").sum()
    assert m3 == 34, f"Expected 34 M3, got {m3}"
    assert len(df) == 2000
    assert df["model"].nunique() > 1, "Rest should include other models"
run("1. Exact model count (34 M3 of 2000)", test_exact_model_count)


def test_multi_model_counts():
    df = gen.generate(n_rows=1000, prompt="1000 rows with 200 M3 and 300 M4", seed=42)
    assert (df["model"] == "M3").sum() == 200
    assert (df["model"] == "M4").sum() == 300
    assert len(df) == 1000
run("2. Multiple model counts (200 M3 + 300 M4)", test_multi_model_counts)


def test_percentage_allocation():
    df = gen.generate(n_rows=1000, prompt="1000 rows 50% M3, 30% M4, 20% M1", seed=42)
    assert (df["model"] == "M3").sum() == 500
    assert (df["model"] == "M4").sum() == 300
    assert (df["model"] == "M1").sum() == 200
run("3. Percentage allocation (50/30/20)", test_percentage_allocation)


def test_exclusive_model():
    df = gen.generate(n_rows=500, prompt="only M3 500 rows", seed=42)
    assert list(df["model"].unique()) == ["M3"]
    assert len(df) == 500
run("4. Exclusive model (only M3)", test_exclusive_model)


def test_model_exclusion():
    df = gen.generate(n_rows=500, prompt="500 rows no M5 no M7", seed=42)
    assert "M5" not in df["model"].values
    assert "M7" not in df["model"].values
run("5. Model exclusion (no M5, no M7)", test_model_exclusion)


def test_rest_equally():
    df = gen.generate(n_rows=1100, prompt="1100 rows 100 M3 rest equally", seed=42)
    assert (df["model"] == "M3").sum() == 100
    others = df[df["model"] != "M3"]["model"].value_counts()
    assert others.max() - others.min() <= 1, "Equal split should be balanced"
run("6. Rest equally", test_rest_equally)


def test_allocation_exceeds_rows():
    """Edge: count exceeds n_rows → clamp to remaining."""
    df = gen.generate(n_rows=50, prompt="50 rows 100 M3", seed=42)
    assert len(df) == 50
    assert (df["model"] == "M3").sum() == 50  # clamped
run("7. Allocation exceeds n_rows (100 M3 in 50 rows)", test_allocation_exceeds_rows)


def test_model_colon_syntax():
    f = gen._parse_prompt("M3=200 M4:300")
    alloc = f.get("model_allocation", {})
    assert alloc.get("M3", {}).get("value") == 200
    assert alloc.get("M4", {}).get("value") == 300
run("8. M3=200 M4:300 syntax", test_model_colon_syntax)


# ═══════════════════════════════════════════════════════════════════════
# 2. FUEL TYPE
# ═══════════════════════════════════════════════════════════════════════

def test_diesel_filter():
    df = gen.generate(n_rows=300, prompt="300 rows diesel", seed=42)
    assert set(df["fuel_type"].unique()) == {"Diesel"}
run("9. Diesel filter", test_diesel_filter)


def test_cng_filter():
    df = gen.generate(n_rows=300, prompt="300 rows CNG vehicles", seed=42)
    assert set(df["fuel_type"].unique()) == {"CNG"}
run("10. CNG filter", test_cng_filter)


def test_multi_fuel():
    f = gen._parse_prompt("1000 rows petrol or diesel")
    assert "fuel_types" in f
    assert set(f["fuel_types"]) == {"Petrol", "Diesel"}
run("11. Multi-fuel (petrol or diesel) — parse", test_multi_fuel)


# ═══════════════════════════════════════════════════════════════════════
# 3. SEGMENT
# ═══════════════════════════════════════════════════════════════════════

def test_segment_b2():
    f = gen._parse_prompt("500 rows segment B2")
    assert f.get("segment") == "B2"
run("12. Segment B2 parse", test_segment_b2)


def test_utility_segment():
    f = gen._parse_prompt("utility vehicles 200 rows")
    assert f.get("segment") == "Utility"
run("13. Utility segment parse", test_utility_segment)


# ═══════════════════════════════════════════════════════════════════════
# 4. TRANSMISSION / STEERING / BRAKES
# ═══════════════════════════════════════════════════════════════════════

def test_automatic_transmission():
    f = gen._parse_prompt("1000 rows automatic transmission")
    assert f.get("transmission_type") == "Automatic"
run("14. Automatic transmission parse", test_automatic_transmission)


def test_manual_transmission():
    f = gen._parse_prompt("500 rows manual cars")
    assert f.get("transmission_type") == "Manual"
run("15. Manual transmission parse", test_manual_transmission)


def test_power_steering():
    f = gen._parse_prompt("power steering vehicles")
    assert f.get("steering_type") == "Power"
run("16. Power steering parse", test_power_steering)


def test_electric_steering():
    f = gen._parse_prompt("electric steering")
    assert f.get("steering_type") == "Electric"
run("17. Electric steering parse", test_electric_steering)


def test_disc_brakes():
    f = gen._parse_prompt("disc brakes 1000 rows")
    assert f.get("rear_brakes_type") == "Disc"
run("18. Disc brakes parse", test_disc_brakes)


def test_drum_brakes():
    f = gen._parse_prompt("drum brake vehicles")
    assert f.get("rear_brakes_type") == "Drum"
run("19. Drum brakes parse", test_drum_brakes)


# ═══════════════════════════════════════════════════════════════════════
# 5. SAFETY FEATURES
# ═══════════════════════════════════════════════════════════════════════

def test_with_parking_camera():
    f = gen._parse_prompt("with parking camera")
    assert "is_parking_camera" in f.get("safety_on", [])
run("20. With parking camera", test_with_parking_camera)


def test_without_esc():
    f = gen._parse_prompt("without esc 500 rows")
    assert "is_esc" in f.get("safety_off", [])
run("21. Without ESC", test_without_esc)


def test_with_brake_assist():
    f = gen._parse_prompt("with brake assist vehicles")
    assert "is_brake_assist" in f.get("safety_on", [])
run("22. With brake assist", test_with_brake_assist)


def test_without_tpms():
    f = gen._parse_prompt("without tpms")
    assert "is_tpms" in f.get("safety_off", [])
run("23. Without TPMS", test_without_tpms)


# ═══════════════════════════════════════════════════════════════════════
# 6. NCAP RATING
# ═══════════════════════════════════════════════════════════════════════

def test_ncap_4plus():
    f = gen._parse_prompt("ncap 4+ vehicles 500 rows")
    assert f.get("ncap_min") == 4
run("24. NCAP 4+", test_ncap_4plus)


def test_5_star():
    f = gen._parse_prompt("5 star rated cars")
    assert f.get("ncap_min") == 5
run("25. 5 star rated", test_5_star)


# ═══════════════════════════════════════════════════════════════════════
# 7. AGE RANGES
# ═══════════════════════════════════════════════════════════════════════

def test_age_range_explicit():
    df = gen.generate(n_rows=200, prompt="200 rows age 40-60", seed=42)
    assert df["customer_age"].min() >= 40
    assert df["customer_age"].max() <= 60
run("26. Explicit age range 40-60", test_age_range_explicit)


def test_young_drivers():
    df = gen.generate(n_rows=200, prompt="200 rows young drivers", seed=42)
    assert df["customer_age"].max() <= 35
run("27. Young drivers (<=35)", test_young_drivers)


def test_senior_drivers():
    df = gen.generate(n_rows=200, prompt="200 rows senior drivers", seed=42)
    assert df["customer_age"].min() >= 55
run("28. Senior drivers (>=55)", test_senior_drivers)


def test_middle_aged():
    f = gen._parse_prompt("middle aged drivers 500 rows")
    assert f.get("age_min") == 35
    assert f.get("age_max") == 50
run("29. Middle aged (35-50)", test_middle_aged)


# ═══════════════════════════════════════════════════════════════════════
# 8. VEHICLE AGE PRESETS
# ═══════════════════════════════════════════════════════════════════════

def test_new_cars():
    f = gen._parse_prompt("new cars 200 rows")
    assert f.get("vehicle_age_min") == 0
    assert f.get("vehicle_age_max") == 2
run("30. New cars (vehicle age 0-2)", test_new_cars)


def test_old_cars():
    f = gen._parse_prompt("old car 200 rows")
    assert f.get("vehicle_age_min") == 8
    assert f.get("vehicle_age_max") == 20
run("31. Old cars (vehicle age 8-20)", test_old_cars)


def test_used_cars():
    f = gen._parse_prompt("used car 200 rows")
    assert f.get("vehicle_age_min") == 8
run("32. Used cars (= old cars)", test_used_cars)


def test_vehicle_age_range():
    f = gen._parse_prompt("vehicle age 3-7")
    assert f.get("vehicle_age_min") == 3
    assert f.get("vehicle_age_max") == 7
run("33. Vehicle age range 3-7", test_vehicle_age_range)


# ═══════════════════════════════════════════════════════════════════════
# 9. SUBSCRIPTION LENGTH
# ═══════════════════════════════════════════════════════════════════════

def test_subscription_range():
    f = gen._parse_prompt("subscription 5-10 200 rows")
    assert f.get("subscription_min") == 5
    assert f.get("subscription_max") == 10
run("34. Subscription range 5-10", test_subscription_range)


# ═══════════════════════════════════════════════════════════════════════
# 10. REGION
# ═══════════════════════════════════════════════════════════════════════

def test_region_filter():
    f = gen._parse_prompt("region C5 500 rows")
    assert f.get("region_code") == "C5"
run("35. Region C5", test_region_filter)


# ═══════════════════════════════════════════════════════════════════════
# 11. CLAIM RATE EDGE CASES
# ═══════════════════════════════════════════════════════════════════════

def test_high_risk():
    f = gen._parse_prompt("high risk 500 rows")
    assert f.get("claim_rate_override") == 0.15
run("36. High risk -> 15%", test_high_risk)


def test_low_risk():
    f = gen._parse_prompt("low risk 500 rows")
    assert f.get("claim_rate_override") == 0.02
run("37. Low risk -> 2%", test_low_risk)


def test_exact_claim_pct():
    df = gen.generate(n_rows=1000, claim_rate=0.12, prompt="1000 rows 12% claim rate", seed=42)
    actual = df["claim_status"].mean()
    assert abs(actual - 0.12) < 0.001, f"Expected 12%, got {actual:.4f}"
run("38. Exact 12% claim rate", test_exact_claim_pct)


def test_zero_claims():
    f = gen._parse_prompt("500 rows no claims")
    assert f.get("claim_rate_override") == 0.0
    df = gen.generate(n_rows=100, claim_rate=0.0, seed=42)
    assert df["claim_status"].sum() == 0
run("39. Zero claims (0%)", test_zero_claims)


def test_all_claims():
    f = gen._parse_prompt("all claims 200 rows")
    assert f.get("claim_rate_override") == 1.0
    df = gen.generate(n_rows=100, claim_rate=1.0, seed=42)
    assert df["claim_status"].sum() == 100
run("40. All claims (100%)", test_all_claims)


# ═══════════════════════════════════════════════════════════════════════
# 12. ROW COUNT SHORTHAND
# ═══════════════════════════════════════════════════════════════════════

def test_1k_rows():
    f = gen._parse_prompt("1k rows diesel")
    assert f.get("n_rows") == 1000
run("41. 1k rows shorthand", test_1k_rows)


def test_2_5k_rows():
    f = gen._parse_prompt("2.5k rows petrol")
    assert f.get("n_rows") == 2500
run("42. 2.5k rows shorthand", test_2_5k_rows)


def test_generate_verb():
    f = gen._parse_prompt("generate 3000 diesel")
    assert f.get("n_rows") == 3000
run("43. 'generate 3000' verb", test_generate_verb)


# ═══════════════════════════════════════════════════════════════════════
# 13. COMBINED PROMPTS
# ═══════════════════════════════════════════════════════════════════════

def test_combined_complex():
    prompt = "3000 rows 100 M3 diesel young drivers 10% claim rate"
    df = gen.generate(n_rows=3000, claim_rate=0.10, prompt=prompt, seed=42)
    assert (df["model"] == "M3").sum() == 100
    assert abs(df["claim_status"].mean() - 0.10) < 0.001
    assert df["customer_age"].max() <= 35
run("44. Combined: 100 M3 + diesel + young + 10% claim", test_combined_complex)


def test_combined_segment_brakes():
    f = gen._parse_prompt("1000 rows segment B2 disc brakes automatic")
    assert f.get("n_rows") == 1000
    assert f.get("segment") == "B2"
    assert f.get("rear_brakes_type") == "Disc"
    assert f.get("transmission_type") == "Automatic"
run("45. Combined: segment B2 + disc brakes + automatic", test_combined_segment_brakes)


def test_combined_safety_ncap():
    f = gen._parse_prompt("500 rows with parking camera ncap 4+ new cars")
    assert f.get("n_rows") == 500
    assert "is_parking_camera" in f.get("safety_on", [])
    assert f.get("ncap_min") == 4
    assert f.get("vehicle_age_max") == 2
run("46. Combined: parking camera + NCAP 4+ + new cars", test_combined_safety_ncap)


def test_combined_exclusion_fuel_age():
    f = gen._parse_prompt("2000 rows no M5 no M7 petrol age 45-65")
    assert "M5" in f.get("exclude_models", set())
    assert "M7" in f.get("exclude_models", set())
    assert f.get("fuel_type") == "Petrol"
    assert f.get("age_min") == 45
    assert f.get("age_max") == 65
run("47. Combined: exclusions + petrol + age range", test_combined_exclusion_fuel_age)


# ═══════════════════════════════════════════════════════════════════════
# 14. EDGE / ADVERSARIAL CASES
# ═══════════════════════════════════════════════════════════════════════

def test_empty_prompt():
    f = gen._parse_prompt("")
    assert f == {}
run("48. Empty prompt -> no filters", test_empty_prompt)


def test_nonsense_prompt():
    f = gen._parse_prompt("lorem ipsum dolor sit amet")
    assert "model" not in f
    assert "fuel_type" not in f
run("49. Nonsense prompt -> no filters", test_nonsense_prompt)


def test_invalid_model_name():
    """M99 doesn't exist - should still generate data from other models."""
    df = gen.generate(n_rows=100, prompt="100 rows 50 M99", seed=42)
    assert len(df) == 100
    # M99 doesn't exist so no M99 rows, rest distributed among all
run("50. Invalid model M99 -> graceful fallback", test_invalid_model_name)


def test_single_row():
    df = gen.generate(n_rows=1, seed=42)
    assert len(df) == 1
    assert "model" in df.columns
run("51. Single row generation", test_single_row)


def test_10k_rows():
    df = gen.generate(n_rows=10000, seed=42)
    assert len(df) == 10000
run("52. 10,000 rows generation", test_10k_rows)


def test_data_schema_integrity():
    """All 41 columns should be present."""
    df = gen.generate(n_rows=100, seed=42)
    expected = 41
    assert len(df.columns) == expected, f"Expected {expected} columns, got {len(df.columns)}"
    assert "policy_id" in df.columns
    assert "claim_status" in df.columns
    assert "model" in df.columns
run("53. Schema integrity (41 columns)", test_data_schema_integrity)


def test_no_nulls():
    df = gen.generate(n_rows=500, seed=42)
    nulls = df.isnull().sum().sum()
    assert nulls == 0, f"Found {nulls} null values"
run("54. No null values", test_no_nulls)


def test_claim_status_binary():
    df = gen.generate(n_rows=1000, seed=42)
    assert set(df["claim_status"].unique()).issubset({0, 1})
run("55. Claim status is binary (0/1)", test_claim_status_binary)


def test_reproducibility():
    df1 = gen.generate(n_rows=100, seed=123)
    df2 = gen.generate(n_rows=100, seed=123)
    assert df1.equals(df2), "Same seed should produce identical data"
run("56. Reproducibility (same seed = same data)", test_reproducibility)


def test_different_seeds():
    df1 = gen.generate(n_rows=100, seed=1)
    df2 = gen.generate(n_rows=100, seed=2)
    assert not df1.equals(df2), "Different seeds should produce different data"
run("57. Different seeds -> different data", test_different_seeds)


def test_non_negative_values():
    df = gen.generate(n_rows=500, seed=42)
    for col in ["vehicle_age", "customer_age", "subscription_length", "displacement", "airbags"]:
        if col in df.columns:
            mn = df[col].min()
            assert mn >= 0, f"{col} has negative value {mn}"
run("58. Non-negative numerical values", test_non_negative_values)


def test_helper_methods():
    models = gen.get_available_models()
    assert len(models) == 11
    assert "M1" in models
    segments = gen.get_available_segments()
    assert "Utility" in segments or "utility" in [s.lower() for s in segments]
    fuels = gen.get_available_fuel_types()
    assert len(fuels) == 3
run("59. Helper methods (models/segments/fuels)", test_helper_methods)


def test_percentages_over_100():
    """70%+50% = 120% -> overshoot, should clamp gracefully."""
    df = gen.generate(n_rows=100, prompt="100 rows 70% M3 50% M4", seed=42)
    assert len(df) == 100
    # M3 gets 70, M4 gets up to 30 (clamped)
    assert (df["model"] == "M3").sum() == 70
run("60. Percentages >100% -> clamped", test_percentages_over_100)


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print(f"  RESULTS:  {PASS} passed  /  {FAIL} failed  /  {PASS + FAIL} total")
print("=" * 64)

if FAIL > 0:
    sys.exit(1)
else:
    print("\n  ALL 60 TESTS PASSED!\n")
    sys.exit(0)
