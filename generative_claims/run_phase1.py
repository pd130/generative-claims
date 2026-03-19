"""Run the full Phase 1 pipeline: Profile → FD Discovery → Schema."""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.profiler.statistical_profiler import StatisticalProfiler
from src.profiler.fd_discovery import FDDiscovery
from src.validator.schema_validator import SchemaValidator
from src.utils.config import get_settings


def main() -> None:
    settings = get_settings()

    # 1. Statistical Profiler (skip if already done)
    print("=" * 60)
    print("Step 1: Statistical Profiler")
    print("=" * 60)
    if settings.statistics_path.exists():
        print(f"  [SKIP] statistics.json already exists.")
    else:
        profiler = StatisticalProfiler()
        stats = profiler.run()
        print(f"  Profiled {len(stats['columns'])} columns.")
    print()

    # 2. FD Discovery (skip if already done)
    print("=" * 60)
    print("Step 2: FD Discovery")
    print("=" * 60)
    if settings.fd_path.exists():
        print(f"  [SKIP] functional_dependencies.json already exists.")
    else:
        fd = FDDiscovery()
        fd_result = fd.run()
        print(FDDiscovery.summarize(fd_result))
    print()

    # 3. Schema Validator (skip if already done)
    print("=" * 60)
    print("Step 3: Schema Validator")
    print("=" * 60)
    if settings.schema_path.exists():
        print(f"  [SKIP] schema.yaml already exists.")
    else:
        sv = SchemaValidator()
        sv.generate_schema()
        print(f"  Schema generated and saved.")
    print()

    print("Phase 1 complete! Outputs:")
    print(f"  - {settings.statistics_path}")
    print(f"  - {settings.fd_path}")
    print(f"  - {settings.schema_path}")


if __name__ == "__main__":
    main()
