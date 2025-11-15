#!/bin/sh
poetry run python scripts/hybrid_tuning.py --samples-per-dim 500 --dims 2 3 --extreme-fraction 1.00 --plot-dir docs/ --plot-prefix extreme- --output docs/hybrid_tuning_summary_extreme.json > docs/hybrid_tuning_summary_extreme.md
poetry run python scripts/hybrid_tuning.py --samples-per-dim 500 --dims 2 3 --extreme-fraction 0.00 --plot-dir docs/ --plot-prefix nonextreme- --output docs/hybrid_tuning_summary_nonextreme.json > docs/hybrid_tuning_summary_nonextreme.md
# poetry run python scripts/hybrid_tuning.py --samples-per-dim 500 --dims 10 20 50 --extreme-fraction 1.00 --plot-dir docs/ --plot-prefix extreme-hd- --output docs/hybrid_tuning_summary_extreme-hd.json
# poetry run python scripts/hybrid_tuning.py --samples-per-dim 500 --dims 10 20 50 --extreme-fraction 0.00 --plot-dir docs/ --plot-prefix nonextreme-hd- --output docs/hybrid_tuning_summary_nonextreme-hd.json
