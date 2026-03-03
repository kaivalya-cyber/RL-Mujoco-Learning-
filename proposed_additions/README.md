# Proposed Additions (Staged)

This folder contains staged additions to integrate later:

- `evaluate.py`: standardized checkpoint evaluation over N episodes
- `generate_report.py`: single consolidated benchmark report (PNG + PDF)
- `tests/test_smoke.py`: smoke tests (imports, env reset/step, checkpoint load, short train)
- `workflows/ci.yml`: GitHub Actions CI workflow template
- `results/`: sample generated metrics + report outputs

These are intentionally isolated from the root project for review/finalization before full integration.
