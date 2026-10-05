# Hyperion

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-137B80?style=for-the-badge)
![Status](https://img.shields.io/badge/status-active%20research-E36209?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-3C9A5F?style=for-the-badge)

![Hyperion](https://www.mycelliumlab.com/assets/hyperion-social.jpg)

**Urban compound flood intelligence. The next six hours, not the next thirty years.**

Hyperion studies compound urban flooding for Mycellium Lab. This repository documents the method and an offline demo; the full implementation is private.

**[Project page on Mycellium Lab](https://www.mycelliumlab.com/hyperion)**

## Research scope

The system combines historical event analysis, coarse-grid scores, leakage checks and operator workflows. Live NHC/NWS feeds describe hazards; historical scores are not live forecasts. The demo uses synthetic data, not surveyed flood truth.

![Hyperion public research product homepage, captured 2026-10-05. Illustrative and historical evidence; not forecast or backend validation.](docs/screenshots/hyperion-public.jpg)

Phase 1 focuses on Florida, Big Bend and Louisiana. Recife is closed. The service uses Python's standard-library HTTP server; scikit-learn and XGBoost support research workflows.

## Offline demo

Python 3.11+; no network or API key required:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
PYTHONPATH=src python -m hyperion_demo.demo
PYTHONPATH=src python -m pytest -q
```

The demo runs Hyperion's leakage inspection, temporal split and calibration metrics on a frozen CC0-1.0 synthetic fixture. It verifies the fixture hash and keeps test timestamps and labels out of calibration.

Source modules and their original revision are recorded in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json). Methods are open for scrutiny. This repository does not provide commercial service, parcel-level scoring or live flood forecasting. See [LICENSE](LICENSE).

**Cesar Augusto** · [GitHub](https://github.com/cesaremcasa) · [LinkedIn](https://www.linkedin.com/in/cesar-augusto-22943a351/) · [korvo.dev](https://korvo.dev)
