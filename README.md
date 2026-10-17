# InspectEdge — normal-image inspection baseline

A CPU research station with a handcrafted patch nearest-neighbor detector, computed anomaly maps, durable local inspection records, and retryable synchronization. **Not a completed factory validation or neural anomaly-detection study.**

## Run

```bash
python -m venv .venv
# Activate .venv.
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m uvicorn server:app --host 127.0.0.1 --port 8023
```

Open http://127.0.0.1:8023. Without a fitted artifact the application explicitly requires setup; it does not return invented anomalies.

## Fit and evaluate

Obtain a permitted MVTec AD category from https://www.mvtec.com/research-teaching/datasets/mvtec-ad . Respect the noncommercial dataset license. No dataset is bundled or downloaded automatically.

```bash
python detector.py --category /path/to/category --mode mvtec --size 128 --memory 1000
```

The expected layout is `train/good/*.png` and `test/<defect-or-good>/*.png`. Normal training images are split into fit and validation sets. The normal-validation 95th percentile fixes the image threshold; test anomaly labels never tune it. The model standardizes handcrafted RGB/gradient descriptors, subsamples a feature memory, and scores nearest-neighbor patch distances. It is not PatchCore or a new algorithm. Duplicate normal images are rejected.

The command saves `artifacts/detector.joblib` and `artifacts/evaluation.json`. Reported metrics are image AUROC, average precision, normal false-alert rate, anomaly detection rate and per-image latency with raw observations. No pixel metric or energy claim is made. Compare multiple sizes/memory settings with a prespecified protocol before interpreting the final test set. Synthetic fixtures must use `--mode synthetic_fixture` and remain visibly labelled.

## Durable synchronization

`INSPECT_DB` selects the local SQLite DB; `INSPECT_MODEL` selects a trusted local model artifact. Never load unknown joblib files. `INSPECT_KEY` defaults to `local-inspect-only` for localhost demonstrations. Run a second instance with a separate DB and set the edge instance's `INSPECT_CENTRAL_URL` to that central service. The UI Sync action sends at most 20 pending records. Central IDs are idempotent and conflicting payloads are rejected. Acknowledged records are marked only after successful delivery. No raw uploaded images are stored; an anomaly map and image hash are retained.

The shared-key single-station demo is not production device authentication. Keep it on localhost. Historical central counts do not prove station liveness.

## Verification boundary

Tests exercise normal-only fitting, reproducibility, fit/validation separation, duplicate-image rejection, missing-model behavior and ingestion retries. Real MVTec images were not available during implementation. No factory detection score, pretrained feature extractor, camera throughput, neural comparison, or complete offline failure study is claimed.

Remaining: three-category MVTec evaluation, a properly attributed pretrained patch-feature method, controlled compression experiments, model deployment/rollback, human review, multi-station identity and fault testing.

Original baseline implementation with AI assistance. Uses NumPy, scikit-learn, Pillow, FastAPI and HTTPX under their respective licenses. Dataset terms apply separately.
