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

Tests exercise normal-only fitting, reproducibility, fit/validation separation, duplicate-image rejection, missing-model behavior and ingestion retries. MVTec has not been evaluated; the subsequent real-data study uses VisA (see below). No factory deployment, camera-throughput or complete offline failure study is claimed.

Remaining: independent MVTec evaluation, controlled compression experiments, model deployment/rollback, human review, multi-station identity and fault testing.

Original baseline implementation with AI assistance. Uses NumPy, scikit-learn, Pillow, FastAPI and HTTPX under their respective licenses. Dataset terms apply separately.

## Executed real-data study: VisA

The original VisA archive was downloaded and SHA256 verified. Evaluated all official test images in **cashew (150), capsules (160), pcb1 (200)**, with training restricted to their official normal-image training sets. See [protocol](experiments/VISA_PROTOCOL.md), [source attribution](evidence/DATA_SOURCES.json), [all handcrafted settings](evidence/visa/evaluation.json) and [exploratory pretrained comparison](evidence/visa_neural/evaluation.json). Compressed raw predictions and image-hash manifests are alongside each report.

| Category | Handcrafted AUROC, 128/1000 | Frozen ResNet18 patch AUROC | Neural detection rate | Neural false-alert rate |
|---|---:|---:|---:|---:|
| Cashew | 0.671 | 0.857 | 61% | 6% |
| Capsules | 0.645 | 0.684 | 22% | 3.3% |
| PCB1 | 0.737 | 0.872 | 59% | 14% |

Thresholds were set on normal validation images, not test anomalies. The capsule miss rate is substantial. AUROC is not detection accuracy. Neural results are exploratory follow-ups using the same test sets; independent confirmation is still needed. Reported model file bytes cover the patch-memory artifact before embedding report metadata; the separate ResNet18 backbone is approximately 45 MB and must also be counted for deployment.

Reproduce with Python 3.12:

```bash
python experiments/download_visa.py --output data/visa
python experiments/visa_benchmark.py --root data/visa/images --split data/visa/1cls.csv
python -m pip install -r requirements-neural.txt --extra-index-url https://download.pytorch.org/whl/cpu
python experiments/visa_benchmark.py --neural --root data/visa/images --split data/visa/1cls.csv --output artifacts/visa_neural
```

Select a category-specific model explicitly, for example set `INSPECT_MODEL` to `artifacts/visa_neural/cashew_128_1000.joblib` before starting the server. A cashew model is not a general-purpose factory inspector. Optional neural inference requires official pretrained weights, downloaded by torchvision on first use.

### Executed HTTP recovery check

`experiments/http_recovery.py` uploads a real image three times through the trained-model API, confirms local retention while central is unavailable, kills/restarts the edge process, then synchronizes to a separate central process. All three records were retained and their full payloads matched centrally; a repeated sync acknowledged zero new records. See [raw evidence](evidence/http_recovery.json). These are three uploads of one image, not three independent anomaly tests.
