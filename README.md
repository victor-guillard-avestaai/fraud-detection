# Credit Card Fraud Detection on GCP

End-to-end credit card fraud detection project built on the Kaggle “Credit Card Fraud Detection” dataset.  
It combines:

- A reproducible offline training pipeline (Python / scikit-learn / XGBoost)
- Cost-aware decision thresholding for an imbalanced classification setting
- A production-ready FastAPI service on Google Cloud Run
- BigQuery logging for monitoring
- A small interactive UI that lets you explore real test-set transactions and see the model’s decisions

Conceptually inspired by **Le Borgne et al., _Reproducible Machine Learning for Credit Card Fraud Detection – Practical Handbook_**, but all implementation is my own.

---

## 1. Problem & Dataset

- **Goal:** detect fraudulent credit card transactions.
- **Dataset:** Kaggle _Credit Card Fraud Detection_ (European card transactions), with:
  - 284,807 transactions
  - 492 frauds (~0.17% positive class)
  - Features:
    - `Time`, `Amount`
    - 28 anonymized PCA components: `V1`…`V28`
    - Target: `Class` (1 = fraud, 0 = genuine)

This is a classic **severely imbalanced, cost-sensitive** classification problem:  
missing a fraud (FN) is much worse than flagging a legitimate transaction (FP).

---

## 2. Modeling Approach

### 2.1 Temporal validation

To avoid leakage from the future, the data is split **by time**, not randomly:

- Sort transactions by `Time`
- Split into **train / validation / test** as:
  - 60% earliest transactions → **train**
  - next 20% → **validation**
  - last 20% → **test**

Splits are deterministic, so results are reproducible.

### 2.2 Models

Two models are trained for comparison:

- **Baseline:** Logistic Regression (with standardization via `StandardScaler`)
- **Main model:** XGBoost classifier

XGBoost hyperparameters are found via a small **random search** on the validation set, optimizing **PR-AUC** (Precision–Recall AUC), while also tracking:

- ROC-AUC
- Precision@Top-100 (P@100): precision among the 100 highest-scored transactions

No resampling (SMOTE, under/oversampling) and no cost-sensitive training are used.  
This follows the handbook’s observation that, for tree-based models like XGBoost in this setting, such techniques can improve AUROC but often degrade **Average Precision** and **Precision@Top-k**, which is what matters most operationally.

### 2.3 Metrics

Primary metrics:

- **PR-AUC** on the **test** set
- **Precision@Top-100** (P@100): precision among the 100 highest-scored transactions

Secondary metrics:

- ROC-AUC
- Precision / Recall / F1 at the chosen decision threshold

---

## 3. Cost-Sensitive Decision Threshold

Rather than using a naive 0.5 threshold, the project explicitly models the **relative cost** of false negatives vs false positives and chooses the threshold that minimizes expected cost on the validation set.

Assumed unit costs:

- **FN (missed fraud):** `C_FN = 1.0`
- **FP (false alert):** `C_FP = 0.05`  
  (raising an alert on a legit transaction is 20× cheaper than missing a fraud)

For each candidate threshold `t` in `[0, 1]`:

1. Convert probabilities to labels: `fraud_prob >= t → predict fraud`.
2. Compute the confusion matrix.
3. Compute expected cost per transaction:

   \[
   \text{ExpectedCost}(t) = \frac{C*{FN} \cdot FN(t) + C*{FP} \cdot FP(t)}{N}
   \]

The best threshold is the one minimizing this expected cost on the **validation** set.

The optimal threshold (≈ **0.044**) and the cost matrix are stored in `metadata.json` and used by the API.

---

## 4. Results (Test Set)

All metrics below are computed on the **held-out test split** (last 20% of transactions in time).

### 4.1 Baseline – Logistic Regression

- **PR-AUC:** ≈ **0.69**
- **ROC-AUC:** ≈ **0.97**

Serves as a simple, strong linear baseline.

### 4.2 XGBoost – Ranking performance (no threshold)

- **PR-AUC:** ≈ **0.79**
- **ROC-AUC:** ≈ **0.98**

Significantly better ranking performance than Logistic Regression, especially in the high-recall, high-precision region.

### 4.3 XGBoost – Cost-optimized threshold

At the chosen threshold **t ≈ 0.044** (with `C_FN = 1`, `C_FP = 0.05`):

- **Precision:** ≈ **0.69**
- **Recall:** ≈ **0.79**
- **F1:** ≈ **0.73**
- **ROC-AUC:** ≈ **0.98**
- **Precision@100:** ≈ **0.59**  
  → among the 100 most suspicious transactions, ~59 are true frauds.
- **Expected cost per transaction:** ≈ **3.05 × 10⁻⁴**

Given the base fraud rate (~0.17%), a P@100 around 0.59 is a large lift over random.

---

## 5. Offline Pipeline

The offline training pipeline lives in the `model/` package.

Key script:

- `model/train.py`

It:

1. Loads raw data from `data/raw/creditcard.csv`.
2. Performs deterministic temporal split into train/val/test.
3. Trains the **baseline** Logistic Regression pipeline.
4. Runs a small random search to train the **XGBoost** pipeline.
5. Evaluates both models on validation and test.
6. Searches the **cost-optimal threshold** on the validation set.
7. Computes final metrics on the test set.
8. Saves artifacts to `model/artifacts/`:
   - `model.joblib` — full sklearn/XGBoost pipeline
   - `metadata.json` — JSON containing:
     - `model_version` (timestamp)
     - `model_type`
     - `features` (ordered list of feature names)
     - `threshold`
     - `cost_matrix` (`FN`, `FP`)
     - `metrics` (validation + test)
9. Exports a small set of real **test-set transactions** to `model/demo_samples.json`:
   - A subset of fraud examples and genuine examples
   - Each with a full feature dict + true label
   - Used by the demo UI

Run it locally:

```bash
# Install dev dependencies
uv sync --extra dev

# Train models and generate artifacts
uv run python -m model.train


```

## 6. Online Inference API

The online service is implemented with FastAPI under `api/`.

### 6.1 Main endpoints

- `GET /health`  
  Returns a simple JSON with:

  {
  "status": "ok",
  "project_id": "...",
  "platform": "loc" | "prod",
  "bq_dataset": "fraud",
  "bq_table_predictions": "predictions",
  "model_path": "...",
  "model_version": "...",
  "threshold": 0.044...
  }

  Used to verify that the correct model and configuration are loaded in each environment.

- `POST /predict`  
  Request body (`TransactionInput`) mirrors the Kaggle feature space:

  {
  "Time": 12345.0,
  "V1": 0.1,
  "V2": -1.2,
  "...": "...",
  "V28": 0.05,
  "Amount": 149.62
  }

  Response (`PredictionResponse`):

  {
  "fraud_prob": 0.7867,
  "fraud_flag": true,
  "model_version": "20251208_153248",
  "threshold": 0.0440881763
  }

  - `fraud_prob` is the estimated probability of fraud (output of `predict_proba` from the pipeline).
  - `fraud_flag` applies the cost-optimized threshold from `metadata.json`.
  - `model_version` and `threshold` come from the same `metadata.json` that was created at training time.

### 6.2 Demo endpoints

- `GET /demo-samples`  
  Returns the curated set of demo samples exported from the test set. Each sample looks like:

  {
  "id": "fraud_001",
  "label": "Fraud example 1",
  "class": 1,
  "features": {
  "Time": 12345.0,
  "V1": ...,
  "...": "...",
  "V28": ...,
  "Amount": 149.62
  }
  }

- `GET /demo`  
  Serves a small static HTML+JS UI that:

  - Loads demo samples from `/demo-samples`
  - Lets the user pick a fraud or genuine example
  - Optionally tweak `Amount` and `Time`
  - Calls `/predict` with the full feature vector
  - Displays:
    - Fraud probability (percentage)
    - Decision threshold (percentage)
    - Model decision (FLAGGED AS FRAUD / NOT FLAGGED)
    - True label (fraud / genuine)
    - Whether this example is a true positive (TP), false positive (FP), true negative (TN), or false negative (FN)
    - Model version

This allows reviewers to interactively feel how the model behaves on real test-set transactions.

---

## 7. Infrastructure (GCP)

The project runs on Google Cloud Platform:

- **Containerization**

  - `Dockerfile` — production image:
    - Uses `uv` for reproducible dependency installation from `pyproject.toml` and `uv.lock`
    - Copies app code: `internalpy/`, `api/`
    - Copies model artifacts: `model/artifacts/`, `model/demo_samples.json`
    - Copies static UI: `static/`
  - `Dockerfile.dev` — development image with hot reload and dev dependencies.

- **Cloud Run**

  - Service: `fraud-detector-api` (region: `us-central1`)
  - Deployed from image in Artifact Registry
  - Environment variables:
    - `PROJECT_ID`
    - `BQ_DATASET=fraud`
    - `BQ_TABLE_PREDICTIONS=predictions`
    - `MODEL_PATH=/app/model/artifacts/model.joblib`
    - `PLATFORM=prod`
  - Secret:
    - `INTERNAL_AUTH_TOKEN` from Secret Manager (for protected endpoints, if needed)

- **BigQuery**

  - Dataset: `fraud`
  - Table: `fraud.predictions`
  - Each `/predict` call logs (best-effort, non-blocking):
    - `prediction_timestamp` (UTC ISO string)
    - `fraud_prob`
    - `fraud_flag`
    - `model_version`
    - `Time`
    - `Amount`

- **Configuration & secrets**
  - `internalpy/config.py` centralizes config in a `Vars` dataclass.
  - Local (`PLATFORM=loc`):
    - Uses `.env` and environment variables.
    - Never calls Secret Manager (to avoid local 403s and keep dev simple).
  - Prod (`PLATFORM=prod`):
    - Uses environment variables and can fall back to Secret Manager for secrets.

---

## 8. CI/CD

GitHub Actions workflow: `.github/workflows/deploy_api.yml`

On push to `main` (and changes to relevant files: `api/**`, `internalpy/**`, `model/**`, `Dockerfile`, etc.):

1. Authenticate to GCP with a CI service account.
2. Build a Docker image from `Dockerfile`.
3. Push the image to Artifact Registry, tagged with:
   - commit SHA
   - `latest`
4. Deploy to Cloud Run:
   - Set env vars (`PROJECT_ID`, `BQ_DATASET`, `BQ_TABLE_PREDICTIONS`, `MODEL_PATH`, `PLATFORM`)
   - Wire `INTERNAL_AUTH_TOKEN` from Secret Manager.

Model updates become simple:

    # retrain & regenerate artifacts
    uv run python -m model.train
    make test

    # commit & push (including model/artifacts and demo_samples)
    git add .
    git commit -m "Retrain XGBoost model and update artifacts"
    git push origin main

GitHub Actions then builds and deploys automatically.

---

## 9. Local Development

### 9.1 Requirements

- Python 3.12
- `uv` for dependency management
- Docker (optional, for local container runs)

### 9.2 Setup

    # Install dev dependencies
    uv sync --extra dev

    # Train models & generate artifacts
    uv run python -m model.train

    # Run tests
    make test

    # Run the API locally
    uv run uvicorn api.main:app --reload

Then visit:

- Swagger docs: `http://127.0.0.1:8000/docs`
- Demo UI: `http://127.0.0.1:8000/demo`

### 9.3 Local Docker dev (optional)

    # Build dev image
    make build

    # Run dev container with hot reload (mounts the repo + .env)
    make run

---

## 10. References

- Le Borgne, Y.-A., Siblini, W., Lebichot, B., Bontempi, G.  
  _Reproducible Machine Learning for Credit Card Fraud Detection – Practical Handbook._  
  Used as conceptual guidance for:

  - temporal validation,
  - choice of metrics (PR-AUC, Precision@Top-k),
  - discussion of resampling and cost-sensitive training trade-offs.

- Kaggle, _Credit Card Fraud Detection_ dataset (European card transactions).
