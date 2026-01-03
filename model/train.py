from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# --------- Global config ---------

RANDOM_SEED = 42
C_FN = 1.0
C_FP = 0.05
TOP_K = 100  # for Precision@k


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'creditcard.csv'
ARTIFACT_DIR = PROJECT_ROOT / 'model' / 'artifacts'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
DEMO_SAMPLES_PATH = PROJECT_ROOT / 'model' / 'demo_samples.json'


# --------- Utilities ---------


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def temporal_train_val_test_split(
    df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by Time and split into train/val/test."""
    if 'Time' not in df.columns:
        raise ValueError("Expected 'Time' column in the dataset.")

    df_sorted = df.sort_values('Time').reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train = df_sorted.iloc[:train_end]
    val = df_sorted.iloc[train_end:val_end]
    test = df_sorted.iloc[val_end:]

    return train, val, test


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """Transaction-level Precision@k."""
    if k <= 0:
        raise ValueError('k must be positive.')
    n = len(y_scores)
    k = min(k, n)

    order = np.argsort(-y_scores)  # descending
    top_k_idx = order[:k]
    y_top = y_true[top_k_idx]
    return float(y_top.sum() / k)


def compute_threshold_free_metrics(
    y_true: np.ndarray, y_scores: np.ndarray, k: int
) -> dict[str, float]:
    pr_auc = average_precision_score(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    p_at_k = precision_at_k(y_true, y_scores, k)
    return {
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'precision_at_k': float(p_at_k),
        'k_value': int(k),
    }


def evaluate_with_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    k: int,
    c_fn: float,
    c_fp: float,
) -> dict[str, float]:
    """Compute threshold-based metrics, expected cost, and top-k metrics."""
    y_pred = (y_scores >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = len(y_true)

    expected_cost = (c_fn * fn + c_fp * fp) / n

    precision = precision_score(y_true, y_pred, zero_division='warn')
    recall = recall_score(y_true, y_pred, zero_division='warn')
    f1 = f1_score(y_true, y_pred, zero_division='warn')

    # threshold-free + top-k
    tf_metrics = compute_threshold_free_metrics(y_true, y_scores, k=k)

    metrics = {
        'expected_cost': float(expected_cost),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
    }
    metrics.update(tf_metrics)
    return metrics


def find_best_threshold_cost(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    c_fn: float = C_FN,
    c_fp: float = C_FP,
    num_thresholds: int = 500,
    k: int = TOP_K,
) -> tuple[float, dict[str, float]]:
    """Grid search threshold to minimize expected cost on validation."""
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    best_cost = float('inf')
    best_threshold = 0.5
    best_metrics: dict[str, float] = {}

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        n = len(y_true)
        cost = (c_fn * fn + c_fp * fp) / n

        if cost < best_cost:
            best_cost = cost
            best_threshold = float(t)
            # Compute full set of metrics at this threshold
            best_metrics = evaluate_with_threshold(
                y_true, y_scores, best_threshold, k=k, c_fn=c_fn, c_fp=c_fp
            )

    return best_threshold, best_metrics


def build_logreg_pipeline() -> Pipeline:
    clf = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='liblinear',
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    pipe = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('clf', clf),
        ]
    )
    return pipe


def build_xgb_pipeline(params: dict[str, Any]) -> Pipeline:
    clf = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=RANDOM_SEED,
        n_jobs=-1,
        **params,
    )
    pipe = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('clf', clf),
        ]
    )
    return pipe


def random_search_xgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_iter: int = 20,
    random_state: int = RANDOM_SEED,
    k: int = TOP_K,
) -> tuple[Pipeline, dict[str, Any], dict[str, float]]:
    """Simple random search over a small hyperparameter grid for XGBoost.

    Returns:
        best_pipeline, best_params, best_val_metrics
    """
    rng = random.Random(random_state)

    param_grid: dict[str, Sequence[Any]] = {
        'n_estimators': [200, 300, 400],
        'learning_rate': [0.03, 0.05, 0.1],
        'max_depth': [3, 4, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 5, 10],
    }

    best_ap = -float('inf')
    best_pipeline: Pipeline | None = None
    best_params: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None

    for i in range(n_iter):
        params: dict[str, Any] = {}
        for name, choices in param_grid.items():
            # choices is a non-empty sequence by construction
            idx = rng.randrange(len(choices))
            params[name] = choices[idx]

        print(f'[XGB search] Iteration {i+1}/{n_iter}, params={params}')

        pipe = build_xgb_pipeline(params)
        pipe.fit(X_train, y_train)

        val_scores = pipe.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, val_scores)
        roc_auc = roc_auc_score(y_val, val_scores)
        p_at_k = precision_at_k(y_val, val_scores, k=k)

        print(
            f'[XGB search] val PR-AUC={pr_auc:.5f}, ' f'ROC-AUC={roc_auc:.5f}, P@{k}={p_at_k:.5f}'
        )

        if pr_auc > best_ap:
            best_ap = pr_auc
            best_pipeline = pipe
            best_params = params
            best_metrics = {
                'pr_auc': float(pr_auc),
                'roc_auc': float(roc_auc),
                'precision_at_k': float(p_at_k),
                'k_value': int(k),
            }

    if best_pipeline is None or best_params is None or best_metrics is None:
        raise RuntimeError('Random search did not find a valid XGBoost model.')

    print(f'[XGB search] Best PR-AUC on val: {best_ap:.5f}')
    return best_pipeline, best_params, best_metrics


def export_demo_samples(
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    output_path: Path = DEMO_SAMPLES_PATH,
    max_fraud: int = 50,
    max_nonfraud: int = 200,
) -> None:
    """
    Export a small set of demo samples from the test set to JSON.

    Each sample contains:
      - id: "fraud_001", "genuine_001", ...
      - class: 1 (fraud) or 0 (genuine)
      - features: full feature dict (Time, V1..V28, Amount)
    """
    fraud_df = test_df[test_df[target_col] == 1].copy()
    nonfraud_df = test_df[test_df[target_col] == 0].copy()

    fraud_df = fraud_df.head(max_fraud)
    nonfraud_df = nonfraud_df.head(max_nonfraud)

    samples: list[dict[str, Any]] = []

    fraud_counter = 0
    for _, row in fraud_df.iterrows():
        fraud_counter += 1
        features = {col: float(row[col]) for col in feature_cols}
        samples.append(
            {
                'id': f'fraud_{fraud_counter:03d}',
                'label': f'Fraud example {fraud_counter}',
                'class': 1,
                'features': features,
            }
        )

    genuine_counter = 0
    for _, row in nonfraud_df.iterrows():
        genuine_counter += 1
        features = {col: float(row[col]) for col in feature_cols}
        samples.append(
            {
                'id': f'genuine_{genuine_counter:03d}',
                'label': f'Genuine example {genuine_counter}',
                'class': 0,
                'features': features,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)

    print(f'[Demo] Exported {len(samples)} demo samples to {output_path}')


# --------- Metadata structures ---------


@dataclass
class MetricsBlock:
    pr_auc: float
    roc_auc: float
    precision_at_k: float
    k_value: int
    precision: float
    recall: float
    f1: float
    expected_cost: float
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass
class Metadata:
    model_type: str
    model_version: str
    features: list[str]
    train_rows: int
    val_rows: int
    test_rows: int
    class_ratio_train: float
    cost_matrix: dict[str, float]
    threshold: float
    metrics: dict[str, dict[str, float]]
    xgb_params: dict[str, Any]
    baseline_logreg: dict[str, float]


# --------- Main training routine ---------


def main() -> None:
    set_global_seed(RANDOM_SEED)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f'Could not find dataset at {DATA_PATH}. '
            'Expected Kaggle creditcard.csv in data/raw/.'
        )

    print(f'[Data] Loading dataset from {DATA_PATH}')
    df = pd.read_csv(DATA_PATH)

    if 'Class' not in df.columns:
        raise ValueError("Expected 'Class' column as target in the dataset.")

    target_col = 'Class'
    feature_cols = [c for c in df.columns if c != target_col]

    # Temporal split
    print('[Data] Performing temporal train/val/test split...')
    train_df, val_df, test_df = temporal_train_val_test_split(df)

    X_train = cast(pd.DataFrame, train_df[feature_cols])
    y_train = np.asarray(train_df[target_col].values.astype(int))

    X_val = cast(pd.DataFrame, val_df[feature_cols])
    y_val = np.asarray(val_df[target_col].values.astype(int))

    X_test = cast(pd.DataFrame, test_df[feature_cols])
    y_test = np.asarray(test_df[target_col].values.astype(int))

    export_demo_samples(test_df, feature_cols, target_col)

    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    n_pos_train = int(np.sum(y_train == 1))
    n_neg_train = int(np.sum(y_train == 0))
    class_ratio_train = n_pos_train / n_neg_train if n_neg_train > 0 else 0.0

    print(
        f'[Data] train={n_train}, val={n_val}, test={n_test}, '
        f'train_pos={n_pos_train}, train_neg={n_neg_train}, '
        f'class_ratio_train={class_ratio_train:.8f}'
    )

    # ----- Baseline: Logistic Regression -----
    print('[Baseline] Training Logistic Regression...')
    lr_pipe = build_logreg_pipeline()
    lr_pipe.fit(X_train, y_train)

    val_scores_lr = lr_pipe.predict_proba(X_val)[:, 1]
    test_scores_lr = lr_pipe.predict_proba(X_test)[:, 1]

    lr_val_pr_auc = average_precision_score(y_val, val_scores_lr)
    lr_val_roc_auc = roc_auc_score(y_val, val_scores_lr)
    lr_test_pr_auc = average_precision_score(y_test, test_scores_lr)
    lr_test_roc_auc = roc_auc_score(y_test, test_scores_lr)

    print(
        f'[Baseline] Logistic Regression - '
        f'val PR-AUC={lr_val_pr_auc:.5f}, val ROC-AUC={lr_val_roc_auc:.5f}, '
        f'test PR-AUC={lr_test_pr_auc:.5f}, test ROC-AUC={lr_test_roc_auc:.5f}'
    )

    baseline_logreg_metrics = {
        'val_pr_auc': float(lr_val_pr_auc),
        'val_roc_auc': float(lr_val_roc_auc),
        'test_pr_auc': float(lr_test_pr_auc),
        'test_roc_auc': float(lr_test_roc_auc),
    }

    # ----- Final model: XGBoost with random search -----
    print('[XGB] Starting random search for XGBoost...')
    xgb_pipe, xgb_best_params, xgb_val_metrics = random_search_xgb(
        X_train, y_train, X_val, y_val, n_iter=20, random_state=RANDOM_SEED, k=TOP_K
    )

    # Fit final XGB pipeline on train only (already done inside random search),
    # we just reuse the best pipeline as-is.

    # ----- Cost-sensitive threshold search on validation -----
    print('[Threshold] Searching threshold on validation set...')
    val_scores_xgb = xgb_pipe.predict_proba(X_val)[:, 1]
    test_scores_xgb = xgb_pipe.predict_proba(X_test)[:, 1]

    best_threshold, val_metrics_at_t = find_best_threshold_cost(
        y_val, val_scores_xgb, c_fn=C_FN, c_fp=C_FP, num_thresholds=500, k=TOP_K
    )

    print(
        f"[Threshold] Best threshold on validation: {best_threshold:.5f}, "
        f"expected_cost={val_metrics_at_t['expected_cost']:.6f}, "
        f"precision={val_metrics_at_t['precision']:.5f}, "
        f"recall={val_metrics_at_t['recall']:.5f}, "
        f"F1={val_metrics_at_t['f1']:.5f}"
    )

    # Metrics on test at the chosen threshold
    test_metrics_at_t = evaluate_with_threshold(
        y_test,
        test_scores_xgb,
        threshold=best_threshold,
        k=TOP_K,
        c_fn=C_FN,
        c_fp=C_FP,
    )

    print(
        "[Test] XGBoost at chosen threshold - "
        f"expected_cost={test_metrics_at_t['expected_cost']:.6f}, "
        f"precision={test_metrics_at_t['precision']:.5f}, "
        f"recall={test_metrics_at_t['recall']:.5f}, "
        f"F1={test_metrics_at_t['f1']:.5f}, "
        f"PR-AUC={test_metrics_at_t['pr_auc']:.5f}, "
        f"ROC-AUC={test_metrics_at_t['roc_auc']:.5f}, "
        f"P@{TOP_K}={test_metrics_at_t['precision_at_k']:.5f}"
    )

    # ----- Build metadata -----
    model_version = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')

    metadata = Metadata(
        model_type='xgboost',
        model_version=model_version,
        features=feature_cols,
        train_rows=n_train,
        val_rows=n_val,
        test_rows=n_test,
        class_ratio_train=float(class_ratio_train),
        cost_matrix={'FN': C_FN, 'FP': C_FP},
        threshold=float(best_threshold),
        metrics={
            'validation': val_metrics_at_t,
            'test': test_metrics_at_t,
            'validation_threshold_free': xgb_val_metrics,
        },
        xgb_params=xgb_best_params,
        baseline_logreg=baseline_logreg_metrics,
    )

    # ----- Save artifacts -----
    model_path = ARTIFACT_DIR / 'model.joblib'
    metadata_path = ARTIFACT_DIR / 'metadata.json'

    print(f'[Artifacts] Saving model to {model_path}')
    joblib.dump(xgb_pipe, model_path)

    print(f'[Artifacts] Saving metadata to {metadata_path}')
    with metadata_path.open('w', encoding='utf-8') as f:
        json.dump(asdict(metadata), f, indent=2)

    print('[Done] Training complete.')


if __name__ == '__main__':
    main()
