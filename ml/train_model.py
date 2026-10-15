# ml/train_model.py
"""
Failure-risk models, trained honestly.

What changed from the first version
-----------------------------------
1. Inputs are ONLY quantities known before the physics runs (see ml/features.py).
   The old model was given the safety factors that DEFINE the label, so its
   AUC of 1.0 meant nothing.
2. Data comes from ml/generate_dataset.py: thousands of independent flights and
   designs instead of one flight set counted twice.
3. Train / validation / test = 60 / 20 / 20, split by flight_id and stratified on
   the label. A flight can never appear in two sets.
4. Baselines are reported next to the ML models so we can say whether ML adds
   anything: a load threshold, a physics rule (load / height^2, the quantity that
   sets beam stress) and logistic regression.
5. The ensemble probability is calibrated on the validation set (isotonic),
   because class weighting inflates raw probabilities.

NOTE ON WHAT THE MODEL IS
-------------------------
Given load, spar height and material the physics is deterministic (the beam
either yields or it does not), so these models are a fast SURROGATE for a hard
rule, not a discovery of new risk factors. Any "probability" they output is
uncertainty near that boundary.
"""

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features import (FEATURES, LABEL, add_engineered_features,
                         assert_no_leakage)

RISK_CUTS = (("CRITICAL", 0.5), ("HIGH", 0.1), ("MODERATE", 0.01))


def risk_level(prob):
    for name, cut in RISK_CUTS:
        if prob > cut:
            return name
    return "LOW"


def _best_f1_threshold(y_true, score):
    """Score threshold that maximises F1 on the given data."""
    prec, rec, thr = precision_recall_curve(y_true, score)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def _metrics(name, y_true, score, threshold):
    y_pred = (score >= threshold).astype(int)
    return {
        "model_name": name,
        "auc_roc": float(roc_auc_score(y_true, score)),
        "avg_precision": float(average_precision_score(y_true, score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "cm": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "y_pred": y_pred,
        "y_prob": np.asarray(score),
    }


class HydraulicFailurePredictor:
    def __init__(self, data_path="data/ml_dataset.csv"):
        self.data_path = data_path
        self.df_raw = None
        self.feature_names = None
        self.rf_model = self.xgb_model = self.logreg = None
        self.calibrator = None
        self.thresholds = {}
        self.results = {}
        self.cv_results = {}
        self.calibration = None

    # ── data ───────────────────────────────────────────────────────────────
    def load_and_prepare(self):
        print("  Loading data...")
        df = add_engineered_features(pd.read_csv(self.data_path))
        self.feature_names = list(FEATURES)
        assert_no_leakage(self.feature_names)
        self.df_raw = df
        y = df[LABEL].astype(int)
        print(f"  Dataset: {len(df):,} rows, {int(y.sum())} failures "
              f"({y.mean():.2%}), {df['flight_id'].nunique():,} unique flights")
        print(f"  Features ({len(self.feature_names)}): {self.feature_names}")
        return df[self.feature_names], y

    def split_data(self, X, y, random_state=42):
        """60/20/20 split by flight_id, stratified on the label."""
        ids = self.df_raw.groupby("flight_id")[LABEL].max()
        tr, tmp = train_test_split(ids.index, test_size=0.4,
                                   stratify=ids.values, random_state=random_state)
        va, te = train_test_split(tmp, test_size=0.5,
                                  stratify=ids.loc[tmp].values,
                                  random_state=random_state)
        assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
        fid = self.df_raw["flight_id"]
        m_tr, m_va, m_te = fid.isin(tr), fid.isin(va), fid.isin(te)
        self.X_train, self.y_train = X[m_tr], y[m_tr]
        self.X_val, self.y_val = X[m_va], y[m_va]
        self.X_test, self.y_test = X[m_te], y[m_te]
        self.groups_dev = fid[m_tr | m_va]
        for nm, yy in (("Train", self.y_train), ("Val  ", self.y_val),
                       ("Test ", self.y_test)):
            print(f"  {nm}: {len(yy):,} rows, {int(yy.sum())} failures")
        return (self.X_train, self.X_val, self.X_test,
                self.y_train, self.y_val, self.y_test)

    # ── models ─────────────────────────────────────────────────────────────
    def _new_rf(self):
        return RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=42, n_jobs=-1)

    def _new_xgb(self, y):
        n_pos = max(int((y == 1).sum()), 1)
        return xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=(len(y) - n_pos) / n_pos,
            eval_metric="aucpr", random_state=42, n_jobs=-1)

    def train_random_forest(self):
        print("  Training Random Forest (300 trees, balanced class weights)...")
        self.rf_model = self._new_rf().fit(self.X_train, self.y_train)
        return self.rf_model

    def train_xgboost(self):
        print("  Training XGBoost (300 rounds, positive-class weight)...")
        self.xgb_model = self._new_xgb(self.y_train).fit(self.X_train, self.y_train)
        return self.xgb_model

    def train_logistic_regression(self):
        self.logreg = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=3000)
        ).fit(self.X_train, self.y_train)
        return self.logreg

    # ── cross-validation ───────────────────────────────────────────────────
    def cross_validate(self, n_splits=5):
        """Grouped 5-fold CV on train+validation (the test set stays untouched)."""
        X = pd.concat([self.X_train, self.X_val])
        y = pd.concat([self.y_train, self.y_val])
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        out = {"Random Forest": {"auc": [], "ap": []}, "XGBoost": {"auc": [], "ap": []}}
        for tr, va in cv.split(X, y, self.groups_dev):
            for name, model in (("Random Forest", self._new_rf()),
                                ("XGBoost", self._new_xgb(y.iloc[tr]))):
                model.fit(X.iloc[tr], y.iloc[tr])
                p = model.predict_proba(X.iloc[va])[:, 1]
                out[name]["auc"].append(roc_auc_score(y.iloc[va], p))
                out[name]["ap"].append(average_precision_score(y.iloc[va], p))
        self.cv_results = {
            n: {k: (float(np.mean(v)), float(np.std(v))) for k, v in d.items()}
            for n, d in out.items()}
        for n, d in self.cv_results.items():
            print(f"  {n:<14} grouped {n_splits}-fold CV  "
                  f"AUC {d['auc'][0]:.4f} +/- {d['auc'][1]:.4f}   "
                  f"AP {d['ap'][0]:.4f} +/- {d['ap'][1]:.4f}")
        return self.cv_results

    # ── ensemble + calibration ─────────────────────────────────────────────
    def _raw_ensemble(self, X):
        return (self.rf_model.predict_proba(X)[:, 1] +
                self.xgb_model.predict_proba(X)[:, 1]) / 2

    def fit_calibrator(self):
        """Isotonic calibration of the averaged score, fit on the VALIDATION set."""
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        self.calibrator.fit(self._raw_ensemble(self.X_val), self.y_val)
        return self.calibrator

    def ensemble_proba(self, X):
        return self.calibrator.predict(self._raw_ensemble(X))

    # ── evaluation ─────────────────────────────────────────────────────────
    def evaluate_model(self, model, model_name):
        """Validation-chosen threshold, test-set metrics."""
        thr = _best_f1_threshold(self.y_val, model.predict_proba(self.X_val)[:, 1])
        res = _metrics(model_name, self.y_test,
                       model.predict_proba(self.X_test)[:, 1], thr)
        self.results[model_name] = res
        return res

    def evaluate_all(self):
        """Evaluate ML models and baselines on the held-out test set."""
        self.evaluate_model(self.rf_model, "Random Forest")
        self.evaluate_model(self.xgb_model, "XGBoost")
        self.evaluate_model(self.logreg, "Logistic regression")

        # Calibrated ensemble
        thr = _best_f1_threshold(self.y_val, self.ensemble_proba(self.X_val))
        self.results["Ensemble (calibrated)"] = _metrics(
            "Ensemble (calibrated)", self.y_test, self.ensemble_proba(self.X_test), thr)

        # Non-ML baselines: threshold chosen on TRAIN data only
        for name, fn in (
            ("Load threshold", lambda X: X["peak_load_kN"].values),
            ("Physics rule (load / height^2)",
             lambda X: (X["peak_load_kN"] / X["spar_height_m"] ** 2).values),
        ):
            thr = _best_f1_threshold(self.y_train, fn(self.X_train))
            self.results[name] = _metrics(name, self.y_test, fn(self.X_test), thr)

        # Calibration of the ensemble probability on the test set
        p = self.ensemble_proba(self.X_test)
        frac_pos, mean_pred = calibration_curve(self.y_test, p, n_bins=10,
                                                strategy="quantile")
        self.calibration = {"prob_true": frac_pos.tolist(),
                            "prob_pred": mean_pred.tolist(),
                            "brier": float(brier_score_loss(self.y_test, p)),
                            "brier_always_base_rate": float(
                                brier_score_loss(self.y_test,
                                                 np.full(len(p), self.y_test.mean())))}
        return self.results

    def print_report(self):
        print(f"\n  TEST SET RESULTS  ({len(self.y_test):,} rows, "
              f"{int(self.y_test.sum())} failures; thresholds set on validation/train)")
        print("  " + "-" * 92)
        print("  " + "Model".ljust(32) + "AUC".rjust(8) + "AvgPrec".rjust(9) +
              "Prec".rjust(8) + "Recall".rjust(8) + "F1".rjust(8) + "  TN/FP/FN/TP")
        print("  " + "-" * 92)
        for name, r in self.results.items():
            tn, fp, fn, tp = r["cm"].ravel()
            print("  " + name.ljust(32) + f"{r['auc_roc']:8.4f}{r['avg_precision']:9.4f}"
                  f"{r['precision']:8.3f}{r['recall']:8.3f}{r['f1']:8.3f}"
                  f"  {tn}/{fp}/{fn}/{tp}")
        c = self.calibration
        print(f"\n  Calibration (ensemble): Brier {c['brier']:.5f} vs "
              f"{c['brier_always_base_rate']:.5f} for always predicting the base rate")

    def summary_text(self):
        """Plain-language verdict: does ML beat the non-ML baselines?"""
        ml = max(("Random Forest", "XGBoost", "Ensemble (calibrated)"),
                 key=lambda n: self.results[n]["avg_precision"])
        base = max(("Load threshold", "Physics rule (load / height^2)"),
                   key=lambda n: self.results[n]["avg_precision"])
        a, b = self.results[ml], self.results[base]
        diff = a["avg_precision"] - b["avg_precision"]
        verdict = ("ML beats" if diff > 0.01 else
                   "ML is no better than" if diff > -0.01 else "ML is worse than")
        return (f"{verdict} the best non-ML baseline: {ml} AP {a['avg_precision']:.3f} "
                f"(F1 {a['f1']:.3f}) vs {base} AP {b['avg_precision']:.3f} "
                f"(F1 {b['f1']:.3f}).")

    # ── persistence ────────────────────────────────────────────────────────
    def save_models(self, output_dir="data"):
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.rf_model, os.path.join(output_dir, "rf_model.joblib"))
        joblib.dump(self.xgb_model, os.path.join(output_dir, "xgb_model.joblib"))
        joblib.dump(self.feature_names, os.path.join(output_dir, "feature_names.joblib"))
        joblib.dump(self.calibrator, os.path.join(output_dir, "ml_calibrator.joblib"))
        full = self.df_raw
        meta = {
            "feature_ranges": {f: [float(full[f].min()), float(full[f].max())]
                               for f in self.feature_names},
            "material": "Aluminum 2024-T3",
            "span_m": 4.0,
            "n_rows": int(len(full)),
            "n_failures": int(full[LABEL].sum()),
        }
        joblib.dump(meta, os.path.join(output_dir, "ml_meta.joblib"))
        metrics = {
            "test_set": {n: {k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in r.items()
                             if k in ("auc_roc", "avg_precision", "precision",
                                      "recall", "f1", "threshold")}
                         for n, r in self.results.items()},
            "confusion_matrices": {n: r["cm"].tolist() for n, r in self.results.items()},
            "grouped_cv_mean_std": self.cv_results,
            "calibration": self.calibration,
            "rows": meta["n_rows"], "failures": meta["n_failures"],
            "test_rows": int(len(self.y_test)), "test_failures": int(self.y_test.sum()),
            "summary": self.summary_text(),
        }
        with open(os.path.join(output_dir, "ml_metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)
        print("  Saved: rf_model, xgb_model, feature_names, ml_calibrator, "
              "ml_meta (joblib) and ml_metrics.json")

    def predict_failure_risk(self, features_dict):
        missing = [f for f in self.feature_names if f not in features_dict]
        if missing:
            raise KeyError(f"Missing model inputs: {missing}")
        row = pd.DataFrame([features_dict])[self.feature_names]
        rf_p = float(self.rf_model.predict_proba(row)[0, 1])
        xg_p = float(self.xgb_model.predict_proba(row)[0, 1])
        ens = float(self.ensemble_proba(row)[0])
        return {"rf_probability": rf_p, "xgb_probability": xg_p,
                "ensemble": ens, "risk_level": risk_level(ens)}


# ── Inference-only loader (used by the dashboard) ──────────────────────────
class FailureRiskModel:
    """Loads the saved models and predicts from a one-row feature frame."""

    def __init__(self, data_dir="data"):
        j = lambda n: joblib.load(os.path.join(data_dir, n))
        self.rf, self.xgb = j("rf_model.joblib"), j("xgb_model.joblib")
        self.feature_names = j("feature_names.joblib")
        self.calibrator = j("ml_calibrator.joblib")
        self.meta = j("ml_meta.joblib")

    def predict(self, row):
        """row: one-row DataFrame from ml.features.build_feature_row."""
        missing = [f for f in self.feature_names if f not in row.columns]
        if missing:
            raise KeyError(f"Missing model inputs: {missing}")
        row = row[self.feature_names]
        rf_p = float(self.rf.predict_proba(row)[0, 1])
        xg_p = float(self.xgb.predict_proba(row)[0, 1])
        ens = float(self.calibrator.predict([(rf_p + xg_p) / 2])[0])
        return {"rf": rf_p, "xgb": xg_p, "ensemble": ens, "level": risk_level(ens)}

    def out_of_range(self, row):
        """Names of inputs outside the range the model was trained on."""
        out = []
        for f, (lo, hi) in self.meta["feature_ranges"].items():
            v = float(row[f].iloc[0])
            if v < lo - 1e-9 or v > hi + 1e-9:
                out.append(f"{f} = {v:.3g} (trained on {lo:.3g} to {hi:.3g})")
        return out


def run_training(data_path="data/ml_dataset.csv", output_dir="data"):
    """Full pipeline: split, train, cross-validate, calibrate, evaluate, save."""
    p = HydraulicFailurePredictor(data_path)
    X, y = p.load_and_prepare()
    p.split_data(X, y)
    p.train_random_forest()
    p.train_xgboost()
    p.train_logistic_regression()
    print("\n  Grouped cross-validation (train + validation only):")
    p.cross_validate()
    p.fit_calibrator()
    p.evaluate_all()
    p.print_report()
    print("\n  " + p.summary_text())
    p.save_models(output_dir)
    return p


if __name__ == "__main__":
    run_training()
