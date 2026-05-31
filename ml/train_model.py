# ml/train_model.py
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score)
import xgboost as xgb
import joblib


class HydraulicFailurePredictor:

    def __init__(self, data_path):
        self.data_path     = data_path
        self.df_raw        = None
        self.feature_names = None
        self.rf_model      = None
        self.xgb_model     = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def load_and_prepare(self):
        print('  Loading data...')
        df = pd.read_csv(self.data_path)
        # Also load dangerous config to get more failure examples
        other_path = self.data_path.replace('mc_standard', 'mc_dangerous')
        import os
        if os.path.exists(other_path):
            df2 = pd.read_csv(other_path)
            df  = pd.concat([df, df2], ignore_index=True)
            print('  Combined both configs for more failure examples.')
        self.df_raw = df
        n_fail = df['any_failure'].sum()
        print('  Dataset: ' + str(len(df)) + ' rows, ' +
              str(n_fail) + ' failures (' +
              str(round(df['any_failure'].mean()*100, 2)) + '%)')

        feature_cols = [
            'peak_load_kN', 'turbulence_intensity', 'sideslip_deg',
            'max_beam_stress_mpa', 'beam_safety_factor',
            'tip_deflection_mm', 'root_moment_kNm',
            'min_line_sf', 'max_line_vm_mpa',
            'config_score', 'n_unsafe_lines',
        ]

        for i in range(1, 4):
            for suf in ['_vm_mpa', '_sf', '_score']:
                col = 'line' + str(i) + suf
                if col in df.columns:
                    feature_cols.append(col)

        df['load_x_turbulence'] = df['peak_load_kN'] * df['turbulence_intensity']
        df['stress_utilization'] = df['max_beam_stress_mpa'] / 345.0
        df['sf_margin']          = df['beam_safety_factor'] - 1.5
        df['log_peak_load']      = np.log1p(df['peak_load_kN'])
        df['load_squared']       = df['peak_load_kN'] ** 2
        feature_cols += ['load_x_turbulence', 'stress_utilization',
                          'sf_margin', 'log_peak_load', 'load_squared']

        scenario_dummies = pd.get_dummies(df['scenario_name'], prefix='scen')
        df = pd.concat([df, scenario_dummies], axis=1)
        self.df_raw = df
        feature_cols += scenario_dummies.columns.tolist()
        feature_cols = [c for c in feature_cols if c in df.columns]

        X = df[feature_cols].fillna(0)
        y = df['any_failure'].astype(int)
        self.feature_names = feature_cols
        print('  Features: ' + str(len(feature_cols)))
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=test_size,
                              random_state=random_state, stratify=y)
        print('  Train: ' + str(len(self.X_train)) + ' samples (' +
              str(self.y_train.sum()) + ' failures)')
        print('  Test:  ' + str(len(self.X_test)) + ' samples (' +
              str(self.y_test.sum()) + ' failures)')
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_random_forest(self):
        print('  Training Random Forest (200 trees)...')
        self.rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=2,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1,
        )
        self.rf_model.fit(self.X_train, self.y_train)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self.rf_model, self.X_train, self.y_train,
                                  cv=cv, scoring='roc_auc')
        print('  RF CV AUC: ' + str(round(scores.mean(),4)) +
              ' +/- ' + str(round(scores.std(),4)))
        return self.rf_model

    def train_xgboost(self):
        print('  Training XGBoost (300 rounds)...')
        n_neg = (self.y_train == 0).sum()
        n_pos = (self.y_train == 1).sum()
        spw   = n_neg / max(n_pos, 1)
        print('  Class weight: ' + str(round(spw, 1)))
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric='auc', random_state=42, n_jobs=-1,
        )
        self.xgb_model.fit(self.X_train, self.y_train,
                            eval_set=[(self.X_test, self.y_test)],
                            verbose=False)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self.xgb_model, self.X_train, self.y_train,
                                  cv=cv, scoring='roc_auc')
        print('  XGB CV AUC: ' + str(round(scores.mean(),4)) +
              ' +/- ' + str(round(scores.std(),4)))
        return self.xgb_model

    def evaluate_model(self, model, model_name):
        print('  Evaluating ' + model_name + '...')
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]

        # Handle edge case: no failures in test set
        unique_classes = len(set(self.y_test))
        if unique_classes < 2:
            print('  WARNING: Test set has only 1 class — using train set for evaluation')
            y_eval  = self.y_train
            yp_eval = model.predict(self.X_train)
            ypr_eval = model.predict_proba(self.X_train)[:, 1]
        else:
            y_eval   = self.y_test
            yp_eval  = y_pred
            ypr_eval = y_prob

        try:
            auc = roc_auc_score(y_eval, ypr_eval)
        except Exception:
            auc = 0.5

        try:
            ap = average_precision_score(y_eval, ypr_eval)
        except Exception:
            ap = 0.0

        cm = confusion_matrix(y_eval, yp_eval)
        print('  AUC-ROC: ' + str(round(auc, 4)))
        print('  Avg Precision: ' + str(round(ap, 4)))

        labels = sorted(set(y_eval))
        target_names = ['Safe' if l==0 else 'Failure' for l in labels]
        print(classification_report(y_eval, yp_eval,
                                     labels=labels,
                                     target_names=target_names,
                                     zero_division=0))
        return {
            'model_name':    model_name,
            'auc_roc':       auc,
            'avg_precision': ap,
            'y_pred':        yp_eval,
            'y_prob':        ypr_eval,
            'cm':            cm,
        }
    def save_models(self, output_dir='data'):
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.rf_model,      os.path.join(output_dir, 'rf_model.joblib'))
        joblib.dump(self.xgb_model,     os.path.join(output_dir, 'xgb_model.joblib'))
        joblib.dump(self.feature_names, os.path.join(output_dir, 'feature_names.joblib'))
        print('  Saved: data/rf_model.joblib')
        print('  Saved: data/xgb_model.joblib')

    def predict_failure_risk(self, features_dict):
        row     = pd.DataFrame([features_dict])
        row     = row.reindex(columns=self.feature_names, fill_value=0)
        rf_prob = self.rf_model.predict_proba(row)[0, 1]
        xg_prob = self.xgb_model.predict_proba(row)[0, 1]
        ens     = (rf_prob + xg_prob) / 2
        level   = ('CRITICAL' if ens > 0.5 else
                   'HIGH'     if ens > 0.1 else
                   'MODERATE' if ens > 0.01 else 'LOW')
        return {'rf_probability': rf_prob, 'xgb_probability': xg_prob,
                'ensemble': ens, 'risk_level': level}