# visualization/ml_plots.py

"""
Machine Learning Results Visualization
========================================
Plots for understanding model performance and predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, precision_recall_curve

def plot_model_comparison(results: dict, y_test):
    """
    Compare RF and XGBoost side by side.
    4 panels: ROC curves, PR curves, confusion matrices, feature importance.
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('ML Model Comparison — Hydraulic Failure Prediction',
                  fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    colors = {'Random Forest': 'steelblue', 'XGBoost': 'darkorange'}

    # ── Panel 1: ROC Curves ────────────────────────────────────────────
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.50)')

    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
        auc = result['auc_roc']
        ax1.plot(fpr, tpr, color=colors.get(name, 'green'),
                  linewidth=2.5,
                  label=f'{name} (AUC={auc:.4f})')

    ax1.fill_between([0, 1], [0, 0], [1, 1], alpha=0.05, color='gray')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve\n(Higher = better)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Precision-Recall Curves ──────────────────────────────
    for name, result in results.items():
        prec, rec, _ = precision_recall_curve(y_test, result['y_prob'])
        ap = result['avg_precision']
        ax2.plot(rec, prec, color=colors.get(name, 'green'),
                  linewidth=2.5,
                  label=f'{name} (AP={ap:.4f})')

    baseline = y_test.mean()
    ax2.axhline(baseline, color='gray', linewidth=1.5, linestyle='--',
                 label=f'Baseline (P={baseline:.4f})')

    ax2.set_xlabel('Recall (Fraction of failures caught)')
    ax2.set_ylabel('Precision (Fraction of alerts correct)')
    ax2.set_title('Precision-Recall Curve\n(Important for imbalanced data)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Confusion matrices ────────────────────────────────────
    for idx, (name, result) in enumerate(results.items()):
        cm = result['cm']
        ax = ax3 if idx == 0 else ax4

        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax)

        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                         ha='center', va='center', fontsize=14,
                         fontweight='bold',
                         color='white' if cm[i, j] > thresh else 'black')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted Safe', 'Predicted Failure'])
        ax.set_yticklabels(['Actual Safe', 'Actual Failure'])
        ax.set_title(f'{name}\nConfusion Matrix')

        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, cm[1,1])
        total = cm.sum()
        ax.set_xlabel(
            f'TN={tn}  FP={fp}  FN={fn}  TP={tp}\n'
            f'Accuracy={(tn+tp)/total*100:.1f}%  '
            f'Recall={tp/max(tp+fn,1)*100:.1f}%'
        )

    plt.tight_layout()
    return fig


def plot_feature_importance(predictor, top_n=20):
    """
    Feature importance from both models.

    Random Forest: mean decrease in impurity
    XGBoost: gain (improvement in loss when feature is used)

    High importance = feature is very predictive of failure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'Feature Importance — Top {top_n} Features',
                  fontsize=13, fontweight='bold')

    for ax, model, name, color in [
        (axes[0], predictor.rf_model,  'Random Forest', 'steelblue'),
        (axes[1], predictor.xgb_model, 'XGBoost',       'darkorange'),
    ]:
        importances = model.feature_importances_
        indices     = np.argsort(importances)[-top_n:]
        features    = [predictor.feature_names[i] for i in indices]
        values      = importances[indices]

        bars = ax.barh(range(top_n), values, color=color,
                        alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{name} Feature Importance\n'
                      f'(Higher = more predictive of failure)')
        ax.grid(True, alpha=0.3, axis='x')

        # Highlight top 5
        for i, (bar, val) in enumerate(zip(bars, values)):
            if i >= top_n - 5:
                bar.set_facecolor('red' if name == 'Random Forest'
                                   else 'darkred')
                ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                         f'{val:.4f}', va='center', fontsize=7)

    plt.tight_layout()
    return fig


def plot_failure_probability_landscape(predictor, df):
    """
    2D heatmap: failure probability across load × turbulence space.

    Shows WHERE in the flight envelope failures are most likely.
    This is what a real-time safety dashboard would show pilots/dispatchers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Failure Probability Landscape\n'
                  '(ML prediction across flight envelope)',
                  fontsize=13, fontweight='bold')

    load_range = np.linspace(df['peak_load_kN'].min(),
                               df['peak_load_kN'].max(), 40)
    turb_range = np.linspace(df['turbulence_intensity'].min(),
                               df['turbulence_intensity'].max(), 40)

    L, T = np.meshgrid(load_range, turb_range)

    # Use mean values for all other features
    mean_features = df[predictor.feature_names].mean().to_dict()

    for ax, model, name in [
        (axes[0], predictor.rf_model,  'Random Forest'),
        (axes[1], predictor.xgb_model, 'XGBoost'),
    ]:
        Z = np.zeros_like(L)

        for i in range(len(turb_range)):
            rows = []
            for j in range(len(load_range)):
                feat = mean_features.copy()
                feat['peak_load_kN']       = L[i, j]
                feat['turbulence_intensity'] = T[i, j]
                feat['log_peak_load']       = np.log1p(L[i, j])
                feat['load_squared']        = L[i, j]**2
                feat['load_x_turbulence']  = L[i, j] * T[i, j]
                rows.append(feat)

            X_grid = pd.DataFrame(rows)[predictor.feature_names].fillna(0)
            Z[i, :] = model.predict_proba(X_grid)[:, 1]

        contour = ax.contourf(L, T, Z, levels=20,
                               cmap='RdYlGn_r', alpha=0.9)
        plt.colorbar(contour, ax=ax, label='P(failure)')

        # Overlay actual data points
        safe_mask = df['any_failure'] == 0
        fail_mask = df['any_failure'] == 1

        ax.scatter(df.loc[safe_mask, 'peak_load_kN'],
                    df.loc[safe_mask, 'turbulence_intensity'],
                    c='green', s=3, alpha=0.3, label='Safe flights')

        if fail_mask.sum() > 0:
            ax.scatter(df.loc[fail_mask, 'peak_load_kN'],
                        df.loc[fail_mask, 'turbulence_intensity'],
                        c='red', s=50, marker='X', zorder=10,
                        label='Failures')

        ax.set_xlabel('Peak Load (kN)')
        ax.set_ylabel('Turbulence Intensity (m/s)')
        ax.set_title(f'{name}\nFailure Probability Landscape')
        ax.legend(fontsize=8)

    plt.tight_layout()
    return fig