import json
import warnings
from datetime import date
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

warnings.filterwarnings('ignore')

SEED = 42
DATA_PATH      = Path('data/synthetic/shipments_train.csv')
PHARMA_PATH    = Path('data/master/pharma_classes.json')
MODEL_PATH     = Path('ml/model.pkl')
EXPLAINER_PATH = Path('ml/shap_explainer.pkl')
METADATA_PATH  = Path('ml/model_metadata.json')
OUTPUTS_PATH   = Path('outputs')
OUTPUTS_PATH.mkdir(exist_ok=True)

CATEGORICAL_COLS = ['pharma_class_id', 'primary_transport_mode', 'packaging_type_used', 'season']
DROP_COLS        = ['shipment_id', 'product_name', 'origin_id', 'destination_id', 'temperature_excursion']
NUMERIC_COLS     = [
    'distance_km', 'weight_kg', 'transit_duration_hours', 'num_handovers',
    'max_ambient_temp_c_expected', 'min_ambient_temp_c_expected',
    'geopolitical_risk_score', 'carrier_reliability_score', 'packaging_autonomy_hours',
]


# ── Feature engineering ───────────────────────────────────────────────────────

def build_pharma_ranges(pharma_path: Path) -> dict:
    with open(pharma_path) as f:
        data = json.load(f)
    return {p['id']: {'temp_min': p['temp_min'], 'temp_max': p['temp_max']}
            for p in data['pharma_classes']}


def engineer_features(df: pd.DataFrame, pharma_ranges: dict,
                       label_encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    # Derived features
    pharma_min = df['pharma_class_id'].map({k: v['temp_min'] for k, v in pharma_ranges.items()})
    pharma_max = df['pharma_class_id'].map({k: v['temp_max'] for k, v in pharma_ranges.items()})
    df['temp_buffer'] = np.minimum(
        pharma_max - df['max_ambient_temp_c_expected'],
        df['min_ambient_temp_c_expected'] - pharma_min,
    )
    df['autonomy_gap']       = df['packaging_autonomy_hours'] - df['transit_duration_hours']
    df['is_intercontinental'] = (df['distance_km'] > 5000).astype(int)

    # Encode categoricals
    fit_mode = label_encoders is None
    if fit_mode:
        label_encoders = {}
    for col in CATEGORICAL_COLS:
        enc_col = f'{col}_enc'
        if fit_mode:
            le = LabelEncoder()
            df[enc_col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            df[enc_col] = le.transform(df[col].astype(str))

    feature_cols = (
        NUMERIC_COLS
        + ['temp_buffer', 'autonomy_gap', 'is_intercontinental']
        + [f'{c}_enc' for c in CATEGORICAL_COLS]
    )
    return df[feature_cols], label_encoders


# ── Training ──────────────────────────────────────────────────────────────────

def build_models():
    return {
        'LogisticRegression':   LogisticRegression(max_iter=1000, random_state=SEED),
        'DecisionTree':         DecisionTreeClassifier(max_depth=8, random_state=SEED),
        'RandomForest':         RandomForestClassifier(n_estimators=200, max_depth=12,
                                                       random_state=SEED, n_jobs=-1),
        'XGBClassifier':        xgb.XGBClassifier(n_estimators=200, max_depth=6,
                                                   learning_rate=0.05, random_state=SEED,
                                                   eval_metric='logloss', verbosity=0),
    }


def evaluate_models(models, X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring='roc_auc', n_jobs=-1)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        results[name] = {
            'model':       model,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std':  cv_scores.std(),
            'auc':         roc_auc_score(y_test, y_prob),
            'accuracy':    accuracy_score(y_test, y_pred),
            'precision':   precision_score(y_test, y_pred),
            'recall':      recall_score(y_test, y_pred),
            'f1':          f1_score(y_test, y_pred),
            'y_prob':      y_prob,
            'y_pred':      y_pred,
        }
    return results


# ── Plot dashboard ────────────────────────────────────────────────────────────

def plot_dashboard(results, best_name, X_test, y_test, feature_names, output_path):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('PharmAshield — Model Evaluation Dashboard', fontsize=16,
                 fontweight='bold', color='white', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    names     = list(results.keys())
    auc_means = [results[n]['cv_auc_mean'] for n in names]
    auc_stds  = [results[n]['cv_auc_std']  for n in names]
    test_aucs = [results[n]['auc']          for n in names]
    colors    = ['#4FC3F7', '#81C784', '#FFB74D', '#F06292']

    # 1 — AUC comparison (CV vs Test)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(names))
    w = 0.35
    ax1.bar(x - w/2, auc_means, w, yerr=auc_stds, label='CV AUC',   color=colors, alpha=0.85,
            capsize=4, error_kw={'ecolor': 'white', 'elinewidth': 1.5})
    ax1.bar(x + w/2, test_aucs,  w,                label='Test AUC', color=colors, alpha=0.45)
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace('Classifier', '').replace('Regression', ' Reg') for n in names],
                        rotation=15, ha='right', fontsize=8)
    ax1.set_ylim(0.5, 1.0)
    ax1.set_ylabel('AUC')
    ax1.set_title('CV vs Test AUC')
    ax1.legend(fontsize=8)

    # 2 — Metrics table
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1']
    headers = ['Model', 'AUC', 'Acc', 'Prec', 'Rec', 'F1']
    rows = [[n] + [f"{results[n][m]:.3f}" for m in metrics] for n in names]
    tbl = ax2.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#1a1a2e' if r % 2 == 0 else '#16213e')
        cell.set_edgecolor('#444466')
        cell.set_text_props(color='white')
        if r == 0:
            cell.set_facecolor('#0f3460')
    ax2.set_title('Metrics Summary', pad=12)

    # 3 — ROC curves
    ax3 = fig.add_subplot(gs[0, 2])
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        label = f"{name.replace('Classifier','').replace('Regression',' Reg')} ({res['auc']:.3f})"
        ax3.plot(fpr, tpr, color=colors[i], lw=2 if name == best_name else 1.2,
                 linestyle='-' if name == best_name else '--', label=label)
    ax3.plot([0, 1], [0, 1], 'w--', alpha=0.4)
    ax3.set_xlabel('FPR'); ax3.set_ylabel('TPR')
    ax3.set_title('ROC Curves')
    ax3.legend(fontsize=7, loc='lower right')

    # 4 — Confusion matrix (best model)
    ax4 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_test, results[best_name]['y_pred'])
    im = ax4.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white', fontsize=14, fontweight='bold')
    ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Pred 0', 'Pred 1']); ax4.set_yticklabels(['True 0', 'True 1'])
    ax4.set_title(f'Confusion Matrix — {best_name}')

    # 5 — Precision / Recall per class
    ax5 = fig.add_subplot(gs[1, 1])
    report = classification_report(y_test, results[best_name]['y_pred'],
                                   output_dict=True)
    classes  = ['0 (No excursion)', '1 (Excursion)']
    prec_vals = [report['0']['precision'], report['1']['precision']]
    rec_vals  = [report['0']['recall'],    report['1']['recall']]
    xc = np.arange(2)
    ax5.bar(xc - 0.2, prec_vals, 0.4, label='Precision', color='#4FC3F7')
    ax5.bar(xc + 0.2, rec_vals,  0.4, label='Recall',    color='#F06292')
    ax5.set_xticks(xc); ax5.set_xticklabels(classes, fontsize=9)
    ax5.set_ylim(0, 1.1)
    ax5.legend(fontsize=9)
    ax5.set_title(f'Precision / Recall — {best_name}')

    # 6 — Feature importance (top 15)
    ax6 = fig.add_subplot(gs[1, 2])
    best_model = results[best_name]['model']
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_[0])
    idx = np.argsort(importances)[-15:]
    ax6.barh([feature_names[i] for i in idx], importances[idx], color='#81C784')
    ax6.set_title(f'Feature Importance (top 15) — {best_name}')
    ax6.set_xlabel('Importance')

    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close()
    print(f"  Dashboard saved -> {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    pharma_ranges = build_pharma_ranges(PHARMA_PATH)

    y = df['temperature_excursion']
    X, label_encoders = engineer_features(df, pharma_ranges)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"  Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")
    print(f"  Excursion rate — train: {y_train.mean():.3f}  |  test: {y_test.mean():.3f}")

    print("\nTraining models...")
    models  = build_models()
    results = evaluate_models(models, X_train, X_test, y_train, y_test)

    best_name = max(results, key=lambda n: results[n]['auc'])
    best      = results[best_name]

    print("\nGenerating evaluation dashboard...")
    plot_dashboard(results, best_name, X_test, y_test,
                   list(X.columns), OUTPUTS_PATH / 'model_evaluation.png')

    # Serialize model
    joblib.dump(best['model'], MODEL_PATH)
    print(f"  Model saved -> {MODEL_PATH}")

    # SHAP explainer
    print("  Computing SHAP values on test set...")
    if isinstance(best['model'], (RandomForestClassifier,
                                  xgb.XGBClassifier,
                                  DecisionTreeClassifier,
                                  GradientBoostingClassifier)):
        explainer   = shap.TreeExplainer(best['model'])
        shap_values = explainer.shap_values(X_test)
    else:
        explainer   = shap.LinearExplainer(best['model'], X_train)
        shap_values = explainer.shap_values(X_test)

    shap_data = {
        'explainer':      explainer,
        'shap_values':    shap_values,
        'feature_names':  list(X.columns),
        'X_test_sample':  X_test.head(100),
    }
    joblib.dump(shap_data, EXPLAINER_PATH)
    print(f"  SHAP explainer saved -> {EXPLAINER_PATH}")

    # Feature importances
    if hasattr(best['model'], 'feature_importances_'):
        imp = best['model'].feature_importances_
    else:
        imp = np.abs(best['model'].coef_[0])
    feat_imp = dict(zip(X.columns, imp.tolist()))

    # Metadata
    metadata = {
        'model_name':    best_name,
        'best_auc':      round(best['auc'], 4),
        'cv_auc_mean':   round(best['cv_auc_mean'], 4),
        'cv_auc_std':    round(best['cv_auc_std'], 4),
        'feature_names': list(X.columns),
        'label_encoders': {
            col: le.classes_.tolist() for col, le in label_encoders.items()
        },
        'feature_engineering': {
            'pharma_temp_ranges': pharma_ranges,
        },
        'training_date':        str(date.today()),
        'n_train':              int(len(X_train)),
        'n_test':               int(len(X_test)),
        'excursion_rate_train': round(float(y_train.mean()), 4),
        'excursion_rate_test':  round(float(y_test.mean()), 4),
        'feature_importances':  {k: round(v, 6) for k, v in feat_imp.items()},
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved -> {METADATA_PATH}")

    # Top-5 feature importances
    top5 = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:5]

    sep = '=' * 60
    print(f"\n{sep}")
    print("PHARMASHIELD MODEL TRAINING COMPLETE")
    print(f"Best model:  {best_name}")
    print(f"Test AUC:    {best['auc']:.4f}")
    print(f"CV AUC:      {best['cv_auc_mean']:.4f} ± {best['cv_auc_std']:.4f}")
    print(f"\nMetrics on test set ({len(X_test)} shipments):")
    print(f"  Accuracy:  {best['accuracy']:.3f}")
    print(f"  Precision: {best['precision']:.3f}")
    print(f"  Recall:    {best['recall']:.3f}")
    print(f"  F1:        {best['f1']:.3f}")
    print(f"\nFeature importances (top 5):")
    for fname, score in top5:
        print(f"  {fname:<35} {score:.4f}")
    print(f"\nFiles saved:")
    print(f"  {MODEL_PATH}")
    print(f"  {EXPLAINER_PATH}")
    print(f"  {METADATA_PATH}")
    print(f"  {OUTPUTS_PATH / 'model_evaluation.png'}")
    print(sep)


if __name__ == '__main__':
    main()
