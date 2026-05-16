import sys
from pathlib import Path

root = Path(__file__).parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st

st.set_page_config(
    page_title="Explainability · PharmaShield",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Model Explainability")
st.caption(
    "Understanding why the model predicts risk — "
    "powered by SHAP (SHapley Additive exPlanations)"
)


# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_shap_data() -> dict:
    return joblib.load(root / 'ml' / 'shap_explainer.pkl')


@st.cache_data
def load_metadata() -> dict:
    with open(root / 'ml' / 'model_metadata.json', encoding='utf-8') as f:
        return json.load(f)


shap_data     = load_shap_data()
metadata      = load_metadata()
feature_names = shap_data['feature_names']
X_sample      = shap_data['X_test_sample']           # shape (100, 16)

# Normalise SHAP values to (n_samples, n_features) for excursion=1
_sv = shap_data['shap_values']
if isinstance(_sv, list):
    sv_positive = np.array(_sv[1])                   # list of per-class arrays
elif _sv.ndim == 3:
    sv_positive = _sv[:, :, 1]                       # (n, features, classes) → class 1
else:
    sv_positive = _sv                                 # already (n, features)

# expected_value for excursion=1
_ev = shap_data['explainer'].expected_value
expected_value = float(_ev[1] if hasattr(_ev, '__len__') else _ev)


# ── Section 1 — Global feature importance ─────────────────────────────────────

st.subheader("Global Feature Importance")
st.caption("Which features matter most across all test shipments?")

mean_abs = np.abs(sv_positive).mean(axis=0)
imp_df = (
    pd.DataFrame({'Feature': feature_names, 'Mean |SHAP|': mean_abs})
    .sort_values('Mean |SHAP|', ascending=True)
    .tail(12)
)

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#1a1d27')
ax.barh(imp_df['Feature'], imp_df['Mean |SHAP|'], color='#7c6af7', alpha=0.85)
ax.set_xlabel('Mean |SHAP value|', color='#e2e8f0')
ax.set_title('Top Feature Importances (SHAP)', color='#e2e8f0', fontsize=13)
ax.tick_params(colors='#94a3b8')
for spine in ax.spines.values():
    spine.set_edgecolor('#2d3148')
st.pyplot(fig)
plt.close()


# ── Section 2 — Individual shipment waterfall ──────────────────────────────────

st.divider()
st.subheader("Individual Shipment Explanation")
st.caption(
    "Select a shipment from the 100-sample test subset to see which features "
    "drove its risk prediction"
)

n_samples   = len(X_sample)
sample_idx  = st.slider("Select test shipment index", 0, n_samples - 1, 0)

# Show raw features for the selected shipment
st.markdown("**Feature values for this shipment:**")
feat_display = X_sample.iloc[[sample_idx]].T.copy()
feat_display.columns = ['Value']
feat_display['Value'] = feat_display['Value'].round(4)
st.dataframe(feat_display, use_container_width=True)

# Waterfall: top-10 features by |SHAP|
shap_single = sv_positive[sample_idx]                # shape (16,)
sorted_idx  = np.argsort(np.abs(shap_single))[-10:]
feat_labels = [feature_names[i] for i in sorted_idx]
vals        = shap_single[sorted_idx]
colors      = ['#ef4444' if v > 0 else '#22c55e' for v in vals]

pred_score  = float(shap_single.sum() + expected_value)

fig2, ax2 = plt.subplots(figsize=(8, 5))
fig2.patch.set_facecolor('#0f1117')
ax2.set_facecolor('#1a1d27')
ax2.barh(feat_labels, vals, color=colors, alpha=0.85)
ax2.axvline(0, color='#94a3b8', linewidth=0.8, linestyle='--')
ax2.set_xlabel('SHAP value (impact on prediction)', color='#e2e8f0')
ax2.set_title(
    f'Feature Impact — Shipment #{sample_idx} | '
    f'Predicted excursion prob: {pred_score:.3f}',
    color='#e2e8f0',
    fontsize=11,
)
ax2.tick_params(colors='#94a3b8')
for spine in ax2.spines.values():
    spine.set_edgecolor('#2d3148')

st.pyplot(fig2)
plt.close()

st.caption(
    "🔴 Red = feature **increases** excursion risk  "
    "| 🟢 Green = feature **decreases** risk  "
    f"| Base value (E[f(x)]): {expected_value:.3f}"
)


# ── Section 3 — How to read SHAP ──────────────────────────────────────────────

st.divider()
st.subheader("How to Read This")
st.markdown("""
**SHAP (SHapley Additive exPlanations)** assigns each feature a contribution value
for each individual prediction, rooted in cooperative game theory.

- A **positive SHAP value** means the feature pushed the prediction toward **higher risk** (excursion more likely)
- A **negative SHAP value** means the feature pushed the prediction toward **lower risk** (excursion less likely)
- The **base value** E[f(x)] = {:.3f} is the average model output across all test shipments

**Key features in this model:**

| Feature | Role |
|---|---|
| `temp_buffer` | Distance between ambient temp and pharma range — dominant predictor |
| `autonomy_gap` | Packaging duration − transit duration; negative = packaging runs out |
| `pharma_class_id_enc` | Encoded class; UltraCold carries highest baseline risk |
| `carrier_reliability_score` | Historical carrier performance; lower = higher risk |
| `packaging_autonomy_hours` | Absolute packaging endurance in hours |
""".format(expected_value))


# ── Section 4 — Model metadata ────────────────────────────────────────────────

st.divider()
with st.expander("Model metadata"):
    st.json({
        'model_name':        metadata.get('model_name'),
        'training_date':     metadata.get('training_date'),
        'test_auc':          metadata.get('best_auc'),
        'cv_auc_mean':       metadata.get('cv_auc_mean'),
        'cv_auc_std':        metadata.get('cv_auc_std'),
        'n_train':           metadata.get('n_train'),
        'n_test':            metadata.get('n_test'),
        'excursion_rate_train': metadata.get('excursion_rate_train'),
    })
