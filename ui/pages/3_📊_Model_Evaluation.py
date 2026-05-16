import sys
from pathlib import Path

root = Path(__file__).parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import json
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Model Evaluation · PharmaShield",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Evaluation")
st.caption("Performance metrics of the trained Random Forest classifier")


# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data
def load_metadata() -> dict:
    with open(root / 'ml' / 'model_metadata.json', encoding='utf-8') as f:
        return json.load(f)


metadata = load_metadata()


# ── Summary metrics ────────────────────────────────────────────────────────────

st.subheader("Best Model Summary")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Model",    metadata.get('model_name', 'RandomForest'))
col2.metric("Test AUC", f"{metadata.get('best_auc', 0):.4f}")
col3.metric(
    "CV AUC",
    f"{metadata.get('cv_auc_mean', 0):.4f}",
    delta=f"± {metadata.get('cv_auc_std', 0):.4f}",
    delta_color="off",
)
col4.metric("Train samples", f"{metadata.get('n_train', 0):,}")
col5.metric("Test samples",  f"{metadata.get('n_test',  0):,}")

st.divider()


# ── Evaluation dashboard image ────────────────────────────────────────────────

eval_img = root / 'outputs' / 'model_evaluation.png'

st.subheader("Full Evaluation Dashboard")
if eval_img.exists():
    st.image(
        str(eval_img),
        use_container_width=True,
        caption=(
            "Left to right: AUC comparison (CV vs Test) · Metrics table · "
            "ROC curves · Confusion matrix · Precision/Recall · Feature importance"
        ),
    )
else:
    st.warning(
        "`outputs/model_evaluation.png` not found. "
        "Run `python ml/train.py` to generate it."
    )

st.divider()


# ── Feature importances table ─────────────────────────────────────────────────

st.subheader("Feature Importances")

if 'feature_importances' in metadata:
    fi_df = (
        pd.DataFrame(
            list(metadata['feature_importances'].items()),
            columns=['Feature', 'Importance'],
        )
        .sort_values('Importance', ascending=False)
        .reset_index(drop=True)
    )
    fi_df['Importance'] = fi_df['Importance'].round(5)
    fi_df.index += 1                  # rank starting at 1

    st.dataframe(fi_df, use_container_width=True)
else:
    st.info("Feature importance details are visible in the dashboard image above.")

st.divider()


# ── Dataset summary ────────────────────────────────────────────────────────────

st.subheader("Dataset Summary")

col_a, col_b, col_c, col_d = st.columns(4)
total = metadata.get('n_train', 0) + metadata.get('n_test', 0)
col_a.metric("Total Shipments", f"{total:,}")
col_b.metric("Excursion Rate (train)", f"{metadata.get('excursion_rate_train', 0):.1%}")
col_c.metric("Excursion Rate (test)",  f"{metadata.get('excursion_rate_test',  0):.1%}")
col_d.metric("Features", len(metadata.get('feature_names', [])))

st.divider()


# ── Feature schema ────────────────────────────────────────────────────────────

with st.expander("Feature schema"):
    feat_rows = []
    fi = metadata.get('feature_importances', {})
    for feat in metadata.get('feature_names', []):
        feat_rows.append({
            'Feature':    feat,
            'Importance': round(fi.get(feat, 0), 5),
            'Type':       'encoded categorical' if feat.endswith('_enc') else 'numeric',
        })
    st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

st.caption(
    f"Trained: {metadata.get('training_date', 'N/A')} · "
    "Algorithm: Random Forest (200 trees, max_depth=12) · "
    "Seed: 42 · Stratified 80/20 split · 5-fold CV"
)
