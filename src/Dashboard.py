"""Dashboard.py — Interactive Streamlit Dashboard for Roughness Data

Launch:
    streamlit run src/Dashboard.py
"""

import os
import sys
import json
import glob

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Allow importing Single.py from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from Single import procesar_carpeta  # noqa: E402

st.set_page_config(page_title="Roughness Data Model", layout="wide")

# ─── Helpers ───────────────────────────────────────────────────────────────

AMPLITUDE_KEYS = ['Ra', 'Rq', 'Rp', 'Rv', 'Rt', 'Rz_ISO']
SHAPE_KEYS = [('Rsk', ''), ('Rku', ''), ('RSm', 'µm'),
              ('Rdq', 'µm/mm'), ('Rda', 'µm/mm'), ('Pc', '1/mm')]
FUNCTIONAL_KEYS = [('Rpk', 'µm'), ('Rk', 'µm'), ('Rvk', 'µm'),
                   ('Mr1', '%'), ('Mr2', '%')]


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ─── Page: Single Specimen ─────────────────────────────────────────────────

def page_single():
    st.header("Single Specimen Analysis")

    specimen_dir = st.text_input("Specimen folder path", value="data/GrupoI/EspeI")

    if not os.path.isdir(specimen_dir):
        st.warning("Enter a valid specimen folder containing .tx1/.tx2/.tx3 files.")
        return

    tx_files = glob.glob(os.path.join(specimen_dir, '*.tx[123]'))
    if not tx_files:
        st.warning("No .tx files found in this directory.")
        return

    use_filter = st.checkbox("Apply ISO 16610 Gaussian filter")
    cutoff = None
    if use_filter:
        cutoff = st.slider("Cutoff wavelength λc (mm)", 0.08, 2.5, 0.8, 0.01)

    if st.button("Run Analysis"):
        with st.spinner("Processing…"):
            result = procesar_carpeta(specimen_dir,
                                      cutoff_mm=cutoff if use_filter else None)

        if result is None:
            st.error("Analysis failed — check the specimen folder.")
            return

        # ── Metrics cards ──
        st.subheader("Roughness Parameters")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Amplitude**")
            for key in AMPLITUDE_KEYS:
                val = result.get(key)
                if val is not None:
                    st.metric(key, f"{val:.3f} µm")

        with col2:
            st.markdown("**Shape & Spacing**")
            for key, unit in SHAPE_KEYS:
                val = result.get(key)
                if val is not None:
                    fmt = f"{val:.2f}" if key == 'Pc' else f"{val:.3f}"
                    st.metric(key, f"{fmt} {unit}".strip())

        with col3:
            st.markdown("**Functional (Rk family)**")
            for key, unit in FUNCTIONAL_KEYS:
                val = result.get(key)
                if val is not None:
                    fmt = f"{val:.2f}" if key in ('Mr1', 'Mr2') else f"{val:.3f}"
                    st.metric(key, f"{fmt} {unit}")

        # ── Profile plots ──
        csv_path = result.get('csv_path')
        if csv_path and os.path.exists(csv_path):
            st.subheader("Profile Plots")
            df_profile = pd.read_csv(csv_path, encoding='utf-8-sig')
            x_col = df_profile.columns[0]
            for col in df_profile.columns[1:]:
                fig = px.line(df_profile, x=x_col, y=col, title=col)
                fig.update_layout(xaxis_title='Position (mm)',
                                  yaxis_title='Height (µm)')
                st.plotly_chart(fig, use_container_width=True)


# ─── Page: Batch Overview ──────────────────────────────────────────────────

def page_batch():
    st.header("Batch Overview")

    summary_path = st.text_input("Batch summary path", "data/batch_summary.json")

    if not os.path.exists(summary_path):
        st.info("Run `python src/Batch.py` first to generate batch_summary.json")
        return

    data = _load_json(summary_path)
    if data is None:
        return

    df = pd.DataFrame(data)

    st.subheader("Summary Table")
    st.dataframe(df, use_container_width=True)

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ('folder', 'csv')]
    if numeric_cols:
        st.subheader("Parameter Heatmap")
        heatmap_data = df[numeric_cols].copy()
        heatmap_data.index = df.get('folder', range(len(df)))
        normed = ((heatmap_data - heatmap_data.min())
                  / (heatmap_data.max() - heatmap_data.min() + 1e-9))
        fig = px.imshow(normed.T, aspect='auto',
                        labels=dict(x='Specimen', y='Parameter',
                                    color='Normalized'),
                        title='Normalized Parameter Heatmap')
        st.plotly_chart(fig, use_container_width=True)


# ─── Page: Group Comparison ────────────────────────────────────────────────

def page_compare():
    st.header("Group Comparison")

    summary_path = st.text_input("Batch summary JSON",
                                 "data/batch_summary.json",
                                 key="compare_path")

    if not os.path.exists(summary_path):
        st.info("Run `python src/Batch.py` first to generate batch_summary.json")
        return

    data = _load_json(summary_path)
    if data is None:
        return

    df = pd.DataFrame(data)

    if 'folder' not in df.columns:
        st.warning("No 'folder' column found in summary.")
        return

    def _extract_group(path):
        parts = str(path).replace('\\', '/').split('/')
        return parts[0] if parts else 'Unknown'

    df['group'] = df['folder'].apply(_extract_group)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ('folder', 'csv')]

    if not numeric_cols:
        st.warning("No numeric metrics found.")
        return

    selected_metric = st.selectbox("Select metric", numeric_cols)

    # Boxplot
    fig_box = px.box(df, x='group', y=selected_metric, points='all',
                     title=f'{selected_metric} by Group')
    st.plotly_chart(fig_box, use_container_width=True)

    # Bar chart of means ± std
    means = df.groupby('group')[selected_metric].agg(['mean', 'std']).reset_index()
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=means['group'], y=means['mean'],
        error_y=dict(type='data', array=means['std']),
        name='Mean ± Std'))
    fig_bar.update_layout(title=f'Mean {selected_metric} ± Std by Group',
                          xaxis_title='Group', yaxis_title=selected_metric)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Stats table
    st.subheader("Group Statistics")
    stats_df = df.groupby('group')[selected_metric].describe()
    st.dataframe(stats_df, use_container_width=True)


# ─── Navigation ────────────────────────────────────────────────────────────

page = st.sidebar.radio("Navigation",
                        ["Single Specimen", "Batch Overview", "Group Comparison"])

if page == "Single Specimen":
    page_single()
elif page == "Batch Overview":
    page_batch()
else:
    page_compare()
