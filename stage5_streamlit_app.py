# stage5_streamlit_app.py
# Stage 5 — Streamlit Demo
# Bias-Aware Radiomic Classification of Brain CT Haemorrhage Using Classical ML
# Owner: Tejaswi
# Run with: streamlit run stage5_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Bias-Aware Radiomic Classification of Brain CT Haemorrhage",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

COLOURS = {
    'bg':          '#f8fcff',
    'navbar_bg':   '#0d47a1',
    'navbar_text': '#ffffff',
    'accent':      '#1976d2',
    'brivo':       '#42a5f5',
    'revolution':  '#ff7043',
    'normal_bg':   '#e3f2fd',
    'haem_bg':     '#ffebee',
    'card_bg':     '#ffffff',
    'text_dim':    '#78909c',
    'border':      '#b0bec5',
}

CONFOUND_LABELS = {
    'CONFOUNDED':                  '🔴 CONFOUNDED',
    'REDUCED_CONFOUND':            '🟡 REDUCED CONFOUND',
    'CLEAN':                       '🟢 CLEAN',
    'REDUCED_CONFOUND_SYNTHETIC':  '🔵 SYNTHETIC',
    'CONFOUNDED_NORMALISED':       '🔴 CONFOUNDED (normalised)',
    'REDUCED_CONFOUND_NORMALISED': '🟡 REDUCED CONFOUND (normalised)',
}

PATIENT_META = {
    'CT180-01-2026':  {'class': 'Normal',      'scanner': 'Revolution'},
    'CT183-01-2026':  {'class': 'Normal',      'scanner': 'Revolution'},
    'CT20015788':     {'class': 'Normal',      'scanner': 'BRIVO'},
    'CT20015807':     {'class': 'Normal',      'scanner': 'BRIVO'},
    'CT20015886':     {'class': 'Normal',      'scanner': 'BRIVO'},
    'CT3225-12-2025': {'class': 'Haemorrhage', 'scanner': 'Revolution'},
    'CT3259-12-2025': {'class': 'Haemorrhage', 'scanner': 'Revolution'},
    'CT3274-12-2025': {'class': 'Haemorrhage', 'scanner': 'Revolution'},
    'CT3277-12-2025': {'class': 'Haemorrhage', 'scanner': 'Revolution'},
    'CT3289-12-2025': {'class': 'Haemorrhage', 'scanner': 'Revolution'},
}

FEATURE_TOOLTIPS = {
    'high_hu_fraction_mean': 'Proportion of brain pixels in the 55–90 HU range — the primary blood-brightness proxy',
    'hu_p90_mean':           '90th percentile Hounsfield Unit — elevated when bright blood regions are present',
    'sharpness_mean':        'Edge sharpness of the image — differs between BRIVO (STND kernel) and Revolution (SOFT kernel)',
    'noise_estimate_mean':   'Standard deviation of HU in the air background — captures scanner-specific noise profile',
    'glcm_contrast_mean':    'GLCM contrast — measures local intensity variation in texture',
    'glcm_homogeneity_mean': 'GLCM homogeneity — smoother textures score higher; differs between scanner kernels',
    'hu_mean_mean':          'Mean Hounsfield Unit across brain-masked pixels',
    'hu_std_mean':           'Standard deviation of HU — spread of brightness values in brain tissue',
}

THUMBNAIL_DIR = Path('results/thumbnails')
DATA_DIR      = Path('data') / 'data'
RESULTS_DIR   = Path('results')
MODELS_DIR    = Path('models')

# ─────────────────────────────────────────────────────────────────
# CSS INJECTION — refined visual styling + dark mode support
# ─────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown(f"""
    <style>
    :root {{
        --bg:           {COLOURS['bg']};
        --navbar-bg:    {COLOURS['navbar_bg']};
        --navbar-text:  {COLOURS['navbar_text']};
        --accent:       {COLOURS['accent']};
        --brivo-color:  {COLOURS['brivo']};
        --rev-color:    {COLOURS['revolution']};
        --normal-bg:    {COLOURS['normal_bg']};
        --haem-bg:      {COLOURS['haem_bg']};
        --card-bg:      {COLOURS['card_bg']};
        --text-dim:     {COLOURS['text_dim']};
        --border-color: {COLOURS['border']};
    }}

    .stApp {{
        background-color: var(--bg);
    }}

    h1, h2, h3 {{
        color: {COLOURS['navbar_bg']};
    }}

    hr {{
        border-color: var(--border-color);
    }}

    /* ── Top navigation bar ── */
    .top-navbar {{
        background: {COLOURS['navbar_bg']};
        padding: 0.8rem 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        color: white;
        border-radius: 0 0 8px 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    .top-navbar .title {{
        font-size: 1.4rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .nav-links {{
        display: flex;
        gap: 1rem;
    }}

    /* ── Hero section ── */
    .hero-block {{
        background: linear-gradient(135deg, {COLOURS['navbar_bg']}, {COLOURS['accent']});
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 28px;
        color: white;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    }}
    .hero-block h1 {{
        color: white !important;
        margin: 0 0 8px 0;
        font-size: 2.2rem;
    }}
    .hero-block p {{
        color: #bbdefb;
        font-size: 1.08rem;
        margin: 0;
    }}

    /* ── Section header ── */
    .section-header {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {COLOURS['navbar_bg']};
        border-bottom: 2px solid {COLOURS['accent']};
        padding-bottom: 8px;
        margin: 28px 0 14px 0;
    }}

    /* ── Metric cards ── */
    [data-testid="metric-container"] {{
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}

    /* ── Expander ── */
    .streamlit-expanderHeader {{
        background-color: var(--normal-bg);
        border-radius: 8px;
    }}

    /* ── Patient cards ── */
    .patient-card {{
        border-radius: 12px;
        padding: 12px 8px;
        margin: 6px 2px;
        text-align: center;
        font-size: 11px;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    .patient-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }}

    /* ── Confound badges ── */
    .badge-confounded {{
        background-color: #ffebee; color: #c62828;
        padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 700;
    }}
    .badge-reduced {{
        background-color: #fffde7; color: #f57f17;
        padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 700;
    }}
    .badge-clean {{
        background-color: #e8f5e9; color: #2e7d32;
        padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 700;
    }}

    /* ── Highlight / warning boxes ── */
    .highlight-box {{
        background: linear-gradient(135deg, {COLOURS['normal_bg']}, #ffffff);
        border-left: 4px solid {COLOURS['accent']};
        border-radius: 8px;
        padding: 16px 22px;
        margin: 14px 0;
    }}
    .finding-box {{
        background-color: {COLOURS['card_bg']};
        border: 1px solid {COLOURS['border']};
        border-left: 4px solid {COLOURS['accent']};
        border-radius: 10px;
        padding: 16px 20px;
        margin: 10px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}
    .warning-overlap {{
        background-color: #fff3e0;
        border-left: 4px solid #e65100;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
    }}

    /* ── Disclaimer ── */
    .disclaimer {{
        font-size: 11px;
        color: {COLOURS['text_dim']};
        font-style: italic;
        text-align: center;
        margin-top: 10px;
    }}

    /* ── Concluding box ── */
    .conclusion-box {{
        background: linear-gradient(135deg, {COLOURS['normal_bg']}, #ffffff);
        border: 2px solid {COLOURS['accent']};
        border-radius: 12px;
        padding: 22px 28px;
        margin-top: 28px;
        text-align: center;
    }}
    .conclusion-box h3 {{
        color: {COLOURS['navbar_bg']} !important;
        margin: 0 0 12px 0;
    }}

    /* ─────────────────────────────────────────────────────────────
       DARK MODE OVERRIDES
       ───────────────────────────────────────────────────────────── */
    [data-baseweb="dark"] .stApp {{
        --bg: #1a1a2e;
        --card-bg: #2a2a3e;
        --normal-bg: #2a3a4a;
        --haem-bg: #4a2a2a;
        --text-dim: #aaa;
        --border-color: #555;
        color: #e0e0e0;
    }}
    [data-baseweb="dark"] .hero-block h1,
    [data-baseweb="dark"] .hero-block p,
    [data-baseweb="dark"] .section-header,
    [data-baseweb="dark"] h1, [data-baseweb="dark"] h2, [data-baseweb="dark"] h3 {{
        color: #bbdefb !important;
    }}
    [data-baseweb="dark"] .badge-confounded {{ background: #4a2020; color: #ff8a80; }}
    [data-baseweb="dark"] .badge-reduced   {{ background: #3e3a20; color: #ffd54f; }}
    [data-baseweb="dark"] .badge-clean     {{ background: #1e3a2a; color: #a5d6a7; }}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# DATA LOADING — cached
# ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_patients():
    path = DATA_DIR / 'features_patients.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_slices():
    path = DATA_DIR / 'features_slices.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_experiments():
    path = RESULTS_DIR / 'experiment_results.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_scanner_offset():
    path = DATA_DIR / 'scanner_offset_vector.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_shap_full():
    path = RESULTS_DIR / 'shap_values_full.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_shap_revolution():
    path = RESULTS_DIR / 'shap_values_revolution.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_pca():
    path = DATA_DIR / 'pca_coordinates.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_shap_stability():
    path = RESULTS_DIR / 'shap_stability_table.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_model(model_path):
    path = MODELS_DIR / model_path
    if not path.exists():
        return None
    return joblib.load(path)


def get_thumbnail_path(patient_id, masked=False):
    suffix = '_masked.png' if masked else '_slice.png'
    return THUMBNAIL_DIR / f"{patient_id}{suffix}"


def get_feat_cols(df):
    return [c for c in df.columns if c.endswith('_mean') or c.endswith('_max')]


def get_rf_results(df_exp, exp_key):
    """Get Random Forest result row for a given experiment key substring."""
    if df_exp is None:
        return None
    exp_mask = df_exp['experiment'].str.contains(exp_key, na=False)
    if 'classifier' in df_exp.columns:
        clf_mask = df_exp['classifier'].eq('RandomForest')
        rows = df_exp[exp_mask & clf_mask]
        if len(rows) == 0:
            rows = df_exp[exp_mask]
    else:
        rows = df_exp[exp_mask]
    return rows.iloc[0] if len(rows) > 0 else None


def format_auc(row):
    if row is None:
        return 'N/A'
    auc = row.get('auc_mean', float('nan'))
    lo  = row.get('ci_lower', float('nan'))
    hi  = row.get('ci_upper', float('nan'))
    if pd.isna(auc):
        return 'N/A'
    if pd.isna(lo) or pd.isna(hi):
        return f'{auc:.3f}'
    return f'{auc:.3f} [{lo:.3f}–{hi:.3f}]'


# ─────────────────────────────────────────────────────────────────
# TOP NAVIGATION BAR (replaces sidebar, includes project title)
# ─────────────────────────────────────────────────────────────────

def render_top_nav():
    # Initialise page state first
    if "page" not in st.session_state:
        st.session_state["page"] = "trap"

    # Hide sidebar
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="stSidebarCollapsedControl"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

    # Custom top navbar matching prototype design
    st.markdown(f"""
    <div class="top-navbar">
        <div class="title">
            <span style="font-size:1.8rem;">🧠</span>
            Bias‑Aware Radiomic Classification of Brain CT Haemorrhage
        </div>
        <div class="nav-links">
            <!-- buttons are placed via Streamlit columns below -->
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation buttons just below the title bar (inline with the navbar design)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🎯 The Scanner Trap", use_container_width=True,
                     type="primary" if st.session_state["page"] == "trap" else "secondary"):
            st.session_state["page"] = "trap"
            st.rerun()
    with col2:
        if st.button("🔍 Detect Haemorrhage", use_container_width=True,
                     type="primary" if st.session_state["page"] == "detect" else "secondary"):
            st.session_state["page"] = "detect"
            st.rerun()
    with col3:
        if st.button("🔬 Bias Investigation Lab", use_container_width=True,
                     type="primary" if st.session_state["page"] == "lab" else "secondary"):
            st.session_state["page"] = "lab"
            st.rerun()

    # Separator line
    st.markdown("<hr style='margin-top:0.5rem; border-color: var(--border-color);'>",
                unsafe_allow_html=True)

    return st.session_state["page"]


# ─────────────────────────────────────────────────────────────────
# PAGE 1 — THE SCANNER TRAP
# ─────────────────────────────────────────────────────────────────

def page_scanner_trap():

    st.markdown(f"""
    <div class="hero-block">
        <h1>🎯 The Scanner Trap</h1>
        <p>A machine learning model trained on this dataset may detect the
        <strong>scanner brand</strong> instead of <strong>brain haemorrhage</strong>.
        This page shows how and why.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── CT Image Comparison ────────────────────────────────────────
    st.markdown('<div class="section-header">Normal Brain VS Haemorrhage — Two Different Scanners</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        brivo_path = get_thumbnail_path('CT20015788')
        if brivo_path.exists():
            st.image(str(brivo_path), use_container_width=True)
        st.markdown(f"""
        <div style="background:{COLOURS['normal_bg']};border:2px solid {COLOURS['brivo']};
                    border-radius:8px;padding:10px;text-align:center;margin-top:6px;">
            <strong style="color:{COLOURS['brivo']};">BRIVO CT325</strong><br>
            <span style="color:{COLOURS['text_dim']};">Patient CT20015788 · Normal · STND kernel</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        rev_path = get_thumbnail_path('CT3225-12-2025')
        if rev_path.exists():
            st.image(str(rev_path), use_container_width=True)
        st.markdown(f"""
        <div style="background:{COLOURS['haem_bg']};border:2px solid {COLOURS['revolution']};
                    border-radius:8px;padding:10px;text-align:center;margin-top:6px;">
            <strong style="color:{COLOURS['revolution']};">Revolution ACTs</strong><br>
            <span style="color:{COLOURS['text_dim']};">Patient CT3225 · Haemorrhage · SOFT kernel</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight-box">
        Different scanners produce images with different noise patterns, sharpness, and texture —
        even when imaging the same type of brain tissue. A model can exploit these hardware differences
        as shortcuts instead of learning real pathology.
    </div>
    """, unsafe_allow_html=True)

    # ── Patient Grid ───────────────────────────────────────────────
    st.markdown('<div class="section-header">The Confound at a Glance</div>',
                unsafe_allow_html=True)

    st.markdown("""
    Each card below represents one patient. **Border colour = scanner type.
    Background colour = diagnosis.**
    Notice that every blue-bordered (BRIVO) card has a blue (Normal) background.
    """)

    patients_list = list(PATIENT_META.keys())
    cols = st.columns(5)

    for i, pid in enumerate(patients_list):
        meta   = PATIENT_META[pid]
        is_brivo  = meta['scanner'] == 'BRIVO'
        is_haem   = meta['class'] == 'Haemorrhage'
        bg_col    = COLOURS['haem_bg']   if is_haem   else COLOURS['normal_bg']
        border_col = COLOURS['revolution'] if not is_brivo else COLOURS['brivo']
        scanner_label = 'BRIVO' if is_brivo else 'Revolution'
        border_style  = 'dashed' if is_brivo else 'solid'
        class_icon    = '🔴' if is_haem else '🟢'

        with cols[i % 5]:
            st.markdown(f"""
            <div style="background:{bg_col};border:3px {border_style} {border_col};
                        border-radius:10px;padding:10px 6px;text-align:center;
                        margin-bottom:8px;min-height:90px;">
                <div style="font-size:18px;">{class_icon}</div>
                <div style="font-size:10px;font-weight:700;color:#37474f;
                            word-break:break-all;">{pid[:12]}</div>
                <div style="font-size:10px;color:{border_col};font-weight:600;">
                    {scanner_label}</div>
                <div style="font-size:10px;color:{COLOURS['text_dim']};">
                    {meta['class']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{COLOURS['haem_bg']};border-left:4px solid #c62828;
                border-radius:6px;padding:12px 16px;margin-top:8px;">
        <strong>⚠️ Confirmed Bias:</strong> All 3 BRIVO-bordered cards are Normal.
        No BRIVO patient has Haemorrhage. Any model can exploit this pattern
        without learning real haemorrhage features.
    </div>
    """, unsafe_allow_html=True)

    # ── Find the Shortcut Widget ───────────────────────────────────
    st.markdown('<div class="section-header">Find the Shortcut</div>',
                unsafe_allow_html=True)

    st.markdown("Select a feature and see how well it separates patients by scanner — "
                "even when restricting to Normal patients only.")

    df_pat = load_patients()
    if df_pat is not None:
        feat_cols = get_feat_cols(df_pat)

        col_a, col_b = st.columns([3, 1])
        with col_a:
            selected_feat = st.selectbox(
                "Select feature to inspect:",
                feat_cols,
                index=feat_cols.index('sharpness_mean') if 'sharpness_mean' in feat_cols else 0
            )
        with col_b:
            normal_only = st.checkbox("Normal patients only", value=False)

        df_plot = df_pat[df_pat['class_label'] == 0].copy() if normal_only else df_pat.copy()

        if selected_feat in df_plot.columns:
            colour_map = {'BRIVO CT325': COLOURS['brivo'],
                          'Revolution ACTs': COLOURS['revolution']}

            fig = go.Figure()
            for scanner, group in df_plot.groupby('scanner_model'):
                fig.add_trace(go.Scatter(
                    x=group[selected_feat],
                    y=group['patient_id'],
                    mode='markers',
                    name=scanner,
                    marker=dict(
                        color=colour_map.get(scanner, '#888'),
                        size=14,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate=(
                        f"<b>%{{y}}</b><br>{selected_feat}: %{{x:.4f}}<br>"
                        f"Scanner: {scanner}<extra></extra>"
                    )
                ))

            fig.update_layout(
                title=dict(
                    text=f"{selected_feat} by Patient" +
                         (" (Normal patients only)" if normal_only else ""),
                    font=dict(color=COLOURS['navbar_bg'])
                ),
                xaxis_title=selected_feat,
                yaxis_title="Patient",
                plot_bgcolor=COLOURS['bg'],
                paper_bgcolor=COLOURS['bg'],
                height=380,
                legend=dict(title="Scanner"),
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            if normal_only:
                st.markdown(
                    "When only Normal patients are shown, any feature separation "
                    "you see is **purely scanner-induced** — pathology cannot explain it."
                )

    # ── Cheat Ceiling ─────────────────────────────────────────────
    st.markdown('<div class="section-header">The Cheat Ceiling</div>',
                unsafe_allow_html=True)

    df_exp = load_experiments()
    exp1_row = get_rf_results(df_exp, 'Exp1') if df_exp is not None else None

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        auc1_val = exp1_row.get('auc_mean', float('nan')) if exp1_row is not None else float('nan')
        auc1_str = f"{auc1_val:.3f}" if not pd.isna(auc1_val) else "N/A"
        st.metric(
            label="🔴 Scanner-Only Baseline AUC",
            value=auc1_str,
            delta="CONFOUNDED — zero image features used",
            delta_color="off"
        )
    with col_m2:
        st.metric(label="Features used", value="0", delta="Scanner name only")
    with col_m3:
        st.metric(label="Patients", value="10", delta="Full dataset")

    with st.expander("What does the Cheat Ceiling mean?"):
        st.markdown(f"""
        The scanner-only baseline predicts **Normal** for all BRIVO patients
        and uses the majority class for Revolution patients — without looking at
        a single pixel of the CT image. If this achieves AUC = **{auc1_str}**,
        then any classifier that scores at or below this value is not learning
        anything beyond scanner identity.

        Any model you trust must **substantially exceed** this number — and even
        then you need to check whether it is using scanner shortcuts.
        """)

    # ── Experiment Comparison ─────────────────────────────────────
    st.markdown('<div class="section-header">Apparent vs Honest Performance</div>',
                unsafe_allow_html=True)

    if df_exp is not None:
        exp_choice = st.radio(
            "Compare against:",
            ["Full Dataset — Exp 2 (CONFOUNDED)",
             "Revolution-Only — Exp 3 (REDUCED CONFOUND)"],
            horizontal=True
        )

        exp_key    = 'Exp2' if 'Exp 2' in exp_choice else 'Exp3'
        chosen_row = get_rf_results(df_exp, exp_key)
        cheat_auc  = auc1_val

        if chosen_row is not None:
            chosen_auc    = chosen_row.get('auc_mean', float('nan'))
            confound_stat = chosen_row.get('confound_status', 'UNKNOWN')
            badge_label   = CONFOUND_LABELS.get(confound_stat, confound_stat)

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=[cheat_auc if not pd.isna(cheat_auc) else 0],
                y=['Scanner-Only Baseline'],
                orientation='h',
                marker_color='#ef9a9a',
                name='Cheat Ceiling 🔴 CONFOUNDED',
                text=[f"AUC = {cheat_auc:.3f}" if not pd.isna(cheat_auc) else "N/A"],
                textposition='inside',
            ))
            fig2.add_trace(go.Bar(
                x=[chosen_auc if not pd.isna(chosen_auc) else 0],
                y=[f"Random Forest ({badge_label})"],
                orientation='h',
                marker_color=COLOURS['accent'] if 'Exp2' in exp_key else '#66bb6a',
                name=badge_label,
                text=[f"AUC = {chosen_auc:.3f}" if not pd.isna(chosen_auc) else "N/A"],
                textposition='inside',
            ))

            fig2.update_layout(
                title="AUC Comparison — Cheat Ceiling vs Selected Model",
                xaxis=dict(range=[0, 1.05], title="AUC"),
                plot_bgcolor=COLOURS['bg'],
                paper_bgcolor=COLOURS['bg'],
                height=220,
                showlegend=True,
                margin=dict(l=20, r=20, t=50, b=20)
            )

            # FIX 6: Add confound annotation to the bar chart
            fig2.add_annotation(
                x=0.02,
                y=1.08,
                xref='paper',
                yref='paper',
                text="🔴 Red bar = CONFOUNDED (scanner shortcut available) | "
                     "🟢/🔵 Coloured bar = label shown in legend",
                showarrow=False,
                font=dict(size=10, color=COLOURS['text_dim']),
                align='left'
            )

            st.plotly_chart(fig2, use_container_width=True)

            r2 = get_rf_results(df_exp, 'Exp2')
            r3 = get_rf_results(df_exp, 'Exp3')
            if r2 is not None and r3 is not None:
                bias = r2.get('auc_mean', 0) - r3.get('auc_mean', 0)
                st.metric(
                    label="📊 Performance Inflation from Scanner Bias",
                    value=f"{bias:.3f}",
                    delta="Exp 2 AUC minus Exp 3 AUC (Random Forest)",
                    delta_color="inverse"
                )
                st.markdown(f"""
                <div class="highlight-box">
                    The full-dataset model (Exp 2) scored <strong>{r2.get('auc_mean',0):.3f}</strong>
                    <span class="badge-confounded">CONFOUNDED</span>. When the scanner confound was
                    removed (Exp 3), honest AUC was <strong>{r3.get('auc_mean',0):.3f}</strong>
                    <span class="badge-reduced">REDUCED CONFOUND</span>.
                    The difference of <strong>{bias:.3f}</strong> was inflated by scanner identity.
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PAGE 2 — DETECT HAEMORRHAGE
# ─────────────────────────────────────────────────────────────────

def page_detect():

    st.markdown(f"""
    <div class="hero-block">
        <h1>🔍 Detect Haemorrhage</h1>
        <p>Select a patient to see how the classifier works — and which image features
        drove the prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    df_pat   = load_patients()
    df_slice = load_slices()
    model    = load_model('exp3_RevOnly_RandomForest.pkl')

    # FIX 2: Default probability to avoid UnboundLocalError
    prob = 0.5

    if df_pat is None:
        st.error("features_patients.csv not found. Check data/ folder.")
        return

    feat_cols = get_feat_cols(df_pat)

    # ── Patient Selection Grid ─────────────────────────────────────
    st.markdown('<div class="section-header">Select a Patient</div>',
                unsafe_allow_html=True)

    if 'selected_patient' not in st.session_state:
        st.session_state['selected_patient'] = 'CT3289-12-2025'

    grid_cols = st.columns(5)
    for i, pid in enumerate(PATIENT_META.keys()):
        meta     = PATIENT_META[pid]
        is_brivo = meta['scanner'] == 'BRIVO'
        is_haem  = meta['class'] == 'Haemorrhage'
        is_sel   = (pid == st.session_state['selected_patient'])

        bg_col      = COLOURS['haem_bg']    if is_haem  else COLOURS['normal_bg']
        border_col  = COLOURS['revolution'] if not is_brivo else COLOURS['brivo']
        border_w    = '4px' if is_sel else '2px'
        border_sty  = 'dashed' if is_brivo else 'solid'

        # FIX 1: Selected patient card highlight border
        with grid_cols[i % 5]:
            thumb_path = get_thumbnail_path(pid)
            if thumb_path.exists():
                border_style_img = (f"border: {border_w} {border_sty} {border_col};"
                                    "border-radius:8px;padding:2px;")
                st.markdown(f'<div style="{border_style_img}">',
                            unsafe_allow_html=True)
                st.image(str(thumb_path), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="height:80px;background:{bg_col};'
                    f'border:{border_w} {border_sty} {border_col};'
                    f'border-radius:8px;display:flex;align-items:center;'
                    f'justify-content:center;font-size:20px;">🧠</div>',
                    unsafe_allow_html=True
                )

            if st.button(
                f"{'✅ ' if is_sel else ''}{pid[:10]}",
                key=f"sel_{pid}",
                use_container_width=True
            ):
                st.session_state['selected_patient'] = pid
                st.rerun()

            st.markdown(f"""
            <div style="font-size:10px;text-align:center;color:{COLOURS['text_dim']};
                        margin-top:-6px;">
                {meta['class']} · {meta['scanner']}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Selected Patient Details ───────────────────────────────────
    sel_pid  = st.session_state['selected_patient']
    sel_meta = PATIENT_META[sel_pid]
    sel_row  = df_pat[df_pat['patient_id'] == sel_pid]

    if len(sel_row) == 0:
        st.warning(f"Patient {sel_pid} not found in features_patients.csv")
        return

    sel_row = sel_row.iloc[0]

    st.markdown(f'<div class="section-header">Patient: {sel_pid}</div>',
                unsafe_allow_html=True)

    col_left, col_mid, col_right = st.columns([1.2, 1.2, 1.2])

    # Left — CT slice display
    with col_left:
        st.markdown("**CT Slice**")
        show_mask = st.checkbox("Show brain mask overlay", key="mask_toggle")
        thumb_path = get_thumbnail_path(sel_pid, masked=show_mask)
        if thumb_path.exists():
            st.image(str(thumb_path), use_container_width=True,
                     caption=f"{'Masked' if show_mask else 'Raw'} slice — {sel_pid}")
        else:
            st.info("Thumbnail not found. Ask Jyothi to generate thumbnails.")

        st.markdown(f"""
        <div style="background:{COLOURS['normal_bg'] if sel_meta['class']=='Normal' else COLOURS['haem_bg']};
                    border-radius:6px;padding:8px;font-size:12px;text-align:center;">
            <strong>Class:</strong> {sel_meta['class']}<br>
            <strong>Scanner:</strong> {sel_meta['scanner']}<br>
            <strong>Slices processed:</strong> {int(sel_row.get('n_slices', 0))}
        </div>
        """, unsafe_allow_html=True)

    # Middle — Feature cards
    with col_mid:
        st.markdown("**Key Radiomic Features**")
        normal_means = df_pat[df_pat['class_label'] == 0][feat_cols].mean()

        key_features = [
            'high_hu_fraction_mean',
            'hu_p90_mean',
            'sharpness_mean',
            'noise_estimate_mean'
        ]

        for feat in key_features:
            if feat not in sel_row.index:
                continue
            val      = sel_row[feat]
            norm_val = normal_means.get(feat, float('nan'))
            above    = val > norm_val if not pd.isna(norm_val) else None
            arrow    = "↑ above normal" if above else "↓ below normal"
            delta_col = ("#c62828" if above and 'high_hu' in feat
                         else "#2e7d32" if not above and 'high_hu' in feat
                         else COLOURS['text_dim'])

            tooltip = FEATURE_TOOLTIPS.get(feat, "")
            st.metric(
                label=feat.replace('_mean','').replace('_',' ').title(),
                value=f"{val:.4f}",
                delta=arrow,
                delta_color="inverse" if ('high_hu' in feat or 'hu_p90' in feat) else "off",
                help=tooltip
            )

    # Right — Prediction gauge
    with col_right:
        st.markdown("**Haemorrhage Risk**")

        if model is not None and len(feat_cols) > 0:
            try:
                x_input = sel_row[feat_cols].values.reshape(1, -1)
                prob    = model.predict_proba(x_input)[0][1]
            except Exception:
                prob = 0.5
        else:
            prob = 0.5

        gauge_colour = '#c62828' if prob >= 0.5 else '#2e7d32'
        pred_label   = "🔴 HAEMORRHAGE" if prob >= 0.5 else "🟢 NORMAL"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={'suffix': '', 'valueformat': '.3f',
                    'font': {'size': 28, 'color': gauge_colour}},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1},
                'bar': {'color': gauge_colour},
                'steps': [
                    {'range': [0, 0.5],  'color': '#e8f5e9'},
                    {'range': [0.5, 1],  'color': '#ffebee'},
                ],
                'threshold': {
                    'line': {'color': '#37474f', 'width': 3},
                    'thickness': 0.75,
                    'value': 0.5
                }
            },
            title={'text': "Haemorrhage Probability",
                   'font': {'color': COLOURS['navbar_bg']}}
        ))
        fig_gauge.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=50, b=10),
            paper_bgcolor=COLOURS['bg']
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div style="text-align:center;font-size:1.1rem;font-weight:700;
                    color:{gauge_colour};padding:8px;">
            {pred_label}
        </div>
        <div style="text-align:center;font-size:11px;color:{COLOURS['text_dim']};">
            Model: Revolution-Only RF
            <span class="badge-reduced">REDUCED CONFOUND</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="disclaimer">⚠️ Research demonstration, not a diagnostic tool.</p>',
                    unsafe_allow_html=True)

    # ── Slice-Level Trend ─────────────────────────────────────────
    st.markdown('<div class="section-header">High-HU Fraction Across Slices</div>',
                unsafe_allow_html=True)

    if df_slice is not None and 'high_hu_fraction' in df_slice.columns:
        df_sel_slices = df_slice[
            df_slice['patient_id'] == sel_pid
        ].sort_values('instance_number')

        if len(df_sel_slices) > 0:
            normal_mean_slice = df_slice[
                df_slice['class_label'] == 0
            ]['high_hu_fraction'].mean()

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=df_sel_slices['instance_number'],
                y=df_sel_slices['high_hu_fraction'],
                mode='lines+markers',
                name=sel_pid,
                line=dict(
                    color=COLOURS['revolution'] if sel_meta['class'] == 'Haemorrhage'
                    else COLOURS['brivo'],
                    width=2
                ),
                marker=dict(size=5)
            ))
            fig_trend.add_hline(
                y=normal_mean_slice,
                line_dash='dash',
                line_color=COLOURS['text_dim'],
                annotation_text=f"Normal class mean ({normal_mean_slice:.4f})",
                annotation_position="top right"
            )
            fig_trend.update_layout(
                title=f"High-HU Fraction per Slice — {sel_pid}",
                xaxis_title="Slice Number (Instance Number)",
                yaxis_title="high_hu_fraction",
                plot_bgcolor=COLOURS['bg'],
                paper_bgcolor=COLOURS['bg'],
                height=280,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            st.markdown(
                "Slices above the dashed line have more pixels in the blood-bright HU range (55–90). "
                "Haemorrhage patients typically show peaks where bleeding is visible."
            )
        else:
            st.info("No slice-level data found for this patient in features_slices.csv")
    else:
        st.info("features_slices.csv not available or missing high_hu_fraction column.")

  

    # ── Side-by-Side Comparison ───────────────────────────────────
    st.markdown('<div class="section-header">Compare with Average Normal Brain</div>',
                unsafe_allow_html=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        sp = get_thumbnail_path(sel_pid)
        if sp.exists():
            st.image(str(sp), caption=f"Selected: {sel_pid} ({sel_meta['class']})",
                     use_container_width=True)

    with col_c2:
        avg_path = THUMBNAIL_DIR / 'average_normal.png'
        if avg_path.exists():
            st.image(str(avg_path), caption="Average Normal Brain (5 Normal patients)",
                     use_container_width=True)

    # ── Explainer ─────────────────────────────────────────────────
    with st.expander("❓ How do we know the model isn't cheating on this patient?"):
        st.markdown(f"""
        Great question. For patient **{sel_pid}** ({sel_meta['scanner']} scanner):

        - The prediction above uses the **Revolution-only model (Experiment 3)**,
          which was trained without any BRIVO patients.
        - This means the BRIVO scanner shortcut was **not available** during training.
        - The model had to learn from image features alone — no scanner identity clue.

        However, scanner fingerprints (sharpness, noise texture, GLCM homogeneity)
        may still influence predictions even within the Revolution scanner group.

        Go to the **Bias Investigation Lab** page for the full audit showing which
        features carry scanner identity and how much they overlap with haemorrhage features.
        """)


# ─────────────────────────────────────────────────────────────────
# PAGE 3 — BIAS INVESTIGATION LAB
# ─────────────────────────────────────────────────────────────────

def page_bias_lab():

    st.markdown(f"""
    <div class="hero-block">
        <h1>🔬 Bias Investigation Lab</h1>
        <p>How much of the model's performance was genuine vs scanner-driven?
        All six experiments are dissected here.</p>
    </div>
    """, unsafe_allow_html=True)

    df_exp    = load_experiments()
    df_offset = load_scanner_offset()
    df_shap_f = load_shap_full()
    df_shap_r = load_shap_revolution()
    df_pca    = load_pca()
    df_stab   = load_shap_stability()
    df_pat    = load_patients()

    feat_cols = get_feat_cols(df_pat) if df_pat is not None else []

    # ── Scanner Fingerprint Chart ──────────────────────────────────
    st.markdown('<div class="section-header">Scanner Fingerprint — Top Features</div>',
                unsafe_allow_html=True)

    if df_offset is not None:
        top_offset = df_offset.nlargest(10, 'abs_offset')
        colours_bar = [
            COLOURS['brivo'] if v > 0 else COLOURS['revolution']
            for v in top_offset['scanner_offset']
        ]

        fig_fp = go.Figure(go.Bar(
            x=top_offset['scanner_offset'],
            y=top_offset['feature'],
            orientation='h',
            marker_color=colours_bar,
            hovertemplate="Feature: %{y}<br>Offset: %{x:.4f}<extra></extra>"
        ))
        fig_fp.add_vline(x=0, line_width=1, line_color=COLOURS['text_dim'])
        fig_fp.update_layout(
            title=dict(
                text="Top 10 Features by Scanner Offset (BRIVO minus Revolution)<br>"
                     "<sub>Blue=BRIVO higher | Orange=Revolution higher</sub>",
                font=dict(color=COLOURS['navbar_bg'])
            ),
            xaxis_title="Scanner Offset",
            yaxis=dict(autorange='reversed'),
            plot_bgcolor=COLOURS['bg'],
            paper_bgcolor=COLOURS['bg'],
            height=380,
            margin=dict(l=20, r=20, t=70, b=20)
        )
        st.plotly_chart(fig_fp, use_container_width=True)
        st.markdown(
            "Features with large offsets have different values depending on scanner brand — "
            "these are the **scanner fingerprints**. If a haemorrhage classifier relies on "
            "these same features, it is partly detecting scanner identity, not pathology."
        )
    else:
        st.info("scanner_offset_vector.csv not found. Ask Lahari to share it.")

    # ── Feature Overlap ───────────────────────────────────────────
    st.markdown('<div class="section-header">Feature Overlap: Haemorrhage vs Scanner Fingerprint</div>',
                unsafe_allow_html=True)

    col_ov1, col_ov2 = st.columns(2)

    top_shap_features    = []
    top_scanner_features = []

    with col_ov1:
        st.markdown(f"**Top SHAP Features — Full Dataset**")
        st.markdown(f"<small><span class='badge-confounded'>CONFOUNDED</span></small>",
                    unsafe_allow_html=True)
        if df_shap_f is not None:
            shap_feat_cols = [c for c in df_shap_f.columns if c != 'patient_id']
            mean_abs_shap  = df_shap_f[shap_feat_cols].abs().mean().sort_values(ascending=False)
            top_shap       = mean_abs_shap.head(10)
            top_shap_features = list(top_shap.index)

            fig_shap = go.Figure(go.Bar(
                x=top_shap.values,
                y=top_shap.index,
                orientation='h',
                marker_color=COLOURS['accent'],
                hovertemplate="Feature: %{y}<br>Mean |SHAP|: %{x:.4f}<extra></extra>"
            ))
            fig_shap.update_layout(
                xaxis_title="Mean |SHAP value|",
                yaxis=dict(autorange='reversed'),
                plot_bgcolor=COLOURS['bg'],
                paper_bgcolor=COLOURS['bg'],
                height=350,
                margin=dict(l=10, r=10, t=20, b=20)
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("shap_values_full.csv not found.")

    with col_ov2:
        st.markdown("**Top Scanner Fingerprint Features**")
        st.markdown(f"<small><span class='badge-clean'>CLEAN (Normal patients only)</span></small>",
                    unsafe_allow_html=True)
        if df_offset is not None:
            top_scan_offset   = df_offset.nlargest(10, 'abs_offset')
            top_scanner_features = list(top_scan_offset['feature'])

            fig_scan = go.Figure(go.Bar(
                x=top_scan_offset['abs_offset'],
                y=top_scan_offset['feature'],
                orientation='h',
                marker_color=COLOURS['revolution'],
                hovertemplate="Feature: %{y}<br>|Offset|: %{x:.4f}<extra></extra>"
            ))
            fig_scan.update_layout(
                xaxis_title="Absolute Scanner Offset",
                yaxis=dict(autorange='reversed'),
                plot_bgcolor=COLOURS['bg'],
                paper_bgcolor=COLOURS['bg'],
                height=350,
                margin=dict(l=10, r=10, t=20, b=20)
            )
            st.plotly_chart(fig_scan, use_container_width=True)
        else:
            st.info("scanner_offset_vector.csv not found.")

    # Overlap detection
    if top_shap_features and top_scanner_features:
        overlap = set(top_shap_features) & set(top_scanner_features)
        if overlap:
            st.markdown(f"""
            <div class="warning-overlap">
                ⚠️ <strong>{len(overlap)} feature(s) appear in BOTH lists:</strong>
                {', '.join(sorted(overlap))}<br>
                These features drive both scanner identification AND haemorrhage classification —
                direct evidence that the full-dataset model is partially using scanner shortcuts.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success(
                "✅ No overlap found between top haemorrhage SHAP features and "
                "top scanner fingerprint features in this analysis."
            )

    with st.expander("📊 SHAP Stability Caveat"):
        st.markdown("""
        SHAP rankings above are computed from the full-dataset model trained on all 10 patients.
        With such a small sample size, feature importance rankings can shift between LOOCV folds.

        **Interpretation guideline:**
        - Features appearing in top-10 across ≥7/10 LOOCV folds = **ROBUST**
        - Features in top-10 across 4–6 folds = **SUGGESTIVE**
        - Features in top-10 across ≤3 folds = **UNSTABLE** — do not over-interpret

        These rankings are indicative, not definitive. See shap_stability_table.csv for fold-level consistency.
        """)

        if df_stab is not None:
            st.dataframe(
                df_stab.sort_values('fold_count', ascending=False),
                use_container_width=True, height=200
            )

    # ── PCA Scatter ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Feature Space — PCA Projection</div>',
                unsafe_allow_html=True)

    if df_pca is not None:
        pcol1, pcol2 = st.columns([1, 3])
        with pcol1:
            show_normal_only = st.checkbox("Show only Normal patients", value=False)
            show_synthetic   = st.checkbox("Show synthetic haemorrhage points", value=True)

        df_pca_plot = df_pca.copy()
        if show_normal_only:
            df_pca_plot = df_pca_plot[df_pca_plot['class_label'] == 0]
        if not show_synthetic and 'is_synthetic' in df_pca_plot.columns:
            df_pca_plot = df_pca_plot[~df_pca_plot['is_synthetic']]

        symbol_map = {0: 'circle', 1: 'diamond'}
        colour_map = {
            'BRIVO CT325':    COLOURS['brivo'],
            'Revolution ACTs': COLOURS['revolution']
        }

        fig_pca = go.Figure()

        for scanner, scanner_grp in df_pca_plot.groupby('scanner_model'):
            for is_synth, synth_grp in scanner_grp.groupby(
                'is_synthetic' if 'is_synthetic' in scanner_grp.columns else 'class_label'
            ):
                # FIX 3: Handle both boolean and string True/False for is_synthetic
                if 'is_synthetic' in df_pca_plot.columns:
                    if isinstance(is_synth, str):
                        synth_flag = is_synth.lower() == 'true'
                    else:
                        synth_flag = bool(is_synth)
                else:
                    synth_flag = False

                for cls_lbl, cls_grp in synth_grp.groupby('class_label'):
                    marker_sym  = 'star' if synth_flag else symbol_map.get(cls_lbl, 'circle')
                    marker_col  = colour_map.get(scanner, '#888')
                    marker_opac = 0.5 if synth_flag else 1.0
                    marker_size = 10 if synth_flag else 14
                    trace_name  = (f"{scanner} · {'Haem' if cls_lbl==1 else 'Normal'}"
                                   + (" (synthetic)" if synth_flag else ""))

                    fig_pca.add_trace(go.Scatter(
                        x=cls_grp['pc1'],
                        y=cls_grp['pc2'],
                        mode='markers',
                        name=trace_name,
                        marker=dict(
                            color=marker_col,
                            symbol=marker_sym,
                            size=marker_size,
                            opacity=marker_opac,
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Class: %{customdata[1]}<br>"
                            "Scanner: %{customdata[2]}<br>"
                            "PC1: %{x:.3f}, PC2: %{y:.3f}"
                            "<extra></extra>"
                        ),
                        customdata=cls_grp[['patient_id','class_name','scanner_model']].values
                    ))

        fig_pca.update_layout(
            title="PCA Feature Space — All Patients<br>"
                  "<sub>Circle=Normal · Diamond=Haemorrhage · Star=Synthetic</sub>",
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            plot_bgcolor=COLOURS['bg'],
            paper_bgcolor=COLOURS['bg'],
            height=420,
            margin=dict(l=20, r=20, t=70, b=20)
        )
        st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.info("pca_coordinates.csv not found. Ask Lahari to generate it.")

    # ── Experiment Results Table ───────────────────────────────────
    st.markdown('<div class="section-header">Six-Experiment Results</div>',
                unsafe_allow_html=True)

    st.markdown(
        "Every AUC is labelled by whether scanner bias was present during evaluation. "
        "Only Random Forest results are shown here for clarity."
    )

    if df_exp is not None:
        if 'classifier' in df_exp.columns:
            df_rf = df_exp[df_exp['classifier'] == 'RandomForest'].copy()
        else:
            df_rf = df_exp.copy()

        display_cols = ['experiment', 'confound_status', 'n_patients',
                        'auc_mean', 'ci_lower', 'ci_upper',
                        'sensitivity', 'specificity']
        display_cols = [c for c in display_cols if c in df_rf.columns]

        df_display = df_rf[display_cols].copy()

        if 'auc_mean' in df_display.columns:
            df_display['AUC [95% CI]'] = df_display.apply(
                lambda r: (f"{r['auc_mean']:.3f} [{r.get('ci_lower',float('nan')):.3f}–"
                           f"{r.get('ci_upper',float('nan')):.3f}]"
                           if not pd.isna(r['auc_mean']) else 'N/A'), axis=1
            )

        if 'confound_status' in df_display.columns:
            df_display['Status'] = df_display['confound_status'].map(
                CONFOUND_LABELS
            ).fillna(df_display['confound_status'])

        show_cols = ['experiment', 'Status', 'n_patients', 'AUC [95% CI]']
        if 'sensitivity' in df_display.columns:
            show_cols.append('sensitivity')
        if 'specificity' in df_display.columns:
            show_cols.append('specificity')
        show_cols = [c for c in show_cols if c in df_display.columns]

        st.dataframe(
            df_display[show_cols].reset_index(drop=True),
            use_container_width=True,
            height=280
        )
    else:
        st.info("experiment_results.csv not found. Ask Lahari to share it.")

    # ── Bias Magnitude and Augmentation Effect ─────────────────────
    st.markdown('<div class="section-header">Key Metrics</div>',
                unsafe_allow_html=True)

    if df_exp is not None:
        r1 = get_rf_results(df_exp, 'Exp1')
        r2 = get_rf_results(df_exp, 'Exp2')
        r3 = get_rf_results(df_exp, 'Exp3')
        r6 = get_rf_results(df_exp, 'Exp6')

        mk1, mk2, mk3 = st.columns(3)

        with mk1:
            if r2 is not None and r3 is not None:
                auc2  = r2.get('auc_mean', float('nan'))
                auc3  = r3.get('auc_mean', float('nan'))
                bias  = auc2 - auc3 if not (pd.isna(auc2) or pd.isna(auc3)) else float('nan')
                st.metric(
                    label="📊 Bias Magnitude",
                    value=f"{bias:.3f}" if not pd.isna(bias) else "N/A",
                    delta="Exp 2 (confounded) minus Exp 3 (reduced confound)",
                    delta_color="inverse",
                    help="How much scanner identity inflated apparent performance"
                )

        with mk2:
            if r3 is not None and r6 is not None:
                auc3 = r3.get('auc_mean', float('nan'))
                auc6 = r6.get('auc_mean', float('nan'))
                aug  = auc6 - auc3 if not (pd.isna(auc6) or pd.isna(auc3)) else float('nan')
                st.metric(
                    label="🔬 Augmentation Effect",
                    value=f"{aug:+.3f}" if not pd.isna(aug) else "N/A",
                    delta="Exp 6 minus Exp 3 (positive = augmentation helped)",
                    delta_color="normal",
                    help="Whether synthetic BRIVO-Haemorrhage samples improved the classifier"
                )

        with mk3:
            if r1 is not None:
                auc1 = r1.get('auc_mean', float('nan'))
                st.metric(
                    label="🔴 Cheat Ceiling",
                    value=f"{auc1:.3f}" if not pd.isna(auc1) else "N/A",
                    delta="Scanner-only baseline (CONFOUNDED)",
                    delta_color="off",
                    help="Maximum AUC achievable using only scanner identity"
                )

        st.markdown(
            "The **bias magnitude** tells us how much scanner identity inflated apparent performance. "
            "The **augmentation effect** tells us whether synthetic data partially corrected it."
        )

    # ── Bias Gap Bar Chart ─────────────────────────────────────────
    st.markdown('<div class="section-header">AUC Comparison Across Experiments</div>',
                unsafe_allow_html=True)

    if df_exp is not None:
        exp_display_keys = [
            ('Exp1', 'Exp 1\nBaseline',    '#ef9a9a',  'CONFOUNDED'),
            ('Exp2', 'Exp 2\nFull',        '#e57373',  'CONFOUNDED'),
            ('Exp3', 'Exp 3\nRev-Only',    '#81c784',  'REDUCED CONFOUND'),
            ('Exp5_Norm_Exp2', 'Exp 5a\nNorm Full', '#ffb74d', 'CONFOUNDED NORM'),
            ('Exp5_Norm_Exp3', 'Exp 5b\nNorm Rev',  '#aed581', 'REDUCED NORM'),
            ('Exp6', 'Exp 6\nAugmented',   '#4fc3f7',  'SYNTHETIC'),
        ]

        bar_labels = []
        bar_aucs   = []
        bar_errors = []
        bar_cols   = []
        bar_status = []

        for key, label, colour, status in exp_display_keys:
            row = get_rf_results(df_exp, key)
            if row is not None:
                auc = row.get('auc_mean', float('nan'))
                ci_lo = row.get('ci_lower', float('nan'))
                ci_hi = row.get('ci_upper', float('nan'))
                if not pd.isna(auc):
                    err = (ci_hi - auc) if not pd.isna(ci_hi) else 0
                    bar_labels.append(label)
                    bar_aucs.append(auc)
                    bar_errors.append(err)
                    bar_cols.append(colour)
                    bar_status.append(status)

        if bar_aucs:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=bar_labels,
                y=bar_aucs,
                error_y=dict(type='data', array=bar_errors, visible=True),
                marker_color=bar_cols,
                text=[f"{v:.3f}" for v in bar_aucs],
                textposition='outside',
                hovertemplate=(
                    "Experiment: %{x}<br>AUC: %{y:.3f}<extra></extra>"
                )
            ))
            fig_bar.update_layout(
                title="AUC by Experiment (Random Forest) — with 95% Bootstrap CI",
                yaxis=dict(range=[0, 1.1], title="AUC"),
                plot_bgcolor=COLOURS['bg'],
                paper_bgcolor=COLOURS['bg'],
                height=380,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── Apply the Fix Button ───────────────────────────────────────
    st.markdown('<div class="section-header">Apply the Fix</div>',
                unsafe_allow_html=True)

    if st.button("🔧 Apply the Fix — See What Happens", use_container_width=True):
        if df_exp is not None:
            r3 = get_rf_results(df_exp, 'Exp3')
            r6 = get_rf_results(df_exp, 'Exp6')

            if r3 is not None and r6 is not None:
                auc3 = r3.get('auc_mean', float('nan'))
                auc6 = r6.get('auc_mean', float('nan'))
                aug_effect = auc6 - auc3 if not (pd.isna(auc3) or pd.isna(auc6)) else float('nan')

                # CI-safe formatting
                ci3_lo = r3.get('ci_lower', float('nan'))
                ci3_hi = r3.get('ci_upper', float('nan'))
                ci6_lo = r6.get('ci_lower', float('nan'))
                ci6_hi = r6.get('ci_upper', float('nan'))
                ci3_str = f"{ci3_lo:.3f}–{ci3_hi:.3f}" if not (pd.isna(ci3_lo) or pd.isna(ci3_hi)) else "N/A"
                ci6_str = f"{ci6_lo:.3f}–{ci6_hi:.3f}" if not (pd.isna(ci6_lo) or pd.isna(ci6_hi)) else "N/A"

                if not pd.isna(aug_effect):
                    if aug_effect > 0.02:
                        interp = ("✅ Synthetic augmentation **improved** performance. "
                                  "Adding simulated BRIVO-Haemorrhage samples helped "
                                  "the model generalise slightly better across scanners.")
                        interp_col = '#e8f5e9'
                    elif aug_effect < -0.02:
                        interp = ("⚠️ Synthetic augmentation did **not help**. "
                                  "The scanner offset estimated from only 3 vs 2 patients "
                                  "was too noisy to produce useful synthetic samples.")
                        interp_col = '#fff3e0'
                    else:
                        interp = ("ℹ️ Effect was **within the noise floor** — inconclusive "
                                  "at this sample size. The offset vector may be partially "
                                  "useful but we cannot confirm it from 10 patients.")
                        interp_col = '#e3f2fd'

                    st.markdown(f"""
                    <div style="background:{interp_col};border-radius:8px;
                                padding:16px 20px;margin:8px 0;">
                        <strong>Without fix (Exp 3 — Revolution-only):</strong>
                        AUC = {auc3:.3f} [{ci3_str}]
                        <span class="badge-reduced">REDUCED CONFOUND</span><br><br>
                        <strong>With synthetic augmentation (Exp 6):</strong>
                        AUC = {auc6:.3f} [{ci6_str}]
                        <span style="background:#e3f2fd;color:#1565c0;padding:2px 8px;
                                     border-radius:12px;font-size:11px;font-weight:700;">SYNTHETIC</span><br><br>
                        <strong>Change:</strong> {aug_effect:+.3f}<br><br>
                        {interp}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Experiment 3 or 6 results not found in experiment_results.csv")
        else:
            st.warning("experiment_results.csv not loaded.")

    # ── Key Findings ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Key Findings</div>',
                unsafe_allow_html=True)

    if df_exp is not None:
        r1 = get_rf_results(df_exp, 'Exp1')
        r2 = get_rf_results(df_exp, 'Exp2')
        r3 = get_rf_results(df_exp, 'Exp3')
        r6 = get_rf_results(df_exp, 'Exp6')

        auc1 = r1.get('auc_mean', float('nan')) if r1 is not None else float('nan')
        auc2 = r2.get('auc_mean', float('nan')) if r2 is not None else float('nan')
        auc3 = r3.get('auc_mean', float('nan')) if r3 is not None else float('nan')
        auc6 = r6.get('auc_mean', float('nan')) if r6 is not None else float('nan')

        bias = auc2 - auc3 if not (pd.isna(auc2) or pd.isna(auc3)) else float('nan')
        aug  = auc6 - auc3 if not (pd.isna(auc6) or pd.isna(auc3)) else float('nan')

        # FIX C: NaN-safe finding strings
        if not pd.isna(auc1):
            finding1_text = (
                f"A classifier using only scanner identity (no image features) achieved "
                f"AUC = **{auc1:.3f}**. This is the maximum bias possible in this dataset — "
                f"pure scanner recognition without seeing a single pixel."
            )
        else:
            finding1_text = "Experiment 1 (scanner baseline) results not available."

        if not (pd.isna(auc2) or pd.isna(auc3) or pd.isna(bias)):
            finding2_text = (
                f"Removing the scanner confound reduced AUC from **{auc2:.3f}** "
                f"(CONFOUNDED, Exp 2) to **{auc3:.3f}** (REDUCED CONFOUND, Exp 3) — "
                f"a drop of **{bias:.3f}**. This is exactly how much scanner identity "
                f"inflated apparent performance."
            )
        else:
            finding2_text = "Experiment 2 or 3 results not available."

        if not pd.isna(aug):
            finding3_text = (
                f"Synthetic BRIVO-Haemorrhage augmentation changed AUC by **{aug:+.3f}** "
                f"(Exp 6 vs Exp 3). "
                + ("The correction was meaningful." if aug > 0.02
                   else "The effect was within the noise floor, consistent with the small sample size.")
            )
        else:
            finding3_text = "Synthetic augmentation result not available (Exp 6 missing)."

        findings = [
            ("🔴 Finding 1 — Scanner Fingerprints Confirmed", finding1_text),
            ("🟡 Finding 2 — Scanner Bias Quantified", finding2_text),
            ("🔵 Finding 3 — Augmentation Effect", finding3_text),
        ]

        for title, body in findings:
            st.markdown(f"""
            <div class="finding-box">
                <strong>{title}</strong><br>
                {body}
            </div>
            """, unsafe_allow_html=True)

    # ── Concluding Statement ───────────────────────────────────────
    # FIX B: try/except guard for NameError
    try:
        _auc2 = auc2
        _auc3 = auc3
        _bias = bias
    except NameError:
        _auc2 = float('nan')
        _auc3 = float('nan')
        _bias = float('nan')

    if not pd.isna(_auc2) and not pd.isna(_auc3) and not pd.isna(_bias):
        bias_pct = (_bias / _auc2 * 100) if _auc2 > 0 else float('nan')
        st.markdown(f"""
        <div class="conclusion-box">
            <h3>Core Finding</h3>
            <p style="font-size:1.05rem;margin:0;">
                Scanner identity explained approximately <strong>
                {bias_pct:.1f}%</strong> of this model's apparent performance inflation.
                The honest AUC on scanner-homogeneous data was <strong>{_auc3:.3f}</strong>.
                This demonstrates why <strong>scanner diversity validation is essential</strong>
                before trusting any medical AI classifier.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    inject_css()
    page = render_top_nav()

    if page == "trap":
        page_scanner_trap()
    elif page == "detect":
        page_detect()
    elif page == "lab":
        page_bias_lab()


if __name__ == '__main__':
    main()


