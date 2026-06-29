"""
SG-CancerSight — Phase 4: Interactive Dashboard
================================================
Author  : Dr. Lakshmi C. | PhD Mathematics | Healthcare Operations Research
Purpose : Interactive Streamlit dashboard for NCCS Data Analyst portfolio
JD Map  : Communication of analytical outputs (Section 3 of NCCS Job Description)

Run locally:
    cd sg-cancersight/dashboard
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Survival analysis ─────────────────────────────────────────────────────
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SG-CancerSight | NCCS Portfolio",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C_STAGE  = {"Stage I": "#1D9E75", "Stage II": "#378ADD", "Stage III": "#D85A30"}
C_ER     = {"Positive": "#1D9E75", "Negative": "#D85A30"}
C_AGE    = {
    "<40":   "#534AB7",
    "40-49": "#378ADD",
    "40–49": "#378ADD",
    "50-59": "#1D9E75",
    "50–59": "#1D9E75",
    "60-69": "#EF9F27",
    "60–69": "#EF9F27",
    "70+":   "#D85A30",
}
GRAY     = "#888780"
BG_CARD  = "#F8F9FA"

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = Path(__file__).resolve().parent.parent
    path = base / "data" / "processed" / "seer_clean.csv"
    df = pd.read_csv(path)
    df["Age Group"] = df["Age Group"].astype(str).replace({
        "40\u201349": "40-49",
        "50\u201359": "50-59",
        "60\u201369": "60-69",
    })
    # Assign costs
    COSTS = {
        "surgery":   {"Stage I": 8500,  "Stage II": 14000, "Stage III": 22000},
        "chemo":     {"Stage I": 0,     "Stage II": 18000, "Stage III": 38000},
        "radiation": {"Stage I": 6000,  "Stage II": 9000,  "Stage III": 14000},
        "followup":  {"Stage I": 1200,  "Stage II": 2000,  "Stage III": 3500},
    }
    def total_cost(row):
        s = row["Stage Group"]
        er = row["Estrogen Status"]
        yrs = min(row["Survival Months"] / 12, 10)
        surgery   = COSTS["surgery"].get(s, 0)
        chemo     = COSTS["chemo"].get(s, 0)
        radiation = COSTS["radiation"].get(s, 0)
        hormone   = 2400 * min(yrs, 5) if er == "Positive" else 0
        followup  = COSTS["followup"].get(s, 0) * yrs
        eol       = 12000 if row["Status_Code"] == 1 else 0
        return surgery + chemo + radiation + hormone + followup + eol

    df["cost_total"] = df.apply(total_cost, axis=1)
    return df

df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0F4C81 0%, #185FA5 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: #B5D4F4; margin: 0.3rem 0 0; font-size: 0.95rem; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #185FA5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 0.5rem;
    }
    .metric-card .label { font-size: 0.8rem; color: #888780; font-weight: 500; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #0F4C81; }
    .metric-card .delta { font-size: 0.8rem; color: #1D9E75; }

    /* Section header */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0F4C81;
        border-bottom: 2px solid #B5D4F4;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #F1EFE8;
        border-radius: 6px 6px 0 0;
        padding: 0.4rem 1rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #185FA5 !important;
        color: white !important;
    }

    /* Sidebar */
    .css-1d391kg { background: #F8F9FA; }

    /* Hide streamlit default header */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Filters")
    st.markdown("---")

    st.markdown("**Cancer Stage**")
    stages_all = ["Stage I", "Stage II", "Stage III"]
    sel_stages = st.multiselect(
        "Select stage(s)",
        options=stages_all,
        default=stages_all,
        label_visibility="collapsed",
    )

    st.markdown("**Estrogen Receptor Status**")
    sel_er = st.multiselect(
        "Select ER status",
        options=["Positive", "Negative"],
        default=["Positive", "Negative"],
        label_visibility="collapsed",
    )

    st.markdown("**Age Group**")
    age_groups_all = ["<40", "40-49", "50-59", "60-69", "70+"]
    sel_age = st.multiselect(
        "Select age group(s)",
        options=age_groups_all,
        default=age_groups_all,
        label_visibility="collapsed",
    )

    st.markdown("**Survival months range**")
    surv_min, surv_max = st.slider(
        "Survival months",
        min_value=int(df["Survival Months"].min()),
        max_value=int(df["Survival Months"].max()),
        value=(int(df["Survival Months"].min()), int(df["Survival Months"].max())),
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Project Info**")
    st.info(
        "📊 **SG-CancerSight**\n\n"
        "Portfolio project for NCCS\n"
        "Data Analyst (DDOIT) role.\n\n"
        "Dataset: SEER Breast Cancer\n"
        "Registry (n = 4,015)\n\n"
        "**Author:** Dr. Lakshmi C.\n"
        "PhD Mathematics | NTU Postdoc"
    )

# ─────────────────────────────────────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────────────────────────────────────
if not sel_stages:
    sel_stages = stages_all
if not sel_er:
    sel_er = ["Positive", "Negative"]
if not sel_age:
    sel_age = age_groups_all

dff = df[
    df["Stage Group"].isin(sel_stages) &
    df["Estrogen Status"].isin(sel_er) &
    df["Age Group"].isin(sel_age) &
    df["Survival Months"].between(surv_min, surv_max)
].copy()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 SG-CancerSight — Breast Cancer Analytics Dashboard</h1>
    <p>Phase 4 | NCCS Data Analyst Portfolio | SEER Breast Cancer Registry</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI METRICS ROW
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">PATIENTS (filtered)</div>
        <div class="value">{len(dff):,}</div>
        <div class="delta">of {len(df):,} total</div>
    </div>""", unsafe_allow_html=True)

with k2:
    mort = dff["Status_Code"].mean() * 100
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #D85A30;">
        <div class="label">MORTALITY RATE</div>
        <div class="value">{mort:.1f}%</div>
        <div class="delta">{dff["Status_Code"].sum():,} events</div>
    </div>""", unsafe_allow_html=True)

with k3:
    med_surv = dff["Survival Months"].median()
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #1D9E75;">
        <div class="label">MEDIAN SURVIVAL</div>
        <div class="value">{med_surv:.0f} mo</div>
        <div class="delta">{med_surv/12:.1f} years</div>
    </div>""", unsafe_allow_html=True)

with k4:
    med_age = dff["Age"].median()
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #EF9F27;">
        <div class="label">MEDIAN AGE</div>
        <div class="value">{med_age:.0f} yrs</div>
        <div class="delta">at diagnosis</div>
    </div>""", unsafe_allow_html=True)

with k5:
    mean_cost = dff["cost_total"].mean()
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #534AB7;">
        <div class="label">MEAN COST / PATIENT</div>
        <div class="value">S${mean_cost:,.0f}</div>
        <div class="delta">proxy estimate</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Patient Overview",
    "📈 Survival Analysis",
    "💰 Health Economics",
    "🔬 Cox Model",
    "📋 Policy Insights",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PATIENT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Patient Demographics & Clinical Characteristics</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Stage distribution donut
        stage_cnt = dff["Stage Group"].value_counts()
        fig = px.pie(
            values=stage_cnt.values,
            names=stage_cnt.index,
            hole=0.45,
            color=stage_cnt.index,
            color_discrete_map=C_STAGE,
            title="Stage Distribution",
        )
        fig.update_traces(textinfo="percent+label", textfont_size=12)
        fig.update_layout(
            showlegend=True,
            height=340,
            margin=dict(t=50, b=10, l=10, r=10),
            font=dict(size=11),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Age group bar
        ag_cnt = dff["Age Group"].value_counts().reindex(
            ["<40","40-49","50-59","60-69","70+"]
        ).fillna(0)
        fig = px.bar(
            x=ag_cnt.index,
            y=ag_cnt.values,
            color=ag_cnt.index,
            color_discrete_map=C_AGE,
            title="Age Group Distribution",
            labels={"x": "Age group", "y": "Count"},
        )
        fig.update_traces(texttemplate="%{y:,}", textposition="outside")
        fig.update_layout(
            showlegend=False,
            height=340,
            margin=dict(t=50, b=30, l=10, r=10),
            font=dict(size=11),
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # ER status by stage grouped bar
        er_stage = dff.groupby(["Stage Group","Estrogen Status"]).size().reset_index(name="n")
        fig = px.bar(
            er_stage,
            x="Stage Group", y="n",
            color="Estrogen Status",
            barmode="group",
            color_discrete_map=C_ER,
            title="ER Status by Stage",
            labels={"n": "Patients"},
        )
        fig.update_layout(
            height=320,
            margin=dict(t=50, b=30, l=10, r=10),
            font=dict(size=11),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Tumour size violin
        fig = px.violin(
            dff, x="Stage Group", y="Tumor Size",
            color="Stage Group",
            color_discrete_map=C_STAGE,
            box=True, points=False,
            title="Tumour Size Distribution by Stage (mm)",
            labels={"Tumor Size": "Size (mm)"},
        )
        fig.update_layout(
            showlegend=False,
            height=320,
            margin=dict(t=50, b=30, l=10, r=10),
            font=dict(size=11),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.markdown('<div class="section-header">Descriptive Statistics by Stage</div>',
                unsafe_allow_html=True)
    tbl = dff.groupby("Stage Group").agg(
        Patients=("Status_Code", "count"),
        Deaths=("Status_Code", "sum"),
        Mortality_pct=("Status_Code", lambda x: round(x.mean()*100, 1)),
        Median_Age=("Age", "median"),
        Median_TumourSize_mm=("Tumor Size", "median"),
        Median_Survival_mo=("Survival Months", "median"),
        Mean_Cost_SGD=("cost_total", lambda x: round(x.mean(), 0)),
    ).reindex(["Stage I","Stage II","Stage III"])
    tbl.columns = ["Patients","Deaths","Mortality (%)","Median Age",
                   "Median Tumour Size (mm)","Median Survival (months)","Mean Cost (SGD)"]
    st.dataframe(tbl, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SURVIVAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Kaplan-Meier Survival Analysis</div>',
                unsafe_allow_html=True)

    surv_col1, surv_col2 = st.columns([2, 1])

    with surv_col1:
        km_group = st.selectbox(
            "Group survival curves by:",
            ["Stage Group", "Estrogen Status", "Age Group", "Receptor Status"],
        )

    with surv_col2:
        show_ci = st.checkbox("Show 95% confidence intervals", value=True)

    # Build KM curves using lifelines → convert to plotly
    fig = go.Figure()

    if km_group == "Stage Group":
        groups = [g for g in ["Stage I","Stage II","Stage III"] if g in dff[km_group].unique()]
        color_map = C_STAGE
    elif km_group == "Estrogen Status":
        groups = ["Positive","Negative"]
        color_map = C_ER
    elif km_group == "Age Group":
        groups = [g for g in ["<40","40-49","50-59","60-69","70+"] if g in dff[km_group].astype(str).unique()]
        color_map = C_AGE
    else:
        groups = dff[km_group].unique().tolist()
        color_map = {"Hormone Receptor+": "#1D9E75", "Triple Negative / HR-": "#D85A30"}

    km_results = {}
    for grp in groups:
        mask = dff[km_group].astype(str) == str(grp)
        if mask.sum() < 5:
            continue
        kmf = KaplanMeierFitter(label=str(grp))
        kmf.fit(
            dff.loc[mask, "Survival Months"],
            dff.loc[mask, "Status_Code"],
        )
        km_results[grp] = kmf
        color = color_map.get(grp, GRAY)
        t   = kmf.timeline
        sf  = kmf.survival_function_[kmf._label].values
        cil = kmf.confidence_interval_[f"{kmf._label}_lower_0.95"].values
        ciu = kmf.confidence_interval_[f"{kmf._label}_upper_0.95"].values

        fig.add_trace(go.Scatter(
            x=t, y=sf,
            mode="lines", name=str(grp),
            line=dict(color=color, width=2.5),
        ))
        if show_ci:
            fig.add_trace(go.Scatter(
                x=np.concatenate([t, t[::-1]]),
                y=np.concatenate([ciu, cil[::-1]]),
                fill="toself",
                fillcolor=color,
                opacity=0.12,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.add_hline(y=0.5, line_dash="dot", line_color=GRAY, line_width=1, opacity=0.7)
    fig.update_layout(
        title=f"Kaplan-Meier Survival Curves by {km_group}",
        xaxis_title="Time (months)",
        yaxis_title="Survival probability S(t)",
        yaxis=dict(range=[0, 1.05]),
        height=430,
        font=dict(size=11),
        legend=dict(orientation="v", x=0.75, y=0.95),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Log-rank test result
    if len(groups) >= 2 and len(km_results) >= 2:
        st.markdown('<div class="section-header">Log-rank Test Results</div>',
                    unsafe_allow_html=True)
        res = multivariate_logrank_test(
            dff["Survival Months"],
            dff[km_group].astype(str),
            dff["Status_Code"],
        )
        p   = res.p_value
        chi = res.test_statistic
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Test statistic (χ²)", f"{chi:.3f}")
        col_b.metric("p-value", f"{p:.2e}")
        col_c.metric("Significance", sig)
        if p < 0.05:
            st.success(f"✅ Statistically significant difference in survival between {km_group} groups (p = {p:.2e})")
        else:
            st.warning(f"⚠️ No significant difference detected (p = {p:.2e})")

    # Median survival table
    st.markdown('<div class="section-header">Median Survival by Group</div>',
                unsafe_allow_html=True)
    med_rows = []
    for grp, kmf in km_results.items():
        mask = dff[km_group].astype(str) == str(grp)
        med = kmf.median_survival_time_
        med_str = f"{med:.0f} months" if (med != np.inf and not np.isnan(med)) else "Not reached"
        med_rows.append({
            "Group": grp,
            "n": int(mask.sum()),
            "Events": int(dff.loc[mask,"Status_Code"].sum()),
            "Mortality (%)": round(dff.loc[mask,"Status_Code"].mean()*100, 1),
            "Median Survival": med_str,
        })
    st.dataframe(pd.DataFrame(med_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HEALTH ECONOMICS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Healthcare Cost Analysis</div>',
                unsafe_allow_html=True)

    econ_col1, econ_col2 = st.columns(2)

    with econ_col1:
        # Cost by stage bar
        cost_stage = dff.groupby("Stage Group")["cost_total"].mean().reindex(
            ["Stage I","Stage II","Stage III"]
        )
        fig = px.bar(
            x=cost_stage.index,
            y=cost_stage.values,
            color=cost_stage.index,
            color_discrete_map=C_STAGE,
            title="Mean Total Cost by Stage (SGD)",
            labels={"x": "Stage", "y": "Mean cost (SGD)"},
            text=cost_stage.values,
        )
        fig.update_traces(texttemplate="S$%{text:,.0f}", textposition="outside")
        fig.update_layout(
            showlegend=False,
            height=360,
            margin=dict(t=50, b=30),
            yaxis=dict(tickformat="$,.0f"),
            font=dict(size=11),
        )
        st.plotly_chart(fig, use_container_width=True)

    with econ_col2:
        # Cost distribution box plot
        fig = px.box(
            dff, x="Stage Group", y="cost_total",
            color="Stage Group",
            color_discrete_map=C_STAGE,
            title="Cost Distribution by Stage (SGD)",
            labels={"cost_total": "Total cost (SGD)", "Stage Group": "Stage"},
            points=False,
        )
        fig.update_layout(
            showlegend=False,
            height=360,
            margin=dict(t=50, b=30),
            yaxis=dict(tickformat="$,.0f"),
            font=dict(size=11),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cost vs survival scatter
    st.markdown('<div class="section-header">Cost vs Survival Trade-off</div>',
                unsafe_allow_html=True)

    fig = px.scatter(
        dff.sample(min(1500, len(dff)), random_state=42),
        x="Survival Months",
        y="cost_total",
        color="Stage Group",
        color_discrete_map=C_STAGE,
        opacity=0.35,
        size_max=5,
        trendline="ols",
        title="Cost vs Survival by Stage",
        labels={
            "cost_total": "Total cost (SGD)",
            "Survival Months": "Survival (months)",
        },
        height=380,
    )
    fig.update_layout(
        font=dict(size=11),
        yaxis=dict(tickformat="$,.0f"),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Population projection slider
    st.markdown('<div class="section-header">Population-Level Cost Projection</div>',
                unsafe_allow_html=True)

    proj_col1, proj_col2 = st.columns([1, 2])
    with proj_col1:
        annual_cases = st.slider("Annual new cases (Singapore)", 1500, 3000, 2200, 50)
        shift_pct = st.slider("% stage shift to earlier diagnosis", 0, 30, 10, 1)

    with proj_col2:
        stage_dist = df["Stage Group"].value_counts(normalize=True)
        mean_cost  = df.groupby("Stage Group")["cost_total"].mean()
        base_cost  = sum(annual_cases * stage_dist.get(s,0) * mean_cost.get(s,0) for s in ["Stage I","Stage II","Stage III"])

        # Apply shift
        new_dist = stage_dist.to_dict()
        shift_amount = (new_dist.get("Stage II",0) + new_dist.get("Stage III",0)) * (shift_pct/100)
        new_dist["Stage I"]   = new_dist.get("Stage I",0)   + shift_amount
        new_dist["Stage II"]  = new_dist.get("Stage II",0)  - shift_amount * 0.5
        new_dist["Stage III"] = new_dist.get("Stage III",0) - shift_amount * 0.5
        new_cost = sum(annual_cases * new_dist.get(s,0) * mean_cost.get(s,0) for s in ["Stage I","Stage II","Stage III"])
        savings  = base_cost - new_cost

        years = list(range(1, 11))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, y=[base_cost*y/1e6 for y in years],
            mode="lines+markers", name="Current trajectory",
            line=dict(color="#D85A30", width=2.5),
            marker=dict(size=7),
        ))
        fig.add_trace(go.Scatter(
            x=years, y=[new_cost*y/1e6 for y in years],
            mode="lines+markers", name=f"With {shift_pct}% stage shift",
            line=dict(color="#1D9E75", width=2.5),
            marker=dict(size=7),
        ))
        fig.add_trace(go.Scatter(
            x=years+years[::-1],
            y=[base_cost*y/1e6 for y in years]+[new_cost*y/1e6 for y in years[::-1]],
            fill="toself", fillcolor="rgba(29,158,117,0.12)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.update_layout(
            title=f"10-year cost projection | Annual savings: S${savings/1e6:.1f}M",
            xaxis_title="Year",
            yaxis_title="Cumulative cost (SGD millions)",
            height=320,
            font=dict(size=11),
            yaxis=dict(tickformat="$,.0fM"),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Current annual cost", f"S${base_cost/1e6:.1f}M")
    mc2.metric("With stage shift", f"S${new_cost/1e6:.1f}M")
    mc3.metric("Annual savings", f"S${savings/1e6:.1f}M", delta=f"-{savings/base_cost*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COX MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Cox Proportional Hazards Model</div>',
                unsafe_allow_html=True)
    st.info("The Cox model estimates the independent effect of each variable on mortality risk, "
            "adjusting for all other covariates simultaneously.")

    with st.spinner("Fitting Cox model on filtered data..."):
        cox_df = dff[["Survival Months","Status_Code","Age","Tumor Size",
                       "Regional Node Positive","Stage Group","Estrogen Status"]].dropna().copy()
        cox_df["Stage_II"]    = (cox_df["Stage Group"] == "Stage II").astype(int)
        cox_df["Stage_III"]   = (cox_df["Stage Group"] == "Stage III").astype(int)
        cox_df["ER_Negative"] = (cox_df["Estrogen Status"] == "Negative").astype(int)

        features = cox_df[["Survival Months","Status_Code","Age","Tumor Size",
                            "Regional Node Positive","Stage_II","Stage_III","ER_Negative"]]

        if len(features) >= 50 and features["Status_Code"].sum() >= 10:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(features, duration_col="Survival Months",
                    event_col="Status_Code", show_progress=False)

            summary = cph.summary.copy()
            summary["HR"]    = np.exp(summary["coef"])
            summary["HR_lo"] = np.exp(summary["coef lower 95%"])
            summary["HR_hi"] = np.exp(summary["coef upper 95%"])

            label_map = {
                "Age"                    : "Age (per 1 year)",
                "Tumor Size"             : "Tumour size (per 1 mm)",
                "Regional Node Positive" : "Positive nodes (per 1 node)",
                "Stage_II"               : "Stage II vs Stage I",
                "Stage_III"              : "Stage III vs Stage I",
                "ER_Negative"            : "ER Negative vs ER Positive",
            }
            summary["Label"] = [label_map.get(i, i) for i in summary.index]
            summary = summary.sort_values("HR", ascending=True)

            # Forest plot
            colors_fp = ["#D85A30" if hr > 1 else "#1D9E75" for hr in summary["HR"]]

            fig = go.Figure()
            for i, (_, row) in enumerate(summary.iterrows()):
                color = "#D85A30" if row["HR"] > 1 else "#1D9E75"
                fig.add_trace(go.Scatter(
                    x=[row["HR_lo"], row["HR_hi"]],
                    y=[row["Label"], row["Label"]],
                    mode="lines",
                    line=dict(color=color, width=3),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=[row["HR"]],
                    y=[row["Label"]],
                    mode="markers",
                    marker=dict(color=color, size=12,
                                line=dict(color="white", width=2)),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{row['Label']}</b><br>"
                        f"HR = {row['HR']:.3f}<br>"
                        f"95% CI: {row['HR_lo']:.3f} – {row['HR_hi']:.3f}<br>"
                        f"p = {row['p']:.4f}<extra></extra>"
                    ),
                ))

            fig.add_vline(x=1.0, line_dash="dash", line_color=GRAY, line_width=1.5)
            fig.update_layout(
                title=f"Hazard Ratio Forest Plot (C-index = {cph.concordance_index_:.3f})",
                xaxis_title="Hazard Ratio (HR) with 95% CI",
                height=420,
                font=dict(size=11),
                margin=dict(l=230),
            )
            st.plotly_chart(fig, use_container_width=True)

            # HR table
            st.markdown('<div class="section-header">Hazard Ratio Table</div>',
                        unsafe_allow_html=True)
            hr_table = summary[["Label","HR","HR_lo","HR_hi","p"]].copy()
            hr_table.columns = ["Variable","HR","95% CI Lower","95% CI Upper","p-value"]
            hr_table["Significance"] = hr_table["p-value"].apply(
                lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            )
            hr_table = hr_table.reset_index(drop=True)
            for col in ["HR","95% CI Lower","95% CI Upper"]:
                hr_table[col] = hr_table[col].round(3)
            hr_table["p-value"] = hr_table["p-value"].apply(lambda x: f"{x:.4f}")
            st.dataframe(hr_table, use_container_width=True, hide_index=True)

            # C-index metric
            c1, c2, c3 = st.columns(3)
            c1.metric("C-index (discrimination)", f"{cph.concordance_index_:.3f}")
            c2.metric("Patients in model", f"{len(features):,}")
            c3.metric("Events (deaths)", f"{features['Status_Code'].sum():,}")
        else:
            st.warning("Insufficient data for Cox model with current filters. Please widen your selection.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — POLICY INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Policy Brief — Key Findings</div>',
                unsafe_allow_html=True)

    # Dynamic findings based on filtered data
    mort_s1 = dff[dff["Stage Group"]=="Stage I"]["Status_Code"].mean()*100 if "Stage I" in dff["Stage Group"].values else 0
    mort_s3 = dff[dff["Stage Group"]=="Stage III"]["Status_Code"].mean()*100 if "Stage III" in dff["Stage Group"].values else 0
    cost_s1 = dff[dff["Stage Group"]=="Stage I"]["cost_total"].mean() if "Stage I" in dff["Stage Group"].values else 0
    cost_s3 = dff[dff["Stage Group"]=="Stage III"]["cost_total"].mean() if "Stage III" in dff["Stage Group"].values else 0

    p1, p2 = st.columns(2)

    with p1:
        st.markdown("""
        <div style="background:#E8F5E9; border-left:4px solid #1D9E75;
                    padding:1rem; border-radius:6px; margin-bottom:1rem;">
            <b style="color:#0F5132">FINDING 1 — Early detection saves lives</b><br><br>
            Stage I patients have significantly better survival outcomes
            than Stage III patients. Kaplan-Meier analysis confirms that
            cancer stage at diagnosis is the strongest predictor of survival.
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#E8F5E9; border-left:4px solid #1D9E75;
                    padding:1rem; border-radius:6px; margin-bottom:1rem;">
            <b style="color:#0F5132">FINDING 2 — ER status independently predicts survival</b><br><br>
            Oestrogen receptor positive patients have significantly better
            outcomes (log-rank p &lt; 0.001). Routine ER/PR testing at
            diagnosis enables targeted hormone therapy for eligible patients.
        </div>""", unsafe_allow_html=True)

    with p2:
        st.markdown(f"""
        <div style="background:#FFF3E0; border-left:4px solid #EF9F27;
                    padding:1rem; border-radius:6px; margin-bottom:1rem;">
            <b style="color:#7D4E00">FINDING 3 — Stage III costs {cost_s3/cost_s1:.1f}x more than Stage I</b><br><br>
            Mean cost per patient: <b>S${cost_s1:,.0f}</b> (Stage I) vs
            <b>S${cost_s3:,.0f}</b> (Stage III). The difference is driven
            by intensive chemotherapy, end-of-life costs, and longer
            high-intensity surveillance requirements.
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#FFF3E0; border-left:4px solid #EF9F27;
                    padding:1rem; border-radius:6px; margin-bottom:1rem;">
            <b style="color:#7D4E00">FINDING 4 — Results are robust</b><br><br>
            Cox multivariable model confirms stage and ER status are
            independent predictors of mortality after adjustment for
            age, tumour size, and lymph node involvement. Sensitivity
            analysis confirms cost findings hold under +/-40% cost variation.
        </div>""", unsafe_allow_html=True)

    # Recommendations
    st.markdown('<div class="section-header">Policy Recommendations</div>',
                unsafe_allow_html=True)

    rec1, rec2, rec3 = st.columns(3)

    with rec1:
        st.markdown("""
        <div style="background:white; border:1px solid #B5D4F4; border-radius:10px;
                    padding:1.2rem; text-align:center; height:180px;">
            <div style="font-size:2rem;">🔬</div>
            <b style="color:#185FA5">National Screening Programme</b><br><br>
            <span style="font-size:0.88rem; color:#52514E;">
            Invest in population breast cancer screening for women 40-69.
            Cost savings exceed programme costs within 3-5 years.
            </span>
        </div>""", unsafe_allow_html=True)

    with rec2:
        st.markdown("""
        <div style="background:white; border:1px solid #B5D4F4; border-radius:10px;
                    padding:1.2rem; text-align:center; height:180px;">
            <div style="font-size:2rem;">🧬</div>
            <b style="color:#185FA5">Routine ER/PR Testing</b><br><br>
            <span style="font-size:0.88rem; color:#52514E;">
            Mandate ER/PR testing at diagnosis for all breast cancer patients
            to enable hormone therapy for eligible patients and reduce
            long-term recurrence costs.
            </span>
        </div>""", unsafe_allow_html=True)

    with rec3:
        st.markdown("""
        <div style="background:white; border:1px solid #B5D4F4; border-radius:10px;
                    padding:1.2rem; text-align:center; height:180px;">
            <div style="font-size:2rem;">📊</div>
            <b style="color:#185FA5">Registry-Linked Cost Database</b><br><br>
            <span style="font-size:0.88rem; color:#52514E;">
            Establish a cancer registry-linked administrative cost database
            at NCCS for real-time health economic monitoring and policy
            evaluation.
            </span>
        </div>""", unsafe_allow_html=True)

    # Data download
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Download Filtered Data</div>',
                unsafe_allow_html=True)

    dl_col1, dl_col2 = st.columns([1, 3])
    with dl_col1:
        csv = dff.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download filtered dataset (CSV)",
            data=csv,
            file_name="sg_cancersight_filtered.csv",
            mime="text/csv",
        )
    with dl_col2:
        st.caption(f"Downloads {len(dff):,} rows with all {dff.shape[1]} variables based on current sidebar filters.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888780; font-size:0.85rem;'>"
    "SG-CancerSight | Phase 4 Dashboard | "
    "Dr. Lakshmi C. | PhD Mathematics | Healthcare Operations Research | "
    "NCCS Data Analyst Portfolio Project"
    "</div>",
    unsafe_allow_html=True,
)
