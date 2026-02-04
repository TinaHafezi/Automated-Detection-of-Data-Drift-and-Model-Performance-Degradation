import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from metrics_store import MetricsStore
from config_loader import Config

def load_metrics():
    conn = sqlite3.connect("metrics.db")
    df = pd.read_sql("SELECT * FROM metrics", conn)
    conn.close()
    return df

def risk_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "System Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'thickness': 0.4},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def system_status(score):
    if score < 30:
        st.success("🟢 SYSTEM HEALTHY")
    elif score < 60:
        st.warning("🟡 SYSTEM AT RISK")
    else:
        st.error("🔴 CRITICAL SYSTEM RISK")

def risk_trend(df):
    risk_df = df[df["metric_name"] == "system_risk_score"]
    risk_df["timestamp"] = pd.to_datetime(risk_df["timestamp"])

    st.line_chart(
        risk_df.set_index("timestamp")["value"]
    )

config = Config()

st.set_page_config(layout="wide")
st.title("🧠 ML Model Monitoring Dashboard")

store = MetricsStore()
df = store.load_history()

if df.empty:
    st.warning("No monitoring data found yet.")
    st.stop()


# Preprocess
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Split metrics
drift_df = df[df["metric_name"].str.contains("PSI_")]
perf_df = df[df["metric_name"].str.contains("current_")]
drop_df = df[df["metric_name"].str.contains("drop_")]
drift_df["feature"] = drift_df["metric_name"].str.replace("PSI_", "")

# Latest snapshot
latest = df.sort_values("timestamp").groupby("metric_name").tail(1)


# DRIFT SECTION
st.header("🧠 Concept / Prediction Drift")

pred_df = df[df["metric_name"].str.contains("PRED_")]

if not pred_df.empty:
    fig_pred = px.line(
        pred_df,
        x="timestamp",
        y="value",
        color="metric_name",
        title="Prediction Drift Metrics"
    )
    st.plotly_chart(fig_pred, use_container_width=True)


st.header("📊 Data Drift Monitoring")

col1, col2 = st.columns(2)

with col1:
    fig_drift = px.line(
        drift_df,
        x="timestamp",
        y="value",
        color="feature",
        title="PSI per Feature Over Time"
    )
    st.plotly_chart(fig_drift, use_container_width=True)

with col2:
    latest_drift = latest[latest["metric_name"].str.contains("PSI_")]
    drifted = (latest_drift["value"] > 0.2).sum()

    st.metric("Drifted Features (Latest Run)", drifted)

    if drifted > 0:
        st.error("⚠️ Data Drift Detected")
    else:
        st.success("✅ Data Stable")


# PERFORMANCE SECTION
st.header("📈 Model Performance Monitoring")

col3, col4 = st.columns(2)

with col3:
    fig_perf = px.line(
        perf_df,
        x="timestamp",
        y="value",
        color="metric_name",
        title="Performance Metrics Over Time"
    )
    st.plotly_chart(fig_perf, use_container_width=True)

with col4:
    latest_perf = latest[latest["metric_name"].str.contains("current_")]
    latest_drop = latest[latest["metric_name"].str.contains("drop_")]

    acc = latest_perf[latest_perf["metric_name"] == "current_accuracy"]["value"]
    drop_acc = latest_drop[latest_drop["metric_name"] == "drop_accuracy"]["value"]

    if not acc.empty:
        st.metric("Current Accuracy", f"{acc.values[0]:.3f}")

    PERF_DROP_THRESHOLD = config.get("monitoring", "performance_drop_threshold")
    
    if not drop_acc.empty:
        if drop_acc.values[0] > PERF_DROP_THRESHOLD:
            st.error(f"⚠️ Accuracy Drop: {drop_acc.values[0]:.3f}")
        else:
            st.success("Performance Stable")


# 📋 RAW DATA
st.header("📋 Raw Metrics Log")
st.dataframe(df.sort_values("timestamp", ascending=False))
