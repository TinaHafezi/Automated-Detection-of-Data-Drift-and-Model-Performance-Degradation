import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from metrics_store import MetricsStore

def load_metrics():
    """Load metrics from SQLite database"""
    import sqlite3
    conn = sqlite3.connect("metrics.db")
    df = pd.read_sql("SELECT * FROM metrics", conn)
    conn.close()
    return df

def risk_gauge(score):
    """Create risk gauge visualization"""
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
    return fig

def display_drift_metrics(df):
    """Display all drift metrics in a compact view"""
    st.subheader("📊 Statistical Drift Metrics")
    
    latest = df.sort_values("timestamp").groupby("metric_name").tail(1)
    
    # Create two columns for KS and Wasserstein
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Kolmogorov-Smirnov Test**")
        ks_metrics = latest[latest["metric_name"].str.startswith("KS_")]
        if not ks_metrics.empty:
            ks_summary = []
            for _, row in ks_metrics.iterrows():
                feature = row["metric_name"].replace("ks_", "")
                value = row["value"]
                try:
                    num_val = float(value)
                    ks_summary.append({
                        "Feature": feature,
                        "KS": f"{num_val:.2e}" if num_val < 0.001 else f"{num_val:.3f}",
                        "Status": "⚠️" if num_val < 0.01 else "✅"
                    })
                except:
                    ks_summary.append({
                        "Feature": feature,
                        "KS": str(value),
                        "Status": "❓"
                    })
            
            ks_df = pd.DataFrame(ks_summary)
            st.dataframe(ks_df, height=200, use_container_width=True)
            
            # Count problematic KS values
            problematic = sum(1 for _, row in ks_metrics.iterrows() 
                            if isinstance(row["value"], (int, float, np.number)) and row["value"] < 0.01)
            if problematic > 0:
                st.error(f"{problematic} features with significantly different distributions")
    
    with col2:
        st.write("**Wasserstein Distance**")
        wasserstein_metrics = latest[latest["metric_name"].str.startswith("WASS_")]
        if not wasserstein_metrics.empty:
            wasserstein_summary = []
            high_drift_count = 0
            
            for _, row in wasserstein_metrics.iterrows():
                feature = row["metric_name"].replace("wasserstein_", "")
                value = row["value"]
                try:
                    num_val = float(value)
                    wasserstein_summary.append({
                        "Feature": feature[:20],  # Truncate long names
                        "Distance": f"{num_val:.1f}",
                        "Status": "🔴" if num_val > 10 else "🟡" if num_val > 1 else "🟢"
                    })
                    if num_val > 10:
                        high_drift_count += 1
                except:
                    pass
            
            if wasserstein_summary:
                wasserstein_df = pd.DataFrame(wasserstein_summary)
                st.dataframe(wasserstein_df, height=200, use_container_width=True)
                
                if high_drift_count > 0:
                    st.error(f"{high_drift_count} features with extreme Wasserstein drift (>10)")
    
    # Data Quality Issues
    st.subheader("🔍 Data Quality Issues")
    dq_col1, dq_col2 = st.columns(2)
    
    with dq_col1:
        st.write("**Missing Values**")
        missing_metrics = latest[latest["metric_name"].str.contains("dq_missing_")]
        if not missing_metrics.empty:
            for _, row in missing_metrics.iterrows():
                feature = row["metric_name"].replace("dq_missing_", "")
                value = row["value"]
                try:
                    if float(value) > 0:
                        st.warning(f"{feature}: {float(value):.1%} missing")
                except:
                    pass
    
    with dq_col2:
        st.write("**Zero Values**")
        zero_metrics = latest[latest["metric_name"].str.contains("dq_zero_")]
        if not zero_metrics.empty:
            for _, row in zero_metrics.iterrows():
                feature = row["metric_name"].replace("dq_zero_", "")
                value = row["value"]
                try:
                    if float(value) > 0.1:  # More than 10% zeros
                        st.warning(f"{feature}: {float(value):.1%} zeros")
                except:
                    pass
    
    # Prediction Drift
    st.subheader("📈 Prediction Drift")
    pred_metrics = latest[latest["metric_name"].str.startswith("PRED_")]
    if not pred_metrics.empty:
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        with pred_col1:
            mean_shift = latest[latest["metric_name"] == "PRED_mean_shift"]["value"].values
            if len(mean_shift) > 0:
                try:
                    val = float(mean_shift[0])
                    st.metric("Mean Shift", f"{val:.2f}")
                    if abs(val) > 10:
                        st.error("Extreme bias!")
                except:
                    st.metric("Mean Shift", "N/A")
        
        with pred_col2:
            dist_shift = latest[latest["metric_name"] == "PRED_distribution_shift"]["value"].values
            if len(dist_shift) > 0:
                try:
                    val = float(dist_shift[0])
                    st.metric("Distribution Shift", f"{val:.2f}")
                    if val > 50:
                        st.error("Major shift!")
                except:
                    st.metric("Distribution Shift", "N/A")
        
        with pred_col3:
            conf_shift = latest[latest["metric_name"] == "PRED_confidence_shift"]["value"].values
            if len(conf_shift) > 0:
                try:
                    val = float(conf_shift[0])
                    st.metric("Confidence Shift", f"{val:.2f}")
                except:
                    st.metric("Confidence Shift", "N/A")

def display_performance_metrics(df):
    """Display model performance metrics"""
    st.subheader("🎯 Model Performance")
    
    latest = df.sort_values("timestamp").groupby("metric_name").tail(1)
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        # Current metrics
        current_metrics = latest[latest["metric_name"].str.startswith("current_")]
        for _, row in current_metrics.iterrows():
            metric = row["metric_name"].replace("current_", "")
            value = row["value"]
            try:
                st.metric(f"Current {metric}", f"{float(value):.3f}")
            except:
                st.metric(f"Current {metric}", str(value))
    
    with perf_col2:
        # Drop metrics
        drop_metrics = latest[latest["metric_name"].str.startswith("drop_")]
        for _, row in drop_metrics.iterrows():
            metric = row["metric_name"].replace("drop_", "")
            value = row["value"]
            try:
                val = float(value)
                st.metric(f"{metric} Drop", f"{val:.3f}")
                if val > 0.05:  # 5% drop threshold
                    st.error(f"Significant {metric} drop!")
            except:
                st.metric(f"{metric} Drop", str(value))

def display_alerts(df):
    """Generate and display system alerts"""
    st.subheader("🚨 System Alerts")
    
    latest = df.sort_values("timestamp").groupby("metric_name").tail(1)
    alerts = []
    
    # Check for extreme Wasserstein drift
    wasserstein_metrics = latest[latest["metric_name"].str.startswith("wasserstein_")]
    extreme_wasserstein = 0
    for _, row in wasserstein_metrics.iterrows():
        try:
            if float(row["value"]) > 100:
                extreme_wasserstein += 1
        except:
            pass
    
    if extreme_wasserstein > 0:
        alerts.append(f"🔴 {extreme_wasserstein} features with extreme Wasserstein drift (>100)")
    
    # Check KS test results
    ks_metrics = latest[latest["metric_name"].str.startswith("ks_")]
    problematic_ks = 0
    for _, row in ks_metrics.iterrows():
        try:
            val = float(row["value"])
            if val < 0.001:  # KS near 0 = completely different
                problematic_ks += 1
        except:
            pass
    
    if problematic_ks > 0:
        alerts.append(f"🟡 {problematic_ks} features with completely different distributions (KS ≈ 0)")
    
    # Check prediction drift
    pred_mean = latest[latest["metric_name"] == "PRED_mean_shift"]["value"].values
    if len(pred_mean) > 0:
        try:
            if abs(float(pred_mean[0])) > 50:
                alerts.append(f"🔴 Extreme prediction bias: {float(pred_mean[0]):.1f}")
            elif abs(float(pred_mean[0])) > 10:
                alerts.append(f"🟡 Significant prediction bias: {float(pred_mean[0]):.1f}")
        except:
            pass
    
    # Check data quality
    missing_metrics = latest[latest["metric_name"].str.contains("dq_missing_")]
    high_missing = 0
    for _, row in missing_metrics.iterrows():
        try:
            if float(row["value"]) > 0.2:  # More than 20% missing
                high_missing += 1
        except:
            pass
    
    if high_missing > 0:
        alerts.append(f"🟡 {high_missing} features with >20% missing values")
    
    # Display alerts
    if alerts:
        for alert in alerts:
            if "🔴" in alert:
                st.error(alert)
            elif "🟡" in alert:
                st.warning(alert)
            else:
                st.info(alert)
    else:
        st.success("✅ No critical alerts - System is stable")

# ========== MAIN DASHBOARD ==========
st.set_page_config(layout="wide")
st.title("🧠 ML Model Monitoring Dashboard")

# Sidebar
with st.sidebar:
    st.header("Settings")
    data_source = st.radio("Data Source", ["Metrics Store", "SQLite Database"])
    
    # Quick filters
    st.subheader("Filters")
    show_only_problems = st.checkbox("Show only problem metrics", value=False)
    
    # Info
    st.subheader("Info")
    st.info("""
    **Thresholds:**
    - KS < 0.01: Different distribution
    - Wasserstein > 10: Significant drift
    - Prediction shift > 10: Major bias
    """)

# Load data
if data_source == "Metrics Store":
    store = MetricsStore()
    df = store.load_history()
else:
    df = load_metrics()

if df.empty:
    st.warning("No monitoring data found yet.")
    st.stop()

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])
latest_timestamp = df["timestamp"].max()
st.caption(f"Last update: {latest_timestamp}")

# ========== RISK OVERVIEW ==========
st.header("⚠️ System Risk Overview")

# Get the ACTUAL risk score from your data (not calculate it)
latest = df.sort_values("timestamp").groupby("metric_name").tail(1)

# Look for system_risk_score in your data
risk_score_series = latest[latest["metric_name"] == "system_risk_score"]["value"]

if not risk_score_series.empty:
    try:
        # Use the risk score from your pipeline
        risk_score = float(risk_score_series.values[0])
    except:
        risk_score = 0
else:
    risk_score = 0

# Display risk gauge and status
col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(risk_gauge(risk_score), use_container_width=True)

with col2:
    # Risk status
    if risk_score < 30:
        st.success("🟢 SYSTEM HEALTHY")
    elif risk_score < 60:
        st.warning("🟡 SYSTEM AT RISK")
    else:
        st.error("🔴 CRITICAL SYSTEM RISK")
    
    st.metric("Risk Score", f"{risk_score:.1f}/100")
    
    # Get drift counts from your data
    static_drift = latest[latest["metric_name"] == "static_drifted_features"]["value"].values
    dynamic_drift = latest[latest["metric_name"] == "dynamic_drifted_features"]["value"].values
    
    st.subheader("Drift Summary")
    if len(static_drift) > 0:
        st.write(f"📊 Static drift: {int(static_drift[0])} features")
    if len(dynamic_drift) > 0:
        st.write(f"📈 Dynamic drift: {int(dynamic_drift[0])} features")
    
    # Time info
    st.write(f"⏰ Last run: {latest_timestamp:%H:%M:%S}")

# ========== ALERTS SECTION ==========
display_alerts(df)

# ========== DRIFT METRICS ==========
display_drift_metrics(df)

# ========== PERFORMANCE METRICS ==========
display_performance_metrics(df)

# ========== TREND VISUALIZATIONS ==========
st.subheader("📈 Trends Over Time")

# Filter for risk score trend
risk_df = df[df["metric_name"] == "system_risk_score"].copy()
if not risk_df.empty:
    risk_df["numeric_value"] = risk_df["value"].apply(lambda x: float(x) if pd.notnull(x) else np.nan)
    
    fig = px.line(
        risk_df.dropna(subset=["numeric_value"]),
        x="timestamp",
        y="numeric_value",
        title="Risk Score Trend Over Time",
        markers=True
    )
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

# Add other metrics visualization
st.subheader("📊 Other Metric Trends")

# Select metrics to visualize (excluding system_risk_score since we already showed it)
metric_options = sorted([m for m in df["metric_name"].unique() if m != "system_risk_score"])
selected_metrics = st.multiselect(
    "Select additional metrics to plot",
    options=metric_options,
    default=["PRED_mean_shift", "dynamic_perf_threshold"],
    max_selections=4
)

if selected_metrics:
    trend_df = df[df["metric_name"].isin(selected_metrics)].copy()
    
    # Convert values to numeric where possible
    def safe_convert(x):
        try:
            return float(x)
        except:
            return np.nan
    
    trend_df["numeric_value"] = trend_df["value"].apply(safe_convert)
    
    # Plot
    fig = px.line(
        trend_df.dropna(subset=["numeric_value"]),
        x="timestamp",
        y="numeric_value",
        color="metric_name",
        title="Selected Metrics Over Time",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

# ========== RECENT DATA ==========
with st.expander("📋 Recent Metrics Data"):
    # Show latest values
    latest_df = df.sort_values("timestamp").groupby("metric_name").tail(1)
    latest_df = latest_df.sort_values("metric_name")
    
    # Filter to show only problematic metrics if selected
    if show_only_problems:
        # Identify problematic metrics
        problem_metrics = []
        for _, row in latest_df.iterrows():
            metric = row["metric_name"]
            value = row["value"]
            
            try:
                num_val = float(value)
                if metric.startswith("ks_") and num_val < 0.01:
                    problem_metrics.append(metric)
                elif metric.startswith("wasserstein_") and num_val > 10:
                    problem_metrics.append(metric)
                elif metric.startswith("PRED_") and abs(num_val) > 10:
                    problem_metrics.append(metric)
                elif "missing" in metric and num_val > 0.2:
                    problem_metrics.append(metric)
            except:
                pass
        
        filtered_df = latest_df[latest_df["metric_name"].isin(problem_metrics)]
    else:
        filtered_df = latest_df
    
    st.dataframe(
        filtered_df[["metric_name", "value", "timestamp"]],
        height=300,
        use_container_width=True
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="latest_metrics.csv",
        mime="text/csv",
    )

# ========== FOOTER ==========
st.markdown("---")
st.caption("ML Model Monitoring Dashboard • Auto-refresh: Reload page for latest data")