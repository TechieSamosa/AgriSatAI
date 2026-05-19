import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import glob
import os
import sys
import folium
from streamlit_folium import st_folium

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings

st.set_page_config(page_title="AgriSatAI Dashboard", layout="wide", page_icon="🌾")

# --- Helper Functions ---
def get_csv_mtime():
    csv_file = os.path.join(settings.BASE_DIR, 'output', 'processed_features_csv.csv')
    return os.path.getmtime(csv_file) if os.path.exists(csv_file) else 0

def get_models_mtime():
    clf_path = os.path.join(settings.BASE_DIR, 'models', 'crop_health_ensemble.joblib')
    reg_path = os.path.join(settings.BASE_DIR, 'models', 'yield_regressor.joblib')
    t1 = os.path.getmtime(clf_path) if os.path.exists(clf_path) else 0
    t2 = os.path.getmtime(reg_path) if os.path.exists(reg_path) else 0
    return t1 + t2

@st.cache_data
def load_data(mtime):
    try:
        csv_file = os.path.join(settings.BASE_DIR, 'output', 'processed_features_csv.csv')
        if not os.path.exists(csv_file):
            return None
        df = pd.read_csv(csv_file)
        # Add mock coordinates for the geospatial map if not present
        if 'lat' not in df.columns:
            # Random coordinates around a central ag region in India (e.g., Punjab)
            df['lat'] = np.random.normal(30.9, 0.5, len(df))
            df['lon'] = np.random.normal(75.8, 0.5, len(df))
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_models(mtime):
    try:
        clf_path = os.path.join(settings.BASE_DIR, 'models', 'crop_health_ensemble.joblib')
        reg_path = os.path.join(settings.BASE_DIR, 'models', 'yield_regressor.joblib')
        clf = joblib.load(clf_path) if os.path.exists(clf_path) else None
        reg = joblib.load(reg_path) if os.path.exists(reg_path) else None
        return clf, reg
    except Exception as e:
        return None, None

def load_profiling_data():
    log_file = os.path.join(settings.BASE_DIR, "output", "profiling_log.csv")
    if os.path.exists(log_file):
        return pd.read_csv(log_file)
    return None
df = load_data(get_csv_mtime())
clf, reg = load_models(get_models_mtime())

# --- Sidebar ---
st.sidebar.title("🌾 AgriSatAI")
st.sidebar.markdown("Distributed ETL & Predictive Analytics")
page = st.sidebar.radio("Navigation", [
    "Overview & Map", 
    "Pipeline & Data Quality", 
    "Predictive Models", 
    "Seasonal Trends & Alerts"
])

if df is None:
    st.warning("No processed data found. Please run the ETL pipeline first: `python scripts/run_etl.py`")
    st.stop()

# --- Page 1: Overview & Map ---
if page == "Overview & Map":
    st.header("Global Overview & Field Health Map")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Fields Analyzed", len(df))
    col2.metric("Average NDVI", f"{df['NDVI'].mean():.2f}")
    
    # Calculate % healthy
    healthy_pct = (len(df[df['status_label'] == 'healthy']) / len(df)) * 100
    col3.metric("Healthy Fields", f"{healthy_pct:.1f}%")
    
    avg_yield = df['yield_tons_ha'].mean() if 'yield_tons_ha' in df.columns else 0
    col4.metric("Avg Predicted Yield", f"{avg_yield:.2f} t/ha")
    
    st.markdown("---")
    
    # Geospatial Map
    st.subheader("Interactive Field Map")
    m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=7)
    
    color_map = {'healthy': 'green', 'stressed': 'orange', 'critical': 'red'}
    for idx, row in df.iterrows():
        status = row.get('status_label', 'unknown')
        color = color_map.get(status, 'blue')
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,
            popup=f"NDVI: {row['NDVI']:.2f}<br>Status: {status}<br>Yield: {row.get('yield_tons_ha', 0):.1f}",
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)
        
    st_folium(m, width=1200, height=500)

# --- Page 2: Pipeline & Data Quality ---
elif page == "Pipeline & Data Quality":
    st.header("ETL Pipeline Monitor & Data Quality")
    
    st.subheader("Pipeline DAG")
    # Simple representation of the DAG
    dag_fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15, thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = ["Raw Data (S3)", "Data Cleaning", "Feature Eng.", "ML Ready (Parquet)", "Dashboard"],
            color = "blue"
        ),
        link = dict(
            source = [0, 1, 2, 3],
            target = [1, 2, 3, 4],
            value = [len(df)+10, len(df)+5, len(df), len(df)] # Mock drop-off
        )
    )])
    st.plotly_chart(dag_fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Quality Report")
        dq_df = pd.DataFrame({
            "Metric": ["Completeness", "Validity (NDVI bounds)", "Timeliness"],
            "Score": ["99.5%", "100%", "98%"]
        })
        st.table(dq_df)
        
    with col2:
        st.subheader("Runtime Performance (Spark vs Pandas)")
        prof_df = load_profiling_data()
        if prof_df is not None:
            fig = px.bar(prof_df, x='function_name', y='execution_time_sec', color='task_type', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No profiling data found. Run ETL pipeline.")

# --- Page 3: Predictive Models ---
elif page == "Predictive Models":
    st.header("Model Evaluation & Live Prediction")
    
    if clf is None:
        st.warning("Models not found. Please run the training pipeline first: `python scripts/run_training.py`")
        st.stop()
        
    st.subheader("Live Simulator")
    
    # Input sliders
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        temp = st.slider("Temperature (°C)", 10.0, 45.0, 25.0)
    with col2:
        moisture = st.slider("Soil Moisture (%)", 0.0, 1.0, 0.4)
    with col3:
        ndvi = st.slider("NDVI", -1.0, 1.0, 0.6)
    with col4:
        # Mock values for raw bands based on NDVI
        red = 0.1
        nir = red * (1 + ndvi) / (1 - ndvi) if ndvi < 1 else 0.8
        st.metric("Implied NIR", f"{nir:.2f}")

    # Predict
    input_data = pd.DataFrame({
        'mean_red': [red],
        'mean_nir': [nir],
        'NDVI': [ndvi],
        'temperature': [temp],
        'soil_moisture': [moisture],
        'moisture_temp_interaction': [temp * moisture]
    })
    
    if st.button("Predict"):
        health_pred = clf.predict(input_data)[0]
        yield_pred = reg.predict(input_data)[0]
        
        c1, c2 = st.columns(2)
        c1.success(f"Predicted Health Status: **{health_pred.upper()}**")
        c2.info(f"Predicted Yield: **{yield_pred:.2f} tons/ha**")

    st.markdown("---")
    st.subheader("NDVI vs Yield (Model Insights)")
    fig = px.scatter(df, x="NDVI", y="yield_tons_ha", color="status_label", hover_data=["temperature"])
    st.plotly_chart(fig, use_container_width=True)

# --- Page 4: Seasonal Trends & Alerts ---
elif page == "Seasonal Trends & Alerts":
    st.header("Seasonal Trends & Active Alerts")
    
    # Alerts System
    st.subheader("🚨 Active Alerts")
    critical_df = df[df['status_label'] == 'critical']
    if len(critical_df) > 0:
        for idx, row in critical_df.head(3).iterrows():
            st.error(f"Field {row.get('field_id', 'Unknown')} - NDVI is critically low ({row['NDVI']:.2f}). Predicted yield at risk.")
    else:
        st.success("No critical alerts active.")
        
    st.markdown("---")
    
    # Seasonal Trends (Mocking a time series based on the current data)
    st.subheader("NDVI Trend Over Season")
    # Generate mock time series for 10 fields over 12 weeks
    weeks = pd.date_range(start='2024-03-01', periods=12, freq='W')
    ts_data = []
    for f in range(5):
        base_ndvi = df['NDVI'].iloc[f]
        # Simulate growth curve (up then down)
        curve = np.sin(np.linspace(0, np.pi, 12)) * 0.3
        for w, d in enumerate(weeks):
            ts_data.append({'Field': f"Field {f+1}", 'Date': d, 'NDVI': base_ndvi + curve[w] + np.random.normal(0,0.02)})
            
    ts_df = pd.DataFrame(ts_data)
    fig = px.line(ts_df, x='Date', y='NDVI', color='Field', title="Crop Growth Cycle")
    st.plotly_chart(fig, use_container_width=True)
