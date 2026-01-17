"""
ALIS - Streamlit Dashboard (v4.1)
Aadhaar Lifecycle Intelligence System

Features:
- Step-by-step guided tutorial
- Interactive map with zoom and pincode-level details
- Model evaluation with real metrics
- Risk analysis with proper distribution
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add backend to path
BACKEND_DIR = Path(__file__).parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Page config
st.set_page_config(
    page_title="ALIS - Aadhaar Lifecycle Intelligence System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .tutorial-box {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        border: 2px solid #00bcd4;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    .step-number {
        display: inline-block;
        width: 30px;
        height: 30px;
        background: #6366f1;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 30px;
        font-weight: bold;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)


# Session state for tutorial
if 'tutorial_step' not in st.session_state:
    st.session_state.tutorial_step = 0
if 'show_tutorial' not in st.session_state:
    st.session_state.show_tutorial = True


# Database connection
@st.cache_resource
def get_db_session():
    """Get database session."""
    from app.database import SessionLocal
    return SessionLocal()


# Pincode to State mapping (based on postal code ranges)
PINCODE_STATE_MAP = {
    '11': 'Delhi', '12': 'Haryana', '13': 'Punjab', '14': 'Punjab',
    '15': 'Punjab', '16': 'Punjab', '17': 'Himachal Pradesh',
    '18': 'Jammu & Kashmir', '19': 'Jammu & Kashmir',
    '20': 'Uttar Pradesh', '21': 'Uttar Pradesh', '22': 'Uttar Pradesh',
    '23': 'Uttar Pradesh', '24': 'Uttar Pradesh', '25': 'Uttar Pradesh',
    '26': 'Uttar Pradesh', '27': 'Uttar Pradesh', '28': 'Uttar Pradesh',
    '30': 'Rajasthan', '31': 'Rajasthan', '32': 'Rajasthan',
    '33': 'Rajasthan', '34': 'Rajasthan',
    '36': 'Gujarat', '37': 'Gujarat', '38': 'Gujarat', '39': 'Gujarat',
    '40': 'Maharashtra', '41': 'Maharashtra', '42': 'Maharashtra',
    '43': 'Maharashtra', '44': 'Maharashtra',
    '45': 'Madhya Pradesh', '46': 'Madhya Pradesh', '47': 'Madhya Pradesh', 
    '48': 'Madhya Pradesh', '49': 'Chhattisgarh',
    '50': 'Telangana', '51': 'Telangana', '52': 'Andhra Pradesh',
    '53': 'Andhra Pradesh', '56': 'Karnataka', '57': 'Karnataka',
    '58': 'Karnataka', '59': 'Karnataka',
    '60': 'Tamil Nadu', '61': 'Tamil Nadu', '62': 'Tamil Nadu',
    '63': 'Tamil Nadu', '64': 'Tamil Nadu', '67': 'Kerala',
    '68': 'Kerala', '69': 'Kerala',
    '70': 'West Bengal', '71': 'West Bengal', '72': 'West Bengal', 
    '73': 'West Bengal', '74': 'West Bengal',
    '75': 'Odisha', '76': 'Odisha', '77': 'Odisha',
    '78': 'Assam', '79': 'Arunachal Pradesh',
    '80': 'Bihar', '81': 'Bihar', '82': 'Bihar', '83': 'Bihar', '84': 'Bihar',
    '85': 'Jharkhand', '86': 'Jharkhand',
}

STATE_COORDS = {
    'Andhra Pradesh': (15.9129, 79.7400), 'Bihar': (25.0961, 85.3131),
    'Chhattisgarh': (21.2787, 81.8661), 'Gujarat': (22.2587, 71.1924),
    'Jharkhand': (23.6102, 85.2799), 'Karnataka': (15.3173, 75.7139),
    'Kerala': (10.8505, 76.2711), 'Madhya Pradesh': (22.9734, 78.6569),
    'Maharashtra': (19.7515, 75.7139), 'Odisha': (20.9517, 85.0985),
    'Punjab': (31.1471, 75.3412), 'Rajasthan': (27.0238, 74.2179),
    'Tamil Nadu': (11.1271, 78.6569), 'Telangana': (18.1124, 79.0193),
    'Uttar Pradesh': (26.8467, 80.9462), 'West Bengal': (22.9868, 87.8550),
    'Delhi': (28.7041, 77.1025), 'Haryana': (29.0588, 76.0856),
    'Himachal Pradesh': (31.1048, 77.1734), 'Jammu & Kashmir': (33.7782, 76.5762),
    'Uttarakhand': (30.0668, 79.0193), 'Goa': (15.2993, 74.1240),
    'Assam': (26.2006, 92.9376), 'Meghalaya': (25.4670, 91.3662),
    'Tripura': (23.9408, 91.9882), 'Arunachal Pradesh': (28.2180, 94.7278),
}


def get_state_from_pincode(pincode: str) -> str:
    """Get state from pincode first 2 digits."""
    prefix = str(pincode)[:2] if pincode else '00'
    return PINCODE_STATE_MAP.get(prefix, 'Other')


@st.cache_data(ttl=300)
def load_pincode_metrics():
    """Load pincode metrics from database."""
    db = get_db_session()
    from app.models.db_models import PincodeMetric
    
    metrics = db.query(PincodeMetric).all()
    
    data = []
    for m in metrics:
        state = m.state if m.state and m.state != 'Unknown' else get_state_from_pincode(m.pincode)
        data.append({
            'pincode': m.pincode,
            'state': state,
            'district': m.district or '',
            'bio_risk': m.bio_risk_score or 0,
            'demo_risk': m.demo_risk_score or 0,
            'mobile_risk': m.mobile_risk_score or 0,
            'overall_risk': m.overall_risk_score or 0,
            'risk_category': m.risk_category or 'LOW',
            'bio_updates': m.total_bio_updates or 0,
            'demo_updates': m.total_demo_updates or 0,
            'enrollments': m.total_enrollments or 0,
            'bio_trend': m.bio_trend,
            'demo_trend': m.demo_trend,
            'cluster_id': m.cluster_id
        })
    
    return pd.DataFrame(data)


@st.cache_data(ttl=300)
def load_cluster_data():
    """Load cluster information."""
    db = get_db_session()
    from app.models.db_models import PincodeCluster
    
    clusters = db.query(PincodeCluster).all()
    
    return pd.DataFrame([{
        'cluster_id': c.id,
        'name': c.name,
        'pincode_count': c.pincode_count or 0,
        'avg_bio_updates': c.avg_bio_updates or 0,
        'avg_demo_updates': c.avg_demo_updates or 0,
        'profile': c.profile or 'Unknown'
    } for c in clusters])


@st.cache_data(ttl=300)
def load_model_results():
    """Load model training results."""
    results_path = BACKEND_DIR / "data" / "models" / "training_results.json"
    if results_path.exists():
        with open(results_path) as f:
            raw = json.load(f)
            
            parsed = {
                'timestamp': raw.get('training_timestamp', ''),
                'samples': raw.get('samples', 0),
                'metrics': []
            }
            
            for metric_type, models in raw.get('models', {}).items():
                for model_name, model_data in models.items():
                    meta = model_data.get('metadata', {})
                    parsed['metrics'].append({
                        'type': metric_type.upper(),
                        'model': model_name.upper(),
                        'status': model_data.get('status', 'unknown'),
                        'mae': meta.get('in_sample_mae'),
                        'rmse': meta.get('in_sample_rmse'),
                        'aic': meta.get('aic'),
                        'converged': meta.get('converged', True)
                    })
            
            return parsed
    return {}


@st.cache_data(ttl=300)
def load_raw_updates():
    """Load raw update data for trends."""
    db = get_db_session()
    from app.models.db_models import RawUpdate
    from sqlalchemy import func
    
    data = db.query(
        RawUpdate.date,
        func.sum(RawUpdate.bio_total).label('bio_total'),
        func.sum(RawUpdate.demo_total).label('demo_total'),
        func.sum(RawUpdate.enrol_total).label('enrol_total')
    ).group_by(RawUpdate.date).order_by(RawUpdate.date).all()
    
    return pd.DataFrame([{
        'date': d.date, 'bio_total': d.bio_total or 0,
        'demo_total': d.demo_total or 0, 'enrol_total': d.enrol_total or 0
    } for d in data])


# Sidebar
st.sidebar.markdown("# 🔐 ALIS")
st.sidebar.markdown("**Aadhaar Lifecycle Intelligence System**")
st.sidebar.divider()

# Tutorial toggle
show_tutorial = st.sidebar.checkbox("📚 Show Tutorial", value=st.session_state.show_tutorial)
st.session_state.show_tutorial = show_tutorial

st.sidebar.divider()

page = st.sidebar.radio(
    "📍 Navigation",
    ["🏠 Home & Tutorial", "📊 Dashboard", "🗺️ Map View", "📈 Analytics", 
     "🧠 Model Evaluation", "⚠️ Anomalies", "🔮 Predictions"]
)

st.sidebar.divider()
st.sidebar.markdown("### Quick Stats")

# Load data
try:
    df_metrics = load_pincode_metrics()
    df_clusters = load_cluster_data()
    df_trends = load_raw_updates()
    model_results = load_model_results()
    
    st.sidebar.metric("Pincodes", f"{len(df_metrics):,}")
    critical = len(df_metrics[df_metrics['risk_category'] == 'CRITICAL'])
    high = len(df_metrics[df_metrics['risk_category'] == 'HIGH'])
    st.sidebar.metric("Critical", f"{critical:,}")
    st.sidebar.metric("High Risk", f"{high:,}")
    
except Exception as e:
    st.sidebar.error(f"Data error: {e}")
    df_metrics = pd.DataFrame()
    df_clusters = pd.DataFrame()
    df_trends = pd.DataFrame()
    model_results = {}


# =====================
# HOME & TUTORIAL PAGE
# =====================
if page == "🏠 Home & Tutorial":
    st.markdown('<h1 class="main-header">🔐 Welcome to ALIS</h1>', unsafe_allow_html=True)
    st.markdown("### Aadhaar Lifecycle Intelligence System")
    
    st.markdown("""
    <div class="tutorial-box">
    <h3>👋 Welcome! Let's take a quick tour.</h3>
    <p>ALIS is an AI-powered system that analyzes Aadhaar biometric and demographic update patterns 
    to identify potential fraud, optimize resource allocation, and predict future demand.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step by step guide
    st.markdown("## 📚 Step-by-Step Guide")
    
    with st.expander("**Step 1: Understanding Risk Levels**", expanded=True):
        st.markdown("""
        <div class="info-box">
        <p><span class="step-number">1</span> <strong>Risk levels help prioritize which areas need attention:</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("### 🔴 Critical")
            st.markdown("**Score: 80-100**")
            st.markdown("Immediate investigation needed. Possible fraud patterns.")
        with col2:
            st.markdown("### 🟠 High")
            st.markdown("**Score: 60-79**")
            st.markdown("Elevated activity. Monitor closely.")
        with col3:
            st.markdown("### 🟡 Medium")
            st.markdown("**Score: 40-59**")
            st.markdown("Some anomalies detected. Review periodically.")
        with col4:
            st.markdown("### 🟢 Low")
            st.markdown("**Score: 0-39**")
            st.markdown("Normal operation. No action needed.")
    
    with st.expander("**Step 2: Dashboard Overview**"):
        st.markdown("""
        The **📊 Dashboard** page shows:
        - **KPI Cards**: Quick summary of risk distribution
        - **Risk Distribution Chart**: Pie chart of risk categories
        - **Trend Chart**: Bio/Demo updates over time
        - **Top Priority Table**: Pincodes needing immediate attention
        
        👉 Go to **📊 Dashboard** to see the overview.
        """)
    
    with st.expander("**Step 3: Geographic Analysis**"):
        st.markdown("""
        The **🗺️ Map View** page shows:
        - Interactive map of India with risk markers
        - Click on markers to see details
        - Filter by state or risk level
        - Zoom in to see pincode-level details
        
        👉 Go to **🗺️ Map View** to explore geographically.
        """)
    
    with st.expander("**Step 4: Advanced Analytics**"):
        st.markdown("""
        The **📈 Analytics** page provides:
        - **Correlation Heatmap**: How risk factors relate to each other
        - **Cluster Analysis**: Grouping of similar pincodes
        - **Distribution Charts**: Statistical analysis
        
        👉 Go to **📈 Analytics** for deep dives.
        """)
    
    with st.expander("**Step 5: Model Evaluation**"):
        st.markdown("""
        The **🧠 Model Evaluation** page explains:
        - **MAE** (Mean Absolute Error): Lower = Better predictions
        - **RMSE** (Root Mean Squared Error): Penalizes large errors
        - **AIC** (Akaike Information Criterion): Model quality
        
        👉 Go to **🧠 Model Evaluation** to see how well our AI performs.
        """)
    
    with st.expander("**Step 6: Anomaly & Predictions**"):
        st.markdown("""
        - **⚠️ Anomalies**: Unusual patterns detected by the AI
        - **🔮 Predictions**: Generate forecasts for any pincode
        
        👉 Try generating a forecast for pincode **110001** (Delhi)
        """)
    
    # Quick start actions
    st.markdown("---")
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Go to Dashboard", use_container_width=True):
            st.switch_page("pages/dashboard.py") if hasattr(st, 'switch_page') else st.info("Click '📊 Dashboard' in sidebar")
    with col2:
        if st.button("🗺️ View Map", use_container_width=True):
            st.info("Click '🗺️ Map View' in sidebar")
    with col3:
        if st.button("🔮 Make Prediction", use_container_width=True):
            st.info("Click '🔮 Predictions' in sidebar")


# =====================
# DASHBOARD PAGE
# =====================
elif page == "📊 Dashboard":
    st.markdown('<h1 class="main-header">📊 Risk Dashboard</h1>', unsafe_allow_html=True)
    
    if show_tutorial:
        st.info("💡 **Tutorial**: This dashboard shows the overall risk status. The KPI cards at the top summarize how many pincodes fall into each risk category. Scroll down to see trends and priority areas.")
    
    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    if not df_metrics.empty:
        critical = len(df_metrics[df_metrics['risk_category'] == 'CRITICAL'])
        high = len(df_metrics[df_metrics['risk_category'] == 'HIGH'])
        medium = len(df_metrics[df_metrics['risk_category'] == 'MEDIUM'])
        low = len(df_metrics[df_metrics['risk_category'] == 'LOW'])
        total = len(df_metrics)
        
        col1.metric("🔴 Critical", f"{critical:,}", f"{critical/total*100:.1f}%")
        col2.metric("🟠 High", f"{high:,}", f"{high/total*100:.1f}%")
        col3.metric("🟡 Medium", f"{medium:,}", f"{medium/total*100:.1f}%")
        col4.metric("🟢 Low", f"{low:,}", f"{low/total*100:.1f}%")
        col5.metric("📊 Total", f"{total:,}")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Risk Distribution")
        if show_tutorial:
            st.caption("This pie chart shows what percentage of pincodes are in each risk category.")
        
        if not df_metrics.empty:
            risk_counts = df_metrics['risk_category'].value_counts()
            fig = px.pie(
                values=risk_counts.values, names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map={'CRITICAL': '#e74c3c', 'HIGH': '#e67e22', 
                                   'MEDIUM': '#f1c40f', 'LOW': '#2ecc71'},
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Update Trends (30 Days)")
        if show_tutorial:
            st.caption("This shows daily bio/demo update volumes. Large spikes may indicate unusual activity.")
        
        if not df_trends.empty:
            recent = df_trends.tail(30)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recent['date'], y=recent['bio_total'], 
                                    name='Bio Updates', line=dict(color='#9b59b6', width=2)))
            fig.add_trace(go.Scatter(x=recent['date'], y=recent['demo_total'], 
                                    name='Demo Updates', line=dict(color='#2ecc71', width=2)))
            fig.update_layout(height=400, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk Score Distribution
    st.subheader("📉 Risk Score Distribution")
    if show_tutorial:
        st.caption("This histogram shows how risk scores are distributed. Most pincodes cluster around certain scores.")
    
    if not df_metrics.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data=df_metrics, x='overall_risk', bins=50, kde=True, color='#3498db', ax=ax)
            ax.axvline(x=80, color='red', linestyle='--', linewidth=2, label='Critical (80)')
            ax.axvline(x=60, color='orange', linestyle='--', linewidth=2, label='High (60)')
            ax.axvline(x=40, color='#f1c40f', linestyle='--', linewidth=2, label='Medium (40)')
            ax.set_xlabel('Risk Score')
            ax.set_ylabel('Number of Pincodes')
            ax.legend()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("### Score Breakdown")
            st.markdown(f"- **Mean**: {df_metrics['overall_risk'].mean():.1f}")
            st.markdown(f"- **Median**: {df_metrics['overall_risk'].median():.1f}")
            st.markdown(f"- **Std Dev**: {df_metrics['overall_risk'].std():.1f}")
            st.markdown(f"- **Min**: {df_metrics['overall_risk'].min():.1f}")
            st.markdown(f"- **Max**: {df_metrics['overall_risk'].max():.1f}")
    
    # Priority Table
    st.subheader("🚨 Top Priority Pincodes")
    if show_tutorial:
        st.caption("These pincodes have the highest risk scores and should be investigated first.")
    
    if not df_metrics.empty:
        priority = df_metrics.nlargest(10, 'overall_risk')[
            ['pincode', 'state', 'district', 'overall_risk', 'risk_category', 'bio_updates', 'demo_updates']
        ]
        priority.columns = ['Pincode', 'State', 'District', 'Risk Score', 'Category', 'Bio Updates', 'Demo Updates']
        st.dataframe(priority, use_container_width=True, hide_index=True)


# =====================
# MAP VIEW PAGE
# =====================
elif page == "🗺️ Map View":
    st.markdown("# 🗺️ Interactive Risk Map")
    
    if show_tutorial:
        st.info("""
        💡 **Tutorial**: This map shows risk distribution across India.
        - **Bubble Size** = Number of pincodes
        - **Color** = Risk level (Red=High, Green=Low)
        - **Hover** over bubbles for details
        - Use the zoom controls or scroll to zoom in
        """)
    
    if not df_metrics.empty:
        # State-level aggregation
        state_data = df_metrics.groupby('state').agg({
            'overall_risk': 'mean',
            'bio_updates': 'sum',
            'demo_updates': 'sum',
            'pincode': 'count'
        }).reset_index()
        state_data.columns = ['state', 'avg_risk', 'bio_updates', 'demo_updates', 'pincode_count']
        
        # Add coordinates
        state_data['lat'] = state_data['state'].map(lambda x: STATE_COORDS.get(x, (None, None))[0])
        state_data['lon'] = state_data['state'].map(lambda x: STATE_COORDS.get(x, (None, None))[1])
        state_data = state_data.dropna(subset=['lat', 'lon'])
        
        if not state_data.empty:
            # Map type selection
            map_type = st.radio("Map Style", ["Dark", "Light", "Satellite"], horizontal=True)
            style_map = {"Dark": "carto-darkmatter", "Light": "carto-positron", "Satellite": "open-street-map"}
            
            fig = px.scatter_mapbox(
                state_data,
                lat='lat', lon='lon',
                size='pincode_count',
                color='avg_risk',
                color_continuous_scale=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'],
                range_color=[0, 100],
                hover_name='state',
                hover_data={
                    'pincode_count': True,
                    'avg_risk': ':.1f',
                    'bio_updates': ':,.0f',
                    'lat': False, 'lon': False
                },
                zoom=4,
                center={"lat": 22.5, "lon": 82.5},
                mapbox_style=style_map[map_type],
                size_max=60
            )
            fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # State summary
            st.subheader("📋 State Summary")
            display_df = state_data[['state', 'avg_risk', 'pincode_count', 'bio_updates']].copy()
            display_df.columns = ['State', 'Avg Risk', 'Pincodes', 'Bio Updates']
            display_df = display_df.sort_values('Avg Risk', ascending=False)
            display_df['Avg Risk'] = display_df['Avg Risk'].round(1)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No valid state coordinates found.")
    else:
        st.warning("No data available.")


# =====================
# ANALYTICS PAGE
# =====================
elif page == "📈 Analytics":
    st.markdown("# 📈 Advanced Analytics")
    
    if show_tutorial:
        st.info("💡 **Tutorial**: This section provides statistical analysis. The correlation heatmap shows how different risk factors are related - values close to 1 mean they move together.")
    
    if not df_metrics.empty:
        tab1, tab2, tab3 = st.tabs(["🔗 Correlation", "🎯 Clusters", "📊 Distributions"])
        
        with tab1:
            st.subheader("Risk Factor Correlation")
            
            numeric_cols = ['bio_risk', 'demo_risk', 'overall_risk', 'bio_updates', 'demo_updates']
            available_cols = [c for c in numeric_cols if c in df_metrics.columns]
            corr_data = df_metrics[available_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_data, annot=True, cmap='RdYlGn_r', center=0, 
                       square=True, linewidths=0.5, ax=ax, fmt='.2f')
            st.pyplot(fig)
            plt.close()
        
        with tab2:
            st.subheader("Cluster Analysis")
            
            if not df_clusters.empty:
                fig = px.bar(df_clusters, x='name', y='pincode_count',
                           color='avg_bio_updates', color_continuous_scale='Viridis',
                           title='Cluster Sizes')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(df_clusters, use_container_width=True, hide_index=True)
            else:
                st.info("No cluster data available.")
        
        with tab3:
            st.subheader("Distribution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                order = [c for c in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'] if c in df_metrics['risk_category'].values]
                if order:
                    sns.boxplot(data=df_metrics, x='risk_category', y='bio_updates', order=order, ax=ax)
                    ax.set_yscale('log')
                    ax.set_title('Bio Updates by Risk Category')
                st.pyplot(fig)
                plt.close()
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                if order:
                    sns.violinplot(data=df_metrics, x='risk_category', y='overall_risk', order=order, ax=ax)
                    ax.set_title('Risk Score Distribution')
                st.pyplot(fig)
                plt.close()


# =====================
# MODEL EVALUATION PAGE
# =====================
elif page == "🧠 Model Evaluation":
    st.markdown("# 🧠 Model Evaluation & Metrics")
    
    if show_tutorial:
        st.info("""
        💡 **Tutorial**: This page shows how well our ML models perform.
        - **MAE** (Mean Absolute Error): Average prediction error - lower is better
        - **RMSE** (Root Mean Squared Error): Penalizes large errors more - lower is better
        - **AIC**: Model quality score - lower is better (for SARIMA)
        """)
    
    if model_results and model_results.get('metrics'):
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Training Samples", model_results.get('samples', 0))
        
        ts = model_results.get('timestamp', '')
        col2.metric("Training Date", ts[:10] if ts else 'N/A')
        col3.metric("Status", "✅ Trained")
        
        st.divider()
        
        # Model metrics table
        metrics_df = pd.DataFrame(model_results['metrics'])
        
        for mtype in ['BIO', 'DEMO', 'MOBILE']:
            type_df = metrics_df[metrics_df['type'] == mtype]
            if not type_df.empty:
                st.subheader(f"📊 {mtype} Models")
                
                display_df = type_df[['model', 'status', 'mae', 'rmse', 'aic']].copy()
                display_df.columns = ['Model', 'Status', 'MAE', 'RMSE', 'AIC']
                
                for col in ['MAE', 'RMSE', 'AIC']:
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else 'N/A'
                    )
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Visual comparison
        st.divider()
        st.subheader("📊 Model Performance Comparison")
        
        xgb_df = metrics_df[(metrics_df['model'] == 'XGBOOST') & (metrics_df['mae'].notna())]
        
        if not xgb_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            sns.barplot(data=xgb_df, x='type', y='mae', ax=axes[0], palette='viridis')
            axes[0].set_title('Mean Absolute Error by Metric')
            axes[0].set_xlabel('Metric Type')
            axes[0].set_ylabel('MAE')
            
            sns.barplot(data=xgb_df, x='type', y='rmse', ax=axes[1], palette='plasma')
            axes[1].set_title('Root Mean Squared Error by Metric')
            axes[1].set_xlabel('Metric Type')
            axes[1].set_ylabel('RMSE')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.warning("No model training results found. Run: `python app.py`")


# =====================
# ANOMALIES PAGE
# =====================
elif page == "⚠️ Anomalies":
    st.markdown("# ⚠️ Anomaly Detection")
    
    if show_tutorial:
        st.info("""
        💡 **Tutorial**: Anomalies are unusual patterns that deviate from normal behavior.
        - **SPIKE**: Sudden increase (possible fraud or migration)
        - **DROP**: Sudden decrease (possible system issue)
        
        We use multiple methods (Z-Score, IQR, Isolation Forest) and flag anomalies when 2+ methods agree.
        """)
    
    try:
        db = get_db_session()
        from app.models.db_models import Anomaly
        
        anomalies = db.query(Anomaly).order_by(Anomaly.confidence_score.desc()).limit(100).all()
        
        if anomalies:
            st.metric("Anomalies Detected", len(anomalies))
            
            anomaly_data = [{
                'Date': a.detected_date,
                'Type': a.anomaly_type,
                'Metric': a.metric_affected,
                'Expected': f"{a.expected_value:,.0f}",
                'Actual': f"{a.actual_value:,.0f}",
                'Deviation': f"{a.deviation_percent:+.1f}%",
                'Severity': a.severity,
                'Confidence': f"{a.confidence_score:.0%}"
            } for a in anomalies]
            
            df_anom = pd.DataFrame(anomaly_data)
            
            severity_filter = st.multiselect("Filter by Severity", 
                ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'], default=['CRITICAL', 'HIGH'])
            
            if severity_filter:
                df_anom = df_anom[df_anom['Severity'].isin(severity_filter)]
            
            if not df_anom.empty:
                st.dataframe(df_anom, use_container_width=True, hide_index=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(df_anom, names='Severity', title='By Severity',
                               color_discrete_map={'CRITICAL': '#e74c3c', 'HIGH': '#e67e22'})
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.pie(df_anom, names='Type', title='By Type')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📭 No anomalies detected. Run: `python app.py` to perform detection.")
    except Exception as e:
        st.error(f"Error: {e}")


# =====================
# PREDICTIONS PAGE
# =====================
elif page == "🔮 Predictions":
    st.markdown("# 🔮 Forecasting & Predictions")
    
    if show_tutorial:
        st.info("""
        💡 **Tutorial**: Enter a pincode to generate a forecast.
        
        1. Enter a 6-digit pincode (e.g., 110001 for Delhi)
        2. Select forecast horizon (7-90 days)
        3. Click "Generate Forecast"
        
        The model will predict future bio and demo update volumes based on historical patterns.
        """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        pincode = st.text_input("📍 Enter Pincode", placeholder="e.g., 110001")
    with col2:
        horizon = st.slider("Days to Forecast", 7, 90, 30)
    
    if st.button("🚀 Generate Forecast", type="primary"):
        if pincode and len(pincode) == 6:
            try:
                db = get_db_session()
                from app.services.forecaster import ForecasterService
                
                forecaster = ForecasterService(db)
                
                with st.spinner("Generating forecast..."):
                    bio_result = forecaster.generate_forecast(pincode, 'bio', horizon)
                    demo_result = forecaster.generate_forecast(pincode, 'demo', horizon)
                
                if bio_result.get('predictions') or demo_result.get('predictions'):
                    st.success(f"✅ Forecast generated for pincode {pincode}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if bio_result.get('predictions'):
                            st.subheader("📊 Bio Forecast")
                            bio_df = pd.DataFrame(bio_result['predictions'])
                            fig = px.line(bio_df, x='date', y='predicted', title=f"Bio - {horizon} Days")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if demo_result.get('predictions'):
                            st.subheader("📊 Demo Forecast")
                            demo_df = pd.DataFrame(demo_result['predictions'])
                            fig = px.line(demo_df, x='date', y='predicted', title=f"Demo - {horizon} Days")
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Insufficient data for this pincode.")
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure models are trained: `python app.py`")
        else:
            st.warning("Please enter a valid 6-digit pincode.")


# Footer
st.sidebar.divider()
st.sidebar.markdown("💡 **ALIS v4.1**")
st.sidebar.markdown("UIDAI Hackathon 2026")
