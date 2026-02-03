import os
import requests
import datetime
import pytz
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Shift & Leave Report",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

st.markdown("""
<style>
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .stTable {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }
        
        h1 {
            font-size: 1.5rem !important;
        }
        h2 {
            font-size: 1.3rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
        }
        
        .stButton > button {
            width: 100%;
            margin: 0.25rem 0;
            padding: 0.5rem;
            font-size: 1rem;
        }
        
        .stProgress > div > div {
            height: 10px;
        }
        
        div[data-testid="stExpander"] {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }
    }
    
    table {
        display: block;
        overflow-x: auto;
        white-space: nowrap;
    }
    
    button[kind="header"] {
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    
    .mobile-card {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
</style>

<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Shift Report">
""", unsafe_allow_html=True)

try:
    cache_data = st.cache_data
except AttributeError:
    cache_data = st.cache

GENDER_MAPPING = st.secrets.get("gender_mapping", {})
DEFAULT_IGNORED_IDS = st.secrets.get("ignored_user_ids", [])

def get_user_gender(user_id):
    return GENDER_MAPPING.get(str(user_id), 'Unknown')

def get_gender_display_name(gender_code):
    gender_names = {'M': 'Male', 'F': 'Female', 'T': 'Terminal', 'Unknown': 'Unknown'}
    return gender_names.get(gender_code, gender_code)

def get_monthly_period(year, month):
    start_date = datetime.date(year, month, 11)
    
    if month == 12:
        next_month = 1
        next_year = year + 1
    else:
        next_month = month + 1
        next_year = year
    
    end_date = datetime.date(next_year, next_month, 10)
    return start_date, end_date

def get_current_period():
    today = datetime.date.today()
    
    if today.day < 11:
        if today.month == 1:
            target_month = 12
            target_year = today.year - 1
        else:
            target_month = today.month - 1
            target_year = today.year
    else:
        target_month = today.month
        target_year = today.year
    
    return get_monthly_period(target_year, target_month)

def format_period_name(start_date, end_date):
    return f"{start_date.strftime('%b %Y')} Period ({start_date.strftime('%d/%m/%y')} - {end_date.strftime('%d/%m/%y')})"

def calculate_expected_hours(weekly_hours, start_date, end_date):
    """Calculate expected hours based on weekly hours and period length"""
    if weekly_hours is None or weekly_hours <= 0:
        return None
    
    days_in_period = (end_date - start_date).days + 1
    weeks = days_in_period / 7
    expected = weeks * weekly_hours
    return round(expected, 2)

def get_variance_color(actual_hours, expected_hours):
    """Determine color based on variance"""
    if expected_hours is None or expected_hours <= 0:
        return None
    
    if actual_hours < expected_hours:
        return "red"
    elif actual_hours > expected_hours:
        return "green"
    else:
        return "gray"

def get_variance_emoji(actual_hours, expected_hours):
    """Get emoji indicator for variance"""
    if expected_hours is None:
        return "❓"
    
    if actual_hours < expected_hours:
        variance_pct = ((expected_hours - actual_hours) / expected_hours) * 100
        return f"🔴 ({variance_pct:.1f}% under)"
    elif actual_hours > expected_hours:
        variance_pct = ((actual_hours - expected_hours) / expected_hours) * 100
        return f"🟢 ({variance_pct:.1f}% over)"
    else:
        return "🟡 (On target)"

# =====================================================================
# ADVANCED ANALYTICS FUNCTIONS
# =====================================================================

def render_analytics_module(user_reports, start_date, end_date):
    """Render all advanced analytics visualizations"""
    
    st.markdown("---")
    st.header("📈 Advanced Analytics Dashboard")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(user_reports)
    df['full_name'] = df['first_name'] + ' ' + df['last_name']
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Distribution", "📍 Location Analysis", "👥 Demographics",
        "🔥 Heatmap", "🎯 Clustering", "💷 Cost Analysis",
        "📈 Compliance", "🔮 Benchmarking", "⚠️ Outliers"
    ])
    
    with tab1:
        render_distribution_analysis(df)
    
    with tab2:
        render_location_analysis(df)
    
    with tab3:
        render_demographic_analysis(df)
    
    with tab4:
        render_heatmap_analysis(df)
    
    with tab5:
        render_clustering_analysis(df)
    
    with tab6:
        render_cost_analysis(df, start_date, end_date)
    
    with tab7:
        render_compliance_analysis(df)
    
    with tab8:
        render_benchmarking_analysis(df)
    
    with tab9:
        render_outlier_analysis(df)


def render_distribution_analysis(df):
    """Tab 1: Distribution & Statistical Analysis"""
    st.subheader("Hours Distribution Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Hours", f"{df['unified_total'].mean():.1f}")
    with col2:
        st.metric("Median Hours", f"{df['unified_total'].median():.1f}")
    with col3:
        st.metric("Std Dev", f"{df['unified_total'].std():.1f}")
    
    # Histogram
    fig = px.histogram(
        df, 
        x='unified_total', 
        nbins=20,
        title='Distribution of Total Hours Worked',
        labels={'unified_total': 'Hours', 'count': 'Number of Staff'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.add_vline(x=df['unified_total'].mean(), line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {df['unified_total'].mean():.1f}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Box plot
    fig_box = px.box(
        df,
        y='unified_total',
        title='Hours Distribution - Box Plot',
        labels={'unified_total': 'Hours'},
        points='outliers'
    )
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Shift vs Leave breakdown
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(
            values=[df['total_shift_hours'].sum(), df['total_leave_hours'].sum()],
            names=['Shift Hours', 'Leave Hours'],
            title='Total Hours Breakdown'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_scatter = px.scatter(
            df,
            x='total_shift_hours',
            y='total_leave_hours',
            hover_data=['full_name'],
            title='Shift Hours vs Leave Hours',
            labels={'total_shift_hours': 'Shift Hours', 'total_leave_hours': 'Leave Hours'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Statistics table
    st.subheader("Detailed Statistics")
    stats_df = pd.DataFrame({
        'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', '25%', '75%', 'Max'],
        'Total Hours': [
            len(df),
            f"{df['unified_total'].mean():.2f}",
            f"{df['unified_total'].median():.2f}",
            f"{df['unified_total'].std():.2f}",
            f"{df['unified_total'].min():.2f}",
            f"{df['unified_total'].quantile(0.25):.2f}",
            f"{df['unified_total'].quantile(0.75):.2f}",
            f"{df['unified_total'].max():.2f}"
        ],
        'Shift Hours': [
            len(df),
            f"{df['total_shift_hours'].mean():.2f}",
            f"{df['total_shift_hours'].median():.2f}",
            f"{df['total_shift_hours'].std():.2f}",
            f"{df['total_shift_hours'].min():.2f}",
            f"{df['total_shift_hours'].quantile(0.25):.2f}",
            f"{df['total_shift_hours'].quantile(0.75):.2f}",
            f"{df['total_shift_hours'].max():.2f}"
        ],
        'Leave Hours': [
            len(df),
            f"{df['total_leave_hours'].mean():.2f}",
            f"{df['total_leave_hours'].median():.2f}",
            f"{df['total_leave_hours'].std():.2f}",
            f"{df['total_leave_hours'].min():.2f}",
            f"{df['total_leave_hours'].quantile(0.25):.2f}",
            f"{df['total_leave_hours'].quantile(0.75):.2f}",
            f"{df['total_leave_hours'].max():.2f}"
        ]
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)


def render_location_analysis(df):
    """Tab 2: Location-based Comparative Analysis"""
    st.subheader("Location Performance Analysis")
    
    # Check if location data exists
    if 'location' not in df.columns or df['location'].isna().all():
        st.info("Location data not available in current dataset. This would require location info from shifts.")
        return
    
    location_summary = df.groupby('location').agg({
        'unified_total': ['count', 'mean', 'sum', 'std'],
        'total_shift_hours': 'mean',
        'total_leave_hours': 'mean',
        'variance': 'mean'
    }).round(2)
    
    st.dataframe(location_summary, use_container_width=True)
    
    # Bar chart by location
    location_avg = df.groupby('location')['unified_total'].agg(['mean', 'count']).reset_index()
    
    fig = px.bar(
        location_avg,
        x='location',
        y='mean',
        title='Average Hours by Location',
        labels={'mean': 'Average Hours', 'location': 'Location'},
        color='mean',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Variance by location
    location_variance = df.groupby('location')['variance'].mean().reset_index()
    fig_var = px.bar(
        location_variance,
        x='location',
        y='variance',
        title='Average Variance by Location',
        labels={'variance': 'Avg Variance (hours)', 'location': 'Location'},
        color='variance',
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig_var, use_container_width=True)


def render_demographic_analysis(df):
    """Tab 3: Gender & Role-based Analysis"""
    st.subheader("Demographic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Hours by Gender**")
        gender_summary = df.groupby('gender').agg({
            'unified_total': ['count', 'mean', 'sum']
        }).round(2)
        st.dataframe(gender_summary, use_container_width=True)
        
        # Box plot by gender
        fig_gender = px.box(
            df,
            x='gender',
            y='unified_total',
            title='Hours Distribution by Gender',
            labels={'unified_total': 'Hours', 'gender': 'Gender'},
            points='outliers'
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        st.write("**Variance by Gender**")
        gender_var = df.groupby('gender')['variance'].agg(['mean', 'count']).round(2)
        st.dataframe(gender_var, use_container_width=True)
        
        # Gender variance comparison
        fig_var_gender = px.bar(
            df.groupby('gender')['variance'].mean().reset_index(),
            x='gender',
            y='variance',
            title='Average Variance by Gender',
            labels={'variance': 'Avg Variance', 'gender': 'Gender'},
            color='variance',
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig_var_gender, use_container_width=True)
    
    # Equity check
    st.subheader("Equity Analysis")
    equity_df = df.groupby('gender').agg({
        'unified_total': ['mean', 'min', 'max', 'std'],
        'variance': 'mean'
    }).round(2)
    st.dataframe(equity_df, use_container_width=True)


def render_heatmap_analysis(df):
    """Tab 4: Heatmap Analysis"""
    st.subheader("Staff Utilization Heatmap")
    
    # Create pivot table for heatmap - Top 20 staff by hours
    top_staff = df.nlargest(20, 'unified_total')
    
    # Simple heatmap showing key metrics per staff member
    heatmap_data = top_staff[['full_name', 'total_shift_hours', 'total_leave_hours', 'unified_total', 'variance']].copy()
    heatmap_data_normalized = heatmap_data.set_index('full_name')
    
    # Normalize for heatmap coloring
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    heatmap_normalized = pd.DataFrame(
        scaler.fit_transform(heatmap_data_normalized),
        columns=heatmap_data_normalized.columns,
        index=heatmap_data_normalized.index
    )
    
    fig = px.imshow(
        heatmap_normalized.T,
        labels=dict(x="Staff Member", y="Metric", color="Normalized Value"),
        title="Top 20 Staff - Performance Heatmap",
        color_continuous_scale='RdYlGn',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Alternative: Variance heatmap by expected compliance
    st.subheader("Variance Compliance Heatmap")
    variance_indicator = top_staff.copy()
    variance_indicator['Status'] = variance_indicator['variance'].apply(
        lambda x: 'Under' if x < 0 else 'Over' if x > 0 else 'On Target'
    )
    
    fig_var = px.bar(
        variance_indicator,
        x='full_name',
        y='variance',
        color='Status',
        color_discrete_map={'Under': '#ef553b', 'On Target': '#636EFA', 'Over': '#00cc96'},
        title='Variance Status - Top 20 Staff',
        labels={'variance': 'Variance (hours)', 'full_name': 'Staff Member'}
    )
    fig_var.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_var, use_container_width=True)


def render_clustering_analysis(df):
    """Tab 5: Clustering & Segmentation"""
    st.subheader("Staff Segmentation Analysis")
    
    # Prepare features for clustering
    df_cluster = df[['unified_total', 'total_shift_hours', 'total_leave_hours']].copy()
    df_cluster = df_cluster.fillna(df_cluster.mean())
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_cluster)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    df['cluster'] = clusters
    
    # Create cluster descriptions
    cluster_names = {
        0: "Cluster A",
        1: "Cluster B",
        2: "Cluster C",
        3: "Cluster D"
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    for i in range(4):
        with [col1, col2, col3, col4][i]:
            cluster_data = df[df['cluster'] == i]
            st.metric(
                f"Cluster {i}",
                f"{len(cluster_data)} staff",
                f"Avg: {cluster_data['unified_total'].mean():.1f}h"
            )
    
    st.subheader("Cluster Characteristics")
    cluster_summary = df.groupby('cluster').agg({
        'unified_total': ['count', 'mean', 'min', 'max'],
        'total_shift_hours': 'mean',
        'total_leave_hours': 'mean',
        'variance': 'mean'
    }).round(2)
    st.dataframe(cluster_summary, use_container_width=True)
    
    # 3D scatter plot
    fig = px.scatter_3d(
        df,
        x='total_shift_hours',
        y='total_leave_hours',
        z='unified_total',
        color='cluster',
        hover_data=['full_name'],
        title='Staff Clustering - 3D View',
        labels={
            'total_shift_hours': 'Shift Hours',
            'total_leave_hours': 'Leave Hours',
            'unified_total': 'Total Hours'
        }
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show cluster membership
    st.subheader("Cluster Membership")
    for i in range(4):
        with st.expander(f"📌 Cluster {i} ({len(df[df['cluster']==i])} members)"):
            cluster_members = df[df['cluster'] == i][['full_name', 'unified_total', 'variance']].sort_values('unified_total', ascending=False)
            st.dataframe(cluster_members, use_container_width=True, hide_index=True)


def render_cost_analysis(df, start_date, end_date):
    """Tab 6: Cost-Benefit Analysis"""
    st.subheader("Cost Analysis")
    
    # Calculate period length for prorating costs
    days_in_period = (end_date - start_date).days + 1
    
    col1, col2, col3 = st.columns(3)
    
    total_hours = df['unified_total'].sum()
    avg_hourly_cost = 18.5  # Estimated average cost per hour (can be configurable)
    total_payroll_cost = total_hours * avg_hourly_cost
    cost_per_shift_hour = total_payroll_cost / df['total_shift_hours'].sum() if df['total_shift_hours'].sum() > 0 else 0
    cost_per_leave_hour = total_payroll_cost * (df['total_leave_hours'].sum() / total_hours) if total_hours > 0 else 0
    
    with col1:
        st.metric("Total Hours", f"{total_hours:.1f}h")
    with col2:
        st.metric("Est. Period Cost", f"£{total_payroll_cost:.2f}")
    with col3:
        st.metric("Cost per Hour", f"£{avg_hourly_cost:.2f}")
    
    st.info("💡 Tip: Update estimated hourly cost in the code for your actual rate")
    
    # Cost breakdown pie chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            values=[df['total_shift_hours'].sum(), df['total_leave_hours'].sum()],
            names=['Shift Hours Cost', 'Leave Hours Cost'],
            title='Payroll Cost Breakdown'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Cost efficiency by person
        df_cost = df.copy()
        df_cost['estimated_cost'] = df_cost['unified_total'] * avg_hourly_cost
        
        fig_cost = px.scatter(
            df_cost,
            x='unified_total',
            y='estimated_cost',
            hover_data=['full_name'],
            title='Hours vs Estimated Cost',
            labels={'unified_total': 'Hours', 'estimated_cost': 'Est. Cost (£)'}
        )
        st.plotly_chart(fig_cost, use_container_width=True)
    
    st.subheader("Cost per Person")
    df_cost_summary = df.copy()
    df_cost_summary['est_cost'] = df_cost_summary['unified_total'] * avg_hourly_cost
    cost_df = df_cost_summary[['full_name', 'unified_total', 'est_cost']].sort_values('est_cost', ascending=False)
    st.dataframe(cost_df, use_container_width=True, hide_index=True)


def render_compliance_analysis(df):
    """Tab 7: Compliance & Risk Dashboard"""
    st.subheader("Compliance Analysis")
    
    # WTR (Working Time Regulations) - max 48h/week
    # For a typical period, max safe hours
    max_weekly_hours = 48
    max_period_hours = max_weekly_hours * ((df['expected_hours'].iloc[0] / 37.5) if not df['expected_hours'].isna().all() else 4)
    
    df['wtr_compliant'] = df['unified_total'] <= 240  # Assume ~5 week period, so 240 hours
    wtr_breaches = df[~df['wtr_compliant']]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("WTR Compliant", f"{df['wtr_compliant'].sum()}/{len(df)}")
    with col2:
        st.metric("Potential Breaches", len(wtr_breaches))
    with col3:
        st.metric("Compliance Rate", f"{(df['wtr_compliant'].sum()/len(df)*100):.1f}%")
    
    st.subheader("⚠️ Working Time Regulations Breaches")
    if len(wtr_breaches) > 0:
        st.warning(f"🚨 {len(wtr_breaches)} staff members may have exceeded 48h/week average")
        breach_df = wtr_breaches[['full_name', 'unified_total', 'variance']].sort_values('unified_total', ascending=False)
        st.dataframe(breach_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ All staff compliant with WTR")
    
    # Minimum hours check
    df['min_hours_met'] = df['unified_total'] >= (df['expected_hours'] * 0.8)
    min_breaches = df[~df['min_hours_met']]
    
    st.subheader("📋 Minimum Hours Policy (80% of Expected)")
    if len(min_breaches) > 0:
        st.warning(f"⚠️ {len(min_breaches)} staff fell below 80% of expected hours")
        min_df = min_breaches[['full_name', 'expected_hours', 'unified_total', 'variance']].sort_values('unified_total')
        st.dataframe(min_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ All staff met minimum hours requirement")
    
    # Leave accrual vs usage
    st.subheader("🏥 Leave Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Leave Hours", f"{df['total_leave_hours'].sum():.1f}h")
    with col2:
        st.metric("Avg Leave per Person", f"{df['total_leave_hours'].mean():.1f}h")
    
    leave_heavy = df[df['total_leave_hours'] > df['total_leave_hours'].mean() + df['total_leave_hours'].std()]
    if len(leave_heavy) > 0:
        st.info(f"📌 {len(leave_heavy)} staff with above-average leave usage")
        st.dataframe(leave_heavy[['full_name', 'total_leave_hours', 'total_shift_hours']], use_container_width=True, hide_index=True)


def render_benchmarking_analysis(df):
    """Tab 8: Benchmarking & Peer Comparison"""
    st.subheader("Benchmarking Analysis")
    
    # Percentile ranking
    df['hours_percentile'] = df['unified_total'].rank(pct=True) * 100
    df['variance_percentile'] = df['variance'].rank(pct=True) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Hours Percentile Distribution**")
        percentile_dist = pd.cut(df['hours_percentile'], bins=5, labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']).value_counts().sort_index()
        st.bar_chart(percentile_dist)
    
    with col2:
        st.write("**Variance Percentile Distribution**")
        variance_dist = pd.cut(df['variance_percentile'], bins=5, labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']).value_counts().sort_index()
        st.bar_chart(variance_dist)
    
    st.subheader("📊 Individual Rankings")
    ranking_df = df[['full_name', 'unified_total', 'hours_percentile', 'variance', 'variance_percentile']].copy()
    ranking_df.columns = ['Staff', 'Hours', 'Hours Percentile', 'Variance', 'Variance Percentile']
    ranking_df = ranking_df.sort_values('Hours', ascending=False)
    st.dataframe(ranking_df, use_container_width=True, hide_index=True)
    
    # Compare to cohort
    st.subheader("🎯 How You Compare")
    selected_staff = st.selectbox("Select staff member:", df['full_name'].values)
    
    if selected_staff:
        staff_data = df[df['full_name'] == selected_staff].iloc[0]
        cohort_mean = df['unified_total'].mean()
        cohort_std = df['unified_total'].std()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Your Hours", f"{staff_data['unified_total']:.1f}h")
        with col2:
            st.metric("Cohort Mean", f"{cohort_mean:.1f}h")
        with col3:
            diff = staff_data['unified_total'] - cohort_mean
            st.metric("Difference", f"{diff:+.1f}h")
        with col4:
            percentile = staff_data['hours_percentile']
            st.metric("Percentile Rank", f"{percentile:.0f}th")
        
        st.info(f"💡 {selected_staff} works {abs(diff):.1f} hours {'more' if diff > 0 else 'less'} than the average staff member and is in the {percentile:.0f}th percentile.")


def render_outlier_analysis(df):
    """Tab 9: Outlier & Risk Analysis"""
    st.subheader("Outlier Detection")
    
    # Statistical outlier detection using IQR
    Q1 = df['unified_total'].quantile(0.25)
    Q3 = df['unified_total'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['unified_total'] < lower_bound) | (df['unified_total'] > upper_bound)]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lower Bound", f"{lower_bound:.1f}h")
    with col2:
        st.metric("Upper Bound", f"{upper_bound:.1f}h")
    with col3:
        st.metric("Outliers Found", len(outliers))
    
    if len(outliers) > 0:
        st.warning(f"⚠️ Found {len(outliers)} statistical outliers")
        st.dataframe(outliers[['full_name', 'unified_total', 'variance']].sort_values('unified_total'), 
                    use_container_width=True, hide_index=True)
    else:
        st.success("✅ No statistical outliers detected")
    
    st.subheader("🎯 Risk Categories")
    
    # Define risk tiers
    df['risk_level'] = 'Low'
    df.loc[df['unified_total'] < Q1, 'risk_level'] = 'High'  # Underperforming
    df.loc[(df['unified_total'] > upper_bound), 'risk_level'] = 'Medium'  # Potential burnout
    df.loc[(df['variance'] < -10), 'risk_level'] = 'High'  # Significant underage
    
    risk_summary = df['risk_level'].value_counts()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 High Risk", risk_summary.get('High', 0))
    with col2:
        st.metric("🟡 Medium Risk", risk_summary.get('Medium', 0))
    with col3:
        st.metric("🟢 Low Risk", risk_summary.get('Low', 0))
    
    st.subheader("Risk Breakdown")
    for risk_level in ['High', 'Medium', 'Low']:
        risk_staff = df[df['risk_level'] == risk_level]
        if len(risk_staff) > 0:
            emoji = '🔴' if risk_level == 'High' else '🟡' if risk_level == 'Medium' else '🟢'
            with st.expander(f"{emoji} {risk_level} Risk ({len(risk_staff)} staff)"):
                st.dataframe(risk_staff[['full_name', 'unified_total', 'variance']].sort_values('unified_total'),
                           use_container_width=True, hide_index=True)


# =====================================================================
# MAIN REPORT FUNCTIONS (from original enhanced script)
# =====================================================================

def generate_pdf_report(user_reports, start_date, end_date, use_threshold_filter=False, 
                       low_threshold=None, high_threshold=None):
    """Generate a styled PDF report with variance analysis"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter),
                           rightMargin=0.75*inch, leftMargin=0.75*inch,
                           topMargin=1*inch, bottomMargin=0.75*inch)
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#111827'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.HexColor('#4B5563'),
        spaceAfter=20,
        alignment=TA_CENTER
    )
    
    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=12,
        spaceBefore=16,
        fontName='Helvetica-Bold'
    )
    
    user_name_style = ParagraphStyle(
        'UserName',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#111827'),
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#374151'),
        spaceAfter=6
    )
    
    story = []
    
    story.append(Paragraph("📊 Shift & Leave Report with Variance Analysis", title_style))
    
    date_range_text = f"Period: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}"
    story.append(Paragraph(date_range_text, subtitle_style))
    
    if user_reports:
        gender_summary = {}
        for rpt in user_reports:
            gender_display = get_gender_display_name(rpt["gender"])
            gender_summary[gender_display] = gender_summary.get(gender_display, 0) + 1
        
        summary_text = f"<b>Total Users:</b> {len(user_reports)} | " + \
                      ", ".join([f"{count} {gender}" for gender, count in gender_summary.items()])
        story.append(Paragraph(summary_text, normal_style))
        story.append(Spacer(1, 0.2*inch))
    
    table_header_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F9FAFB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#111827')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
    ]
    
    table_data_style = [
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#374151')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')]),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]
    
    groups = [("All Users", user_reports, None)]
    
    for group_name, group_reports, bg_color in groups:
        if not group_reports:
            continue
            
        story.append(Paragraph(f"{group_name} ({len(group_reports)} users)", section_header_style))
        
        summary_data = [['Name', 'ID', 'Weekly Hrs', 'Expected', 'Actual', 'Variance']]
        
        for rpt in group_reports:
            fname = rpt["first_name"]
            lname = rpt["last_name"]
            uid = rpt["user_id"]
            
            expected = rpt.get("expected_hours")
            actual = rpt["unified_total"]
            weekly = rpt.get("weekly_hours", 0)
            
            if expected is not None:
                variance = actual - expected
                variance_str = f"{variance:+.2f}"
                if variance < 0:
                    variance_indicator = f"🔴 {variance_str}"
                elif variance > 0:
                    variance_indicator = f"🟢 {variance_str}"
                else:
                    variance_indicator = f"🟡 {variance_str}"
            else:
                expected = "N/A"
                variance_indicator = "N/A"
            
            summary_data.append([
                f"{fname} {lname}",
                str(uid),
                f"{weekly:.1f}" if weekly else "N/A",
                f"{expected:.2f}" if isinstance(expected, float) else str(expected),
                f"{actual:.2f}",
                variance_indicator
            ])
        
        col_widths = [2*inch, 0.7*inch, 1*inch, 1*inch, 1*inch, 1.3*inch]
        summary_table = Table(summary_data, colWidths=col_widths)
        
        summary_table.setStyle(TableStyle(table_header_style + table_data_style))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


# =====================================================================
# MAIN APP
# =====================================================================

if 'current_period_year' not in st.session_state:
    current_start, current_end = get_current_period()
    st.session_state.current_period_year = current_start.year
    st.session_state.current_period_month = current_start.month

if 'mobile_view' not in st.session_state:
    st.session_state.mobile_view = False

if 'generated_report' not in st.session_state:
    st.session_state.generated_report = None

if 'enable_analytics' not in st.session_state:
    st.session_state.enable_analytics = False

st.title("📊 Shift & Leave Report with Advanced Analytics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🔄 Refresh", use_container_width=True, help="Clear cache and refresh"):
        st.cache_data.clear()
        st.session_state.generated_report = None
        st.rerun()
with col2:
    if st.button("⚙️ Settings", use_container_width=True, help="Open settings sidebar"):
        st.session_state.sidebar_state = 'expanded'
        st.rerun()
with col3:
    if st.button("📱 Mobile" if not st.session_state.mobile_view else "💻 Desktop", 
                 use_container_width=True, help="Toggle mobile/desktop view"):
        st.session_state.mobile_view = not st.session_state.mobile_view
        st.rerun()
with col4:
    quick_generate = st.button("⚡ Quick Report", use_container_width=True, 
                               help="Generate with current settings")

st.sidebar.header("⚙️ Configuration")

# ANALYTICS TOGGLE - PROMINENT AT TOP
st.sidebar.markdown("### 📊 Analytics Module")
enable_analytics = st.sidebar.checkbox(
    "Enable Advanced Analytics",
    value=st.session_state.enable_analytics,
    help="Toggle advanced data visualizations (9 analysis tabs with clustering, compliance, cost analysis, etc.)"
)
st.session_state.enable_analytics = enable_analytics

if enable_analytics:
    st.sidebar.success("✅ Analytics enabled - will show after report generation")
else:
    st.sidebar.info("💡 Enable above to add advanced analytics dashboard")

st.sidebar.markdown("---")

with st.sidebar.expander("📅 Date Selection", expanded=True):
    manual_dates = st.checkbox("Manual Date Selection", value=False, 
                               help="Toggle to manually select dates")

    if manual_dates:
        default_start = datetime.date(2025, 6, 11)
        default_end = datetime.date(2025, 7, 10)
        start_date = st.date_input("Start Date", value=default_start)
        end_date = st.date_input("End Date", value=default_end)
        if start_date > end_date:
            st.error("Start date must be before end date.")
    else:
        current_start, current_end = get_monthly_period(
            st.session_state.current_period_year, 
            st.session_state.current_period_month
        )
        
        period_name = current_start.strftime('%b %Y')
        date_range = f"{current_start.strftime('%d/%m')} - {current_end.strftime('%d/%m')}"
        st.info(f"**{period_name}**\n{date_range}")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("⬅️", help="Previous month", use_container_width=True):
                if st.session_state.current_period_month == 1:
                    st.session_state.current_period_month = 12
                    st.session_state.current_period_year -= 1
                else:
                    st.session_state.current_period_month -= 1
                st.rerun()
        
        with col2:
            if st.button("📅", help="Current period", use_container_width=True):
                current_start, current_end = get_current_period()
                st.session_state.current_period_year = current_start.year
                st.session_state.current_period_month = current_start.month
                st.rerun()
        
        with col3:
            if st.button("➡️", help="Next month", use_container_width=True):
                if st.session_state.current_period_month == 12:
                    st.session_state.current_period_month = 1
                    st.session_state.current_period_year += 1
                else:
                    st.session_state.current_period_month += 1
                st.rerun()
        
        start_date, end_date = get_monthly_period(
            st.session_state.current_period_year, 
            st.session_state.current_period_month
        )

with st.sidebar.expander("🔍 Filters"):
    enable_gender_filter = st.checkbox("Enable Gender Filter", value=False)
    
    if enable_gender_filter:
        gender_options = st.multiselect(
            "Select Genders",
            options=['Male', 'Female', 'Terminal', 'Unknown'],
            default=['Male', 'Female', 'Terminal', 'Unknown']
        )
        selected_gender_codes = []
        for option in gender_options:
            if option == 'Male':
                selected_gender_codes.append('M')
            elif option == 'Female':
                selected_gender_codes.append('F')
            elif option == 'Terminal':
                selected_gender_codes.append('T')
            elif option == 'Unknown':
                selected_gender_codes.append('Unknown')
    else:
        selected_gender_codes = ['M', 'F', 'T', 'Unknown']
    
    enable_variance_filter = st.checkbox("Filter by Variance Status", value=False,
                                        help="Show only over/under/on-target staff")
    
    if enable_variance_filter:
        variance_options = st.multiselect(
            "Variance Status",
            options=['Under-worked 🔴', 'On-target 🟡', 'Over-worked 🟢'],
            default=['Under-worked 🔴', 'On-target 🟡', 'Over-worked 🟢']
        )
    else:
        variance_options = ['Under-worked 🔴', 'On-target 🟡', 'Over-worked 🟢']
    
    use_threshold_filter = st.checkbox("Enable Threshold Filter", value=False)
    
    if use_threshold_filter:
        col1, col2 = st.columns(2)
        with col1:
            low_threshold = st.number_input("Min Hours", value=156, step=1)
        with col2:
            high_threshold = st.number_input("Max Hours", value=170, step=1)
        if low_threshold > high_threshold:
            st.error("Min must be ≤ Max")
    else:
        low_threshold = None
        high_threshold = None

with st.sidebar.expander("⚙️ Settings"):
    sort_option = st.radio(
        "Sort by:",
        ("Alphabetical", "Highest Hours", "Lowest Hours", "Most Under-worked", "Most Over-worked"),
        index=1,
        horizontal=True
    )
    
    sort_mapping = {
        "Alphabetical": "Alphabetical (First Name)",
        "Highest Hours": "Highest to Lowest",
        "Lowest Hours": "Lowest to Highest",
        "Most Under-worked": "Most Under-worked",
        "Most Over-worked": "Most Over-worked"
    }
    sort_option = sort_mapping.get(sort_option, sort_option)
    
    ignored_input = st.text_input(
        "Ignored IDs", 
        value=','.join(map(str, DEFAULT_IGNORED_IDS)),
        help="Comma-separated user IDs to ignore"
    )

# ANALYTICS TOGGLE - PROMINENT AT TOP (moved to top of sidebar)

api_key_input = st.sidebar.text_input(
    "🔑 API Key", 
    type="password",
    help="Enter your API key"
)

ignored_user_ids = set()
for part in ignored_input.split(","):
    part = part.strip()
    if part.isdigit():
        ignored_user_ids.add(int(part))

generate = st.sidebar.button("📊 Generate Report", use_container_width=True, type="primary")

if quick_generate:
    generate = True

if not manual_dates:
    period_display = format_period_name(start_date, end_date)
    st.info(f"📅 **Period**: {period_display}")
else:
    st.info(f"📅 **Period**: {start_date.strftime('%d/%m/%y')} - {end_date.strftime('%d/%m/%y')}")

active_filters = []
if enable_gender_filter:
    active_filters.append(f"Gender: {', '.join(gender_options)}")
if enable_variance_filter:
    active_filters.append(f"Variance: {', '.join(variance_options)}")
if use_threshold_filter:
    active_filters.append(f"Hours: {low_threshold}-{high_threshold}")
if ignored_user_ids:
    active_filters.append(f"Ignored: {len(ignored_user_ids)} users")

if active_filters:
    st.info(f"🔍 **Filters**: {' | '.join(active_filters)}")

API_KEY = None

if api_key_input:
    API_KEY = api_key_input.strip()
elif os.environ.get("API_KEY"):
    API_KEY = os.environ.get("API_KEY")

if not API_KEY:
    st.error("⚠️ **API Key Required**")
    st.write("Enter your API key in the sidebar to continue.")
    st.stop()

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

BASE_API_URL = st.secrets.get("rc_base_url", "https://api.example.com")
USERS_BASE_URL    = f"{BASE_API_URL}/v1/users"
SHIFTS_BASE_URL   = f"{BASE_API_URL}/v1/shifts"  
LOCATIONS_BASE_URL= f"{BASE_API_URL}/v1/locations"
ROLES_BASE_URL    = f"{BASE_API_URL}/v1/roles"
LEAVE_BASE_URL    = f"{BASE_API_URL}/v1/leave"
CONTRACTS_BASE_URL= f"{BASE_API_URL}/v1/contracts"

def date_to_unix_timestamp(date_obj: datetime.date, hour=0, minute=0, second=0, timezone_str='Europe/London'):
    try:
        local_tz = pytz.timezone(timezone_str)
        dt_naive = datetime.datetime.combine(date_obj, datetime.time(hour, minute, second))
        local_dt = local_tz.localize(dt_naive, is_dst=None)
        utc_dt = local_dt.astimezone(pytz.utc)
        return int(utc_dt.timestamp())
    except Exception as e:
        st.warning(f"Error converting date '{date_obj}' to timestamp: {e}")
        return None

def date_str_to_date(date_str: str):
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except:
        return None

def test_api_connection():
    try:
        response = requests.get(USERS_BASE_URL, headers=HEADERS, params={"limit": 1})
        if response.status_code == 401:
            return False, "Invalid API key or unauthorized access"
        elif response.status_code == 403:
            return False, "API key valid but insufficient permissions"
        elif response.status_code == 429:
            return False, "Rate limit exceeded"
        elif response.status_code >= 500:
            return False, f"Server error: {response.status_code}"
        else:
            response.raise_for_status()
            return True, "Connection successful"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

@cache_data(ttl=300)
def get_rc_users():
    try:
        resp = requests.get(USERS_BASE_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 401:
            st.error("🔒 Invalid API key")
        elif resp.status_code == 403:
            st.error("🚫 Insufficient permissions")
        elif resp.status_code == 429:
            st.error("⏰ Rate limit exceeded")
        else:
            st.error(f"Error {resp.status_code}")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timeout")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error")
        return None

@cache_data(ttl=300)
def get_rc_shifts(start_ts, end_ts, user_id):
    params = {
        "start": start_ts,
        "end": end_ts,
        "published": "true"
    }
    params["users[]"] = [user_id]
    try:
        resp = requests.get(SHIFTS_BASE_URL, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except:
        return None

@cache_data(ttl=300)
def get_rc_leave(start_str, end_str, user_id):
    params = {
        "start": start_str,
        "end": end_str,
        "include_deleted": "false",
        "include_denied": "false",
        "include_requested": "false",
        "include_expired": "true"
    }
    params["users[]"] = [user_id]
    try:
        resp = requests.get(LEAVE_BASE_URL, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except:
        return None

@cache_data(ttl=600)
def get_user_weekly_hours(user_id):
    """Fetch user's contract information including weekly hours"""
    try:
        resp = requests.get(f"{USERS_BASE_URL}/{user_id}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        user_data = resp.json()
        
        if "weekly_hours" in user_data:
            return float(user_data.get("weekly_hours", 0)) or None
        
        if "contract_id" in user_data or "contract" in user_data:
            contract_id = user_data.get("contract_id") or user_data.get("contract", {}).get("id")
            if contract_id:
                contract_resp = requests.get(f"{CONTRACTS_BASE_URL}/{contract_id}", 
                                            headers=HEADERS, timeout=30)
                if contract_resp.status_code == 200:
                    contract_data = contract_resp.json()
                    return float(contract_data.get("weekly_hours", 0)) or None
        
        return None
    except:
        return None

@cache_data(ttl=600)
def get_location_name(location_id):
    if location_id is None:
        return "N/A"
    try:
        resp = requests.get(f"{LOCATIONS_BASE_URL}/{location_id}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("name", f"Loc {location_id}")
    except:
        return f"Loc {location_id}"

@cache_data(ttl=600)
def get_role_name(role_id):
    if role_id is None:
        return "N/A"
    try:
        resp = requests.get(f"{ROLES_BASE_URL}/{role_id}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("name", f"Role {role_id}")
    except:
        return f"Role {role_id}"

def process_user_shifts(shifts_json, report_start_date, report_end_date):
    total_seconds = 0
    processed = []
    if not shifts_json:
        return 0.0, []
    for shift in shifts_json:
        st_unix = shift.get("start_time")
        en_unix = shift.get("end_time")
        minutes_break = shift.get("minutes_break", 0) or 0
        if st_unix is None or en_unix is None:
            continue
        duration = en_unix - st_unix
        net = duration - minutes_break * 60
        total_seconds += net
        processed.append({
            "shift_id": shift.get("id"),
            "start_time_unix": st_unix,
            "location_id": shift.get("location"),
            "role_id": shift.get("role"),
            "net_work_hours": round(net/3600, 2)
        })
    total_hours = round(total_seconds/3600, 2)
    return total_hours, processed

def process_user_leave(leave_json, report_start_date, report_end_date):
    total_days = 0.0
    total_hours = 0.0
    processed = []
    if not leave_json:
        return 0.0, 0.0, []
    for record in leave_json:
        if record.get("status") != "approved":
            continue
        days_sum = 0.0
        hours_sum = 0.0
        for date_entry in record.get("dates", []):
            dstr = date_entry.get("date")
            ddate = date_str_to_date(dstr)
            if not ddate:
                continue
            if report_start_date <= ddate <= report_end_date:
                days_val = date_entry.get("days", 0) or 0
                hours_val = date_entry.get("hours", 0) or 0
                days_sum += days_val
                hours_sum += hours_val
        if days_sum > 0 or hours_sum > 0:
            total_days += days_sum
            total_hours += hours_sum
            processed.append({
                "type": record.get("type"),
                "status": record.get("status"),
                "days_in_report_period": days_sum,
                "hours_in_report_period": hours_sum
            })
    total_days = round(total_days, 1)
    total_hours = round(total_hours, 1)
    return total_days, total_hours, processed

def weekday_and_ddmmyy(unix_ts):
    dt_utc = datetime.datetime.fromtimestamp(unix_ts, tz=pytz.utc)
    local_tz = pytz.timezone('Europe/London')
    dt_local = dt_utc.astimezone(local_tz)
    day_name = dt_local.strftime('%A')
    date_str = dt_local.strftime('%d/%m/%y')
    return day_name, date_str

def sort_user_reports(user_reports, sort_option):
    if sort_option == "Alphabetical (First Name)":
        return sorted(user_reports, key=lambda r: (r.get("first_name", "").lower(), r.get("last_name", "").lower()))
    elif sort_option == "Highest to Lowest":
        return sorted(user_reports, key=lambda r: r["unified_total"], reverse=True)
    elif sort_option == "Lowest to Highest":
        return sorted(user_reports, key=lambda r: r["unified_total"])
    elif sort_option == "Most Under-worked":
        # Sort by variance ascending (most negative first), None values last
        return sorted(user_reports, key=lambda r: (r.get("variance") is None, r.get("variance", 0)))
    elif sort_option == "Most Over-worked":
        # Sort by variance descending (most positive first), None values last
        return sorted(user_reports, key=lambda r: (r.get("variance") is None, r.get("variance", 0)), reverse=True)
    else:
        return user_reports

def filter_by_gender(user_list, selected_gender_codes):
    filtered = []
    for user in user_list:
        user_gender = get_user_gender(user["id"])
        if user_gender in selected_gender_codes:
            user["gender"] = user_gender
            filtered.append(user)
    return filtered

def filter_by_variance(user_reports, variance_options):
    """Filter users based on variance status"""
    filtered = []
    for rpt in user_reports:
        expected = rpt.get("expected_hours")
        actual = rpt["unified_total"]
        
        if expected is None:
            continue
        
        if actual < expected and 'Under-worked 🔴' in variance_options:
            filtered.append(rpt)
        elif actual > expected and 'Over-worked 🟢' in variance_options:
            filtered.append(rpt)
        elif actual == expected and 'On-target 🟡' in variance_options:
            filtered.append(rpt)
    
    return filtered

def display_user(rpt, show_variance_indicator=True):
    fname = rpt["first_name"]
    lname = rpt["last_name"]
    uid = rpt["user_id"]
    gender = rpt["gender"]
    gender_display = get_gender_display_name(gender)
    
    st.subheader(f"{fname} {lname} ({gender_display}) (ID: {uid})")

    ut = rpt["unified_total"]
    expected = rpt.get("expected_hours")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Hours", f"{ut:.2f}")
    with col2:
        st.metric("Shift Hours", f"{rpt['total_shift_hours']:.2f}")
    with col3:
        st.metric("Leave Hours", f"{rpt['total_leave_hours']:.2f}")
    
    if show_variance_indicator and expected is not None:
        st.markdown("---")
        variance = ut - expected
        variance_pct = (variance / expected * 100) if expected > 0 else 0
        variance_color = get_variance_color(ut, expected)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Weekly Contract", f"{rpt.get('weekly_hours', 'N/A')}")
        with col2:
            st.metric("Expected Hours", f"{expected:.2f}")
        with col3:
            st.markdown(f"<span style='color:{variance_color};'><b>Variance</b><br>{variance:+.2f}h ({variance_pct:+.1f}%)</span>", 
                       unsafe_allow_html=True)
        with col4:
            emoji = get_variance_emoji(ut, expected)
            st.metric("Status", emoji.split('(')[0].strip())

    st.markdown("---")

if st.sidebar.button("🔍 Test Connection"):
    with st.spinner("Testing..."):
        success, message = test_api_connection()
        if success:
            st.sidebar.success(f"✅ {message}")
        else:
            st.sidebar.error(f"❌ {message}")

if generate:
    if start_date > end_date:
        st.error("Invalid date range.")
        st.stop()

    with st.spinner("Testing connection..."):
        success, message = test_api_connection()
        if not success:
            st.error(f"❌ {message}")
            st.stop()

    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")
    start_ts = date_to_unix_timestamp(start_date, hour=0, minute=0, second=0)
    end_ts   = date_to_unix_timestamp(end_date, hour=23, minute=59, second=59)

    with st.spinner("Fetching users..."):
        all_users = get_rc_users()
        if all_users is None:
            st.error("Could not retrieve users.")
            st.stop()

    refined = []
    for u in all_users:
        uid = u.get("id")
        if uid is None or uid in ignored_user_ids:
            continue
        refined.append({
            "id": uid,
            "first_name": u.get("first_name") or "",
            "last_name": u.get("last_name") or ""
        })

    if enable_gender_filter:
        refined = filter_by_gender(refined, selected_gender_codes)

    user_reports = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_users = len(refined)
    
    for idx, u in enumerate(refined):
        uid = u["id"]
        fname = u["first_name"]
        lname = u["last_name"]
        gender = u.get("gender", get_user_gender(uid))

        progress_bar.progress((idx+1)/total_users)
        status_text.text(f"Processing {fname} {lname} ({idx+1}/{total_users})")

        weekly_hours = get_user_weekly_hours(uid)

        shifts_json = get_rc_shifts(start_ts, end_ts, uid)
        total_shift_hours, processed_shifts = process_user_shifts(shifts_json, start_date, end_date)

        leave_json = get_rc_leave(start_str, end_str, uid)
        total_leave_days, total_leave_hours, processed_leave_records = process_user_leave(leave_json, start_date, end_date)

        unified_total = total_shift_hours + total_leave_hours
        
        expected_hours = calculate_expected_hours(weekly_hours, start_date, end_date)
        variance = (unified_total - expected_hours) if expected_hours else None

        user_reports.append({
            "user_id": uid,
            "first_name": fname,
            "last_name": lname,
            "gender": gender,
            "weekly_hours": weekly_hours,
            "expected_hours": expected_hours,
            "total_shift_hours": total_shift_hours,
            "processed_shifts": processed_shifts,
            "total_leave_days": total_leave_days,
            "total_leave_hours": total_leave_hours,
            "processed_leave_records": processed_leave_records,
            "unified_total": unified_total,
            "variance": variance
        })

    progress_bar.empty()
    status_text.empty()

    if enable_variance_filter:
        user_reports = filter_by_variance(user_reports, variance_options)

    user_reports = sort_user_reports(user_reports, sort_option)
    
    st.session_state.generated_report = {
        'user_reports': user_reports,
        'start_date': start_date,
        'end_date': end_date,
        'use_threshold_filter': use_threshold_filter,
        'low_threshold': low_threshold,
        'high_threshold': high_threshold
    }

    st.write("## 📋 Report Results")
    
    if user_reports:
        col1, col2 = st.columns([3, 1])
        with col2:
            with st.spinner("Generating PDF..."):
                pdf_buffer = generate_pdf_report(
                    user_reports, 
                    start_date, 
                    end_date,
                    use_threshold_filter,
                    low_threshold,
                    high_threshold
                )
                
                filename = f"shift_variance_report_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.pdf"
                st.download_button(
                    label="📄 Export PDF",
                    data=pdf_buffer,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
    
    if user_reports:
        gender_summary = {}
        for rpt in user_reports:
            gender_display = get_gender_display_name(rpt["gender"])
            gender_summary[gender_display] = gender_summary.get(gender_display, 0) + 1
        
        summary_text = ", ".join([f"{count} {gender}" for gender, count in gender_summary.items()])
        
        under_worked = len([r for r in user_reports if r.get("variance") and r["variance"] < 0])
        over_worked = len([r for r in user_reports if r.get("variance") and r["variance"] > 0])
        on_target = len([r for r in user_reports if r.get("variance") == 0])
        
        st.success(f"👥 {len(user_reports)} users | {summary_text} | 🔴 {under_worked} under | 🟢 {over_worked} over | 🟡 {on_target} on-target")
    
    # Display main report
    st.write(f"### All Users ({len(user_reports)})")
    for rpt in user_reports[:10]:  # Show first 10 to keep main report concise
        display_user(rpt, show_variance_indicator=True)
    
    if len(user_reports) > 10:
        st.info(f"Showing first 10 of {len(user_reports)} users. Scroll down or use Analytics for full view.")
    
    # RENDER ANALYTICS MODULE IF ENABLED
    if enable_analytics and user_reports:
        render_analytics_module(user_reports, start_date, end_date)
    
    st.success("✅ Report complete!")

else:
    st.write("### 👋 Welcome!")
    st.write("Configure settings in the sidebar and click **Generate Report** to begin.")
    
    with st.expander("✨ Features"):
        st.write("**Main Report:**")
        st.write("• Variance analysis with color-coded status indicators")
        st.write("• Weekly contract hours integration")
        st.write("• Expected vs actual hours comparison")
        
        st.write("\n**Advanced Analytics (Toggle in Sidebar):**")
        st.write("• 📊 Distribution analysis with histograms and box plots")
        st.write("• 📍 Location-based performance comparison")
        st.write("• 👥 Demographic equity analysis")
        st.write("• 🔥 Staff utilization heatmaps")
        st.write("• 🎯 K-means clustering (4-segment staff profiles)")
        st.write("• 💷 Cost-benefit analysis by person")
        st.write("• 📈 Compliance dashboard (WTR, minimum hours, leave)")
        st.write("• 🔮 Peer benchmarking & percentile rankings")
        st.write("• ⚠️ Outlier detection & risk categorization")
    
    with st.expander("⚡ Quick Start"):
        st.write("1. Enter your API key in the sidebar")
        st.write("2. Toggle **Enable Advanced Analytics** if you want data visualizations")
        st.write("3. Select your date range or use period navigation")
        st.write("4. Configure filters as needed")
        st.write("5. Click **Generate Report**")
        st.write("6. View main report, then scroll for analytics if enabled")
