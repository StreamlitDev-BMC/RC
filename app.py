import os
import requests
import datetime
import pytz
import pandas as pd
import streamlit as st
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

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

# PDF Generation Function
def generate_pdf_report(user_reports, start_date, end_date, use_threshold_filter=False, 
                       low_threshold=None, high_threshold=None):
    """Generate a styled PDF report with Tailwind-inspired design"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=0.75*inch, leftMargin=0.75*inch,
                           topMargin=1*inch, bottomMargin=0.75*inch)
    
    # Custom styles inspired by Tailwind CSS
    styles = getSampleStyleSheet()
    
    # Title style (text-4xl font-bold text-gray-900)
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#111827'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    # Subtitle style (text-lg text-gray-600)
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.HexColor('#4B5563'),
        spaceAfter=20,
        alignment=TA_CENTER
    )
    
    # Section header style (text-xl font-semibold text-gray-800)
    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=12,
        spaceBefore=16,
        fontName='Helvetica-Bold'
    )
    
    # User name style (text-base font-semibold text-gray-900)
    user_name_style = ParagraphStyle(
        'UserName',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#111827'),
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    # Normal text style (text-sm text-gray-700)
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#374151'),
        spaceAfter=6
    )
    
    story = []
    
    # Title
    story.append(Paragraph("📊 Shift & Leave Report", title_style))
    
    # Date range
    date_range_text = f"Period: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}"
    story.append(Paragraph(date_range_text, subtitle_style))
    
    # Summary section
    if user_reports:
        gender_summary = {}
        for rpt in user_reports:
            gender_display = get_gender_display_name(rpt["gender"])
            gender_summary[gender_display] = gender_summary.get(gender_display, 0) + 1
        
        summary_text = f"<b>Total Users:</b> {len(user_reports)} | " + \
                      ", ".join([f"{count} {gender}" for gender, count in gender_summary.items()])
        story.append(Paragraph(summary_text, normal_style))
        story.append(Spacer(1, 0.2*inch))
    
    # Table header style (bg-gray-50 border-b-2 border-gray-300)
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
    
    # Table data style (text-sm text-gray-700)
    table_data_style = [
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#374151')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')]),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]
    
    # Group users by threshold if enabled
    if use_threshold_filter and low_threshold and high_threshold:
        below = [rpt for rpt in user_reports if rpt["unified_total"] < low_threshold]
        above = [rpt for rpt in user_reports if rpt["unified_total"] > high_threshold]
        between = [rpt for rpt in user_reports if low_threshold <= rpt["unified_total"] <= high_threshold]
        
        groups = [
            (f"🔴 Below {low_threshold} hours", below, colors.HexColor('#FEE2E2')),
            (f"🟢 {low_threshold}-{high_threshold} hours", between, colors.HexColor('#D1FAE5')),
            (f"🟠 Above {high_threshold} hours", above, colors.HexColor('#FED7AA')),
        ]
    else:
        groups = [("All Users", user_reports, None)]
    
    for group_name, group_reports, bg_color in groups:
        if not group_reports:
            continue
            
        # Section header
        story.append(Paragraph(f"{group_name} ({len(group_reports)} users)", section_header_style))
        
        # Create summary table
        summary_data = [['Name', 'ID', 'Gender', 'Shift Hours', 'Leave Hours', 'Total Hours']]
        
        for rpt in group_reports:
            fname = rpt["first_name"]
            lname = rpt["last_name"]
            uid = rpt["user_id"]
            gender_display = get_gender_display_name(rpt["gender"])
            
            summary_data.append([
                f"{fname} {lname}",
                str(uid),
                gender_display,
                f"{rpt['total_shift_hours']:.2f}",
                f"{rpt['total_leave_hours']:.2f}",
                f"{rpt['unified_total']:.2f}"
            ])
        
        # Create table
        col_widths = [2.2*inch, 0.8*inch, 0.8*inch, 1*inch, 1*inch, 1*inch]
        summary_table = Table(summary_data, colWidths=col_widths)
        
        # Apply styles
        summary_table.setStyle(TableStyle(table_header_style + table_data_style))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Detailed breakdown for each user
        for rpt in group_reports:
            fname = rpt["first_name"]
            lname = rpt["last_name"]
            uid = rpt["user_id"]
            gender_display = get_gender_display_name(rpt["gender"])
            
            # User header
            user_header = f"<b>{fname} {lname}</b> ({gender_display}) - ID: {uid}"
            story.append(Paragraph(user_header, user_name_style))
            
            # User stats
            stats_text = f"Total Hours: <b>{rpt['unified_total']:.2f}</b> | " + \
                        f"Shift Hours: {rpt['total_shift_hours']:.2f} | " + \
                        f"Leave Hours: {rpt['total_leave_hours']:.2f}"
            story.append(Paragraph(stats_text, normal_style))
            
            # Shifts table
            if rpt["processed_shifts"]:
                shift_data = [['Day', 'Date', 'Location', 'Role', 'Hours']]
                for s in rpt["processed_shifts"]:
                    day_name = datetime.datetime.fromtimestamp(s["start_time_unix"]).strftime('%A')
                    date_str = datetime.datetime.fromtimestamp(s["start_time_unix"]).strftime('%d/%m/%y')
                    loc = s.get("location_name", "Unknown")
                    role = s.get("role_name", "Unknown")
                    shift_data.append([
                        day_name[:3],
                        date_str,
                        loc,
                        role,
                        f"{s.get('net_work_hours', 0):.2f}"
                    ])
                
                shift_table = Table(shift_data, colWidths=[0.7*inch, 0.9*inch, 1.8*inch, 1.8*inch, 0.7*inch])
                shift_table.setStyle(TableStyle(table_header_style + table_data_style))
                story.append(shift_table)
                story.append(Spacer(1, 0.1*inch))
            
            # Leave table
            if rpt["total_leave_hours"] > 0 and rpt["processed_leave_records"]:
                leave_data = [['Type', 'Status', 'Days', 'Hours']]
                for rec in rpt["processed_leave_records"]:
                    leave_data.append([
                        rec.get("type", "Unknown"),
                        rec.get("status", "Unknown"),
                        f"{rec.get('days_in_report_period', 0):.1f}",
                        f"{rec.get('hours_in_report_period', 0):.2f}"
                    ])
                
                leave_table = Table(leave_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1*inch])
                leave_table.setStyle(TableStyle(table_header_style + table_data_style))
                story.append(leave_table)
            
            story.append(Spacer(1, 0.2*inch))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

if 'current_period_year' not in st.session_state:
    current_start, current_end = get_current_period()
    st.session_state.current_period_year = current_start.year
    st.session_state.current_period_month = current_start.month

if 'mobile_view' not in st.session_state:
    st.session_state.mobile_view = False

if 'generated_report' not in st.session_state:
    st.session_state.generated_report = None

st.title("📊 Shift & Leave Report")

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
        gender_options = {
            'Male': 'M',
            'Female': 'F',
            'Terminal': 'T',
            'Unknown': 'Unknown'
        }
        selected_genders = st.multiselect(
            "Select Genders",
            options=list(gender_options.keys()),
            default=list(gender_options.keys())
        )
        selected_gender_codes = [gender_options[g] for g in selected_genders]
    else:
        selected_gender_codes = ['M', 'F', 'T', 'Unknown']
    
    # UPDATED DEFAULT: Threshold filter disabled by default
    use_threshold_filter = st.checkbox("Enable Threshold Filter", value=False,
                                      help="Filter users by total hours")
    
    if use_threshold_filter:
        col1, col2 = st.columns(2)
        with col1:
            low_threshold = st.number_input("Low Threshold", min_value=0.0, value=120.0, step=10.0)
        with col2:
            high_threshold = st.number_input("High Threshold", min_value=0.0, value=160.0, step=10.0)
        
        show_threshold_color = st.checkbox("Color Code by Threshold", value=True)
    else:
        low_threshold = None
        high_threshold = None
        show_threshold_color = False

with st.sidebar.expander("🔧 Advanced Settings"):
    # UPDATED DEFAULT: Sort by highest hours by default
    sort_option = st.selectbox(
        "Sort By",
        options=["Highest Hours", "Lowest Hours", "Name (A-Z)", "Name (Z-A)", "User ID"],
        index=0,  # Changed from previous default to "Highest Hours"
        help="Sort order for report results"
    )
    
    ignored_user_ids_input = st.text_area(
        "Ignored User IDs (comma-separated)",
        value=",".join(map(str, DEFAULT_IGNORED_IDS)),
        help="User IDs to exclude from the report"
    )
    
    try:
        ignored_user_ids = [int(x.strip()) for x in ignored_user_ids_input.split(",") if x.strip()]
    except ValueError:
        st.error("Invalid user IDs. Please enter comma-separated numbers.")
        ignored_user_ids = DEFAULT_IGNORED_IDS

st.sidebar.header("🔐 API Configuration")
api_key = st.sidebar.text_input("API Key", type="password", value=st.secrets.get("rc_api_key", ""))
base_url = st.sidebar.text_input("Base URL", value=st.secrets.get("rc_base_url", ""))

generate = st.sidebar.button("🔍 Generate Report", use_container_width=True, type="primary") or quick_generate

# Helper functions (keeping the original ones from the file)
@cache_data(ttl=600)
def test_api_connection():
    if not api_key or not base_url:
        return False, "API key or base URL not configured"
    
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(f"{base_url}/users", headers=headers, timeout=10)
        if response.status_code == 200:
            return True, "Connection successful"
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def date_to_unix_timestamp(date_obj, hour=0, minute=0, second=0):
    dt = datetime.datetime.combine(date_obj, datetime.time(hour=hour, minute=minute, second=second))
    london_tz = pytz.timezone('Europe/London')
    dt = london_tz.localize(dt)
    return int(dt.timestamp())

@cache_data(ttl=600)
def get_rc_users():
    if not api_key or not base_url:
        return None
    
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(f"{base_url}/users", headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None

@cache_data(ttl=600)
def get_rc_shifts(start_ts, end_ts, user_id):
    if not api_key or not base_url:
        return None
    
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{base_url}/shifts?start={start_ts}&end={end_ts}&user_id={user_id}"
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None

@cache_data(ttl=600)
def get_rc_leave(start_str, end_str, user_id):
    if not api_key or not base_url:
        return None
    
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{base_url}/leave?start={start_str}&end={end_str}&user_id={user_id}"
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None

@cache_data(ttl=3600)
def get_rc_locations():
    if not api_key or not base_url:
        return {}
    
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(f"{base_url}/locations", headers=headers, timeout=30)
        if response.status_code == 200:
            locations = response.json()
            return {loc.get("id"): loc.get("name", "Unknown") for loc in locations}
        else:
            return {}
    except Exception:
        return {}

@cache_data(ttl=3600)
def get_rc_roles():
    if not api_key or not base_url:
        return {}
    
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(f"{base_url}/roles", headers=headers, timeout=30)
        if response.status_code == 200:
            roles = response.json()
            return {role.get("id"): role.get("name", "Unknown") for role in roles}
        else:
            return {}
    except Exception:
        return {}

def get_location_name(location_id):
    locations = get_rc_locations()
    return locations.get(location_id, "Unknown")

def get_role_name(role_id):
    roles = get_rc_roles()
    return roles.get(role_id, "Unknown")

def process_user_shifts(shifts_json, start_date, end_date):
    if not shifts_json:
        return 0.0, []
    
    total_hours = 0.0
    processed_shifts = []
    
    for shift in shifts_json:
        start_unix = shift.get("start_time_unix")
        end_unix = shift.get("end_time_unix")
        
        if start_unix and end_unix:
            shift_start = datetime.datetime.fromtimestamp(start_unix).date()
            shift_end = datetime.datetime.fromtimestamp(end_unix).date()
            
            if shift_start >= start_date and shift_end <= end_date:
                hours = (end_unix - start_unix) / 3600.0
                total_hours += hours
                
                shift_info = {
                    "start_time_unix": start_unix,
                    "end_time_unix": end_unix,
                    "net_work_hours": hours,
                    "location_id": shift.get("location_id"),
                    "role_id": shift.get("role_id"),
                    "location_name": get_location_name(shift.get("location_id")),
                    "role_name": get_role_name(shift.get("role_id"))
                }
                processed_shifts.append(shift_info)
    
    return total_hours, processed_shifts

def process_user_leave(leave_json, start_date, end_date):
    if not leave_json:
        return 0.0, 0.0, []
    
    total_days = 0.0
    total_hours = 0.0
    processed_records = []
    
    for record in leave_json:
        leave_start_str = record.get("start_date")
        leave_end_str = record.get("end_date")
        
        if not leave_start_str or not leave_end_str:
            continue
        
        leave_start = datetime.datetime.strptime(leave_start_str, "%Y-%m-%d").date()
        leave_end = datetime.datetime.strptime(leave_end_str, "%Y-%m-%d").date()
        
        overlap_start = max(leave_start, start_date)
        overlap_end = min(leave_end, end_date)
        
        if overlap_start <= overlap_end:
            days_in_period = (overlap_end - overlap_start).days + 1
            hours_in_period = days_in_period * 8.0
            
            total_days += days_in_period
            total_hours += hours_in_period
            
            record_info = {
                "type": record.get("type", "Unknown"),
                "status": record.get("status", "Unknown"),
                "days_in_report_period": days_in_period,
                "hours_in_report_period": hours_in_period
            }
            processed_records.append(record_info)
    
    return total_days, total_hours, processed_records

def filter_by_gender(users, selected_gender_codes):
    filtered = []
    for u in users:
        uid = u.get("id")
        user_gender = get_user_gender(uid)
        if user_gender in selected_gender_codes:
            u["gender"] = user_gender
            filtered.append(u)
    return filtered

def sort_user_reports(reports, sort_option):
    if sort_option == "Highest Hours":
        return sorted(reports, key=lambda x: x["unified_total"], reverse=True)
    elif sort_option == "Lowest Hours":
        return sorted(reports, key=lambda x: x["unified_total"])
    elif sort_option == "Name (A-Z)":
        return sorted(reports, key=lambda x: (x["first_name"], x["last_name"]))
    elif sort_option == "Name (Z-A)":
        return sorted(reports, key=lambda x: (x["first_name"], x["last_name"]), reverse=True)
    elif sort_option == "User ID":
        return sorted(reports, key=lambda x: x["user_id"])
    else:
        return reports

def weekday_and_ddmmyy(unix_ts):
    dt = datetime.datetime.fromtimestamp(unix_ts)
    day_name = dt.strftime("%A")
    date_str = dt.strftime("%d/%m/%y")
    return day_name, date_str

def display_user(rpt, show_threshold_color=False):
    uid = rpt["user_id"]
    fname = rpt["first_name"]
    lname = rpt["last_name"]
    gender_display = get_gender_display_name(rpt["gender"])
    
    with st.expander(f"**{fname} {lname}** ({gender_display}) - ID: {uid} - **{rpt['unified_total']:.2f}h**"):
        ut = rpt["unified_total"]
        if show_threshold_color and use_threshold_filter:
            if ut < low_threshold:
                color = "red"
            elif ut > high_threshold:
                color = "orange"
            else:
                color = "green"
            st.markdown(
                f"**Total Hours**: <span style='color:{color};'>{ut:.1f} hours</span>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"**Total Hours**: {ut:.2f} hours")

        st.markdown(f"**Shift Hours**: {rpt['total_shift_hours']:.2f}")
        if rpt["processed_shifts"]:
            rows = []
            for s in rpt["processed_shifts"]:
                day_name, date_str = weekday_and_ddmmyy(s["start_time_unix"])
                loc = s.get("location_name", "Unknown")
                role = s.get("role_name", "Unknown")
                rows.append({
                    "Day": day_name,
                    "Date": date_str,
                    "Location": loc,
                    "Role": role,
                    "Hours": s.get("net_work_hours", 0)
                })
            df_shifts = pd.DataFrame(rows)
            st.dataframe(df_shifts, use_container_width=True, hide_index=True)
        else:
            st.write("No shifts found.")

        if rpt["total_leave_hours"] > 0 or rpt["total_leave_days"] > 0:
            st.markdown(f"**Leave Days**: {rpt['total_leave_days']:.1f}")
            st.markdown(f"**Leave Hours**: {rpt['total_leave_hours']:.2f}")
            if rpt["processed_leave_records"]:
                rows = []
                for rec in rpt["processed_leave_records"]:
                    rows.append({
                        "Type": rec.get("type"),
                        "Status": rec.get("status"),
                        "Days": f"{rec.get('days_in_report_period', 0):.1f}",
                        "Hours": f"{rec.get('hours_in_report_period', 0):.2f}"
                    })
                df_leave = pd.DataFrame(rows)
                st.dataframe(df_leave, use_container_width=True, hide_index=True)

def display_user_mobile(rpt, show_threshold_color=False):
    uid = rpt["user_id"]
    fname = rpt["first_name"]
    lname = rpt["last_name"]
    gender_display = get_gender_display_name(rpt["gender"])
    
    st.subheader(f"{fname} {lname} ({gender_display}) (ID: {uid})")

    ut = rpt["unified_total"]
    if show_threshold_color and use_threshold_filter:
        if ut < low_threshold:
            color = "red"
        elif ut > high_threshold:
            color = "orange"
        else:
            color = "green"
        st.markdown(
            f"**Total Hours**: <span style='color:{color};'>{ut:.1f} hours</span>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(f"**Total Hours**: {ut:.2f} hours")

    st.markdown(f"**Shift Hours**: {rpt['total_shift_hours']:.2f}")
    if rpt["processed_shifts"]:
        rows = []
        for s in rpt["processed_shifts"]:
            day_name, date_str = weekday_and_ddmmyy(s["start_time_unix"])
            loc = s.get("location_name", "Unknown")
            role = s.get("role_name", "Unknown")
            rows.append({
                "Day": day_name,
                "Date": date_str,
                "Location": loc,
                "Role": role,
                "Hours": s.get("net_work_hours", 0)
            })
        df_shifts = pd.DataFrame(rows)
        st.dataframe(df_shifts, use_container_width=True, hide_index=True)
    else:
        st.write("No shifts found.")

    if rpt["total_leave_hours"] > 0 or rpt["total_leave_days"] > 0:
        st.markdown(f"**Leave Days**: {rpt['total_leave_days']:.1f}")
        st.markdown(f"**Leave Hours**: {rpt['total_leave_hours']:.2f}")
        if rpt["processed_leave_records"]:
            rows = []
            for rec in rpt["processed_leave_records"]:
                rows.append({
                    "Type": rec.get("type"),
                    "Status": rec.get("status"),
                    "Days": f"{rec.get('days_in_report_period', 0):.1f}",
                    "Hours": f"{rec.get('hours_in_report_period', 0):.2f}"
                })
            df_leave = pd.DataFrame(rows)
            st.dataframe(df_leave, use_container_width=True, hide_index=True)

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

    if use_threshold_filter and low_threshold and high_threshold and low_threshold > high_threshold:
        st.error("Invalid threshold range.")
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

        shifts_json = get_rc_shifts(start_ts, end_ts, uid)
        total_shift_hours, processed_shifts = process_user_shifts(shifts_json, start_date, end_date)

        leave_json = get_rc_leave(start_str, end_str, uid)
        total_leave_days, total_leave_hours, processed_leave_records = process_user_leave(leave_json, start_date, end_date)

        unified_total = total_shift_hours + total_leave_hours

        user_reports.append({
            "user_id": uid,
            "first_name": fname,
            "last_name": lname,
            "gender": gender,
            "total_shift_hours": total_shift_hours,
            "processed_shifts": processed_shifts,
            "total_leave_days": total_leave_days,
            "total_leave_hours": total_leave_hours,
            "processed_leave_records": processed_leave_records,
            "unified_total": unified_total
        })

    progress_bar.empty()
    status_text.empty()

    user_reports = sort_user_reports(user_reports, sort_option)
    
    # Store report in session state for PDF generation
    st.session_state.generated_report = {
        'user_reports': user_reports,
        'start_date': start_date,
        'end_date': end_date,
        'use_threshold_filter': use_threshold_filter,
        'low_threshold': low_threshold,
        'high_threshold': high_threshold
    }

    st.write("## 📋 Report Results")
    
    # PDF Export Button
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
                
                filename = f"shift_report_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.pdf"
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
        st.success(f"👥 {len(user_reports)} users | {summary_text}")
    
    if st.session_state.mobile_view:
        if use_threshold_filter and low_threshold and high_threshold:
            below = [rpt for rpt in user_reports if rpt["unified_total"] < low_threshold]
            above = [rpt for rpt in user_reports if rpt["unified_total"] > high_threshold]
            between = [rpt for rpt in user_reports if low_threshold <= rpt["unified_total"] <= high_threshold]

            if below:
                st.write(f"### 🔴 Below {low_threshold}h ({len(below)} users)")
                for rpt in below:
                    display_user_mobile(rpt, show_threshold_color=True)

            if above:
                st.write(f"### 🟠 Above {high_threshold}h ({len(above)} users)")
                for rpt in above:
                    display_user_mobile(rpt, show_threshold_color=True)

            if between:
                st.write(f"### 🟢 {low_threshold}-{high_threshold}h ({len(between)} users)")
                for rpt in between:
                    display_user_mobile(rpt, show_threshold_color=True)
        else:
            st.write(f"### All Users ({len(user_reports)})")
            for rpt in user_reports:
                display_user_mobile(rpt)
    else:
        if use_threshold_filter and low_threshold and high_threshold:
            below = [rpt for rpt in user_reports if rpt["unified_total"] < low_threshold]
            above = [rpt for rpt in user_reports if rpt["unified_total"] > high_threshold]
            between = [rpt for rpt in user_reports if low_threshold <= rpt["unified_total"] <= high_threshold]

            if below:
                st.header(f"Below {low_threshold} hours ({len(below)} users)")
                for rpt in below:
                    display_user(rpt, show_threshold_color=True)

            if above:
                st.header(f"Above {high_threshold} hours ({len(above)} users)")
                for rpt in above:
                    display_user(rpt, show_threshold_color=True)

            if between:
                st.header(f"Between {low_threshold}-{high_threshold} hours ({len(between)} users)")
                for rpt in between:
                    display_user(rpt, show_threshold_color=True)
        else:
            st.header(f"All Users ({len(user_reports)})")
            for rpt in user_reports:
                display_user(rpt)

    st.success("✅ Report complete!")
else:
    st.write("### 👋 Welcome!")
    st.write("Configure settings in the sidebar and click **Generate Report** to begin.")
    
    with st.expander("📱 Mobile Tips"):
        st.write("• Add this page to your home screen for app-like access")
        st.write("• Use the Mobile button above to switch to compact view")
        st.write("• Swipe right or tap ⚙️ to open settings")
        st.write("• Tables are horizontally scrollable on mobile")
    
    with st.expander("⚡ Quick Start"):
        st.write("1. Enter your API key in the sidebar")
        st.write("2. Select your date range or use period navigation")
        st.write("3. Configure filters as needed")
        st.write("4. Click **Generate Report** or **⚡ Quick Report**")
