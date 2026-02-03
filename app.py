import os
import requests
import datetime
import pytz
import pandas as pd
import streamlit as st
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

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
    
    days_in_period = (end_date - start_date).days + 1  # inclusive
    weeks = days_in_period / 7
    expected = weeks * weekly_hours
    return round(expected, 2)

def get_variance_color(actual_hours, expected_hours):
    """Determine color based on variance"""
    if expected_hours is None or expected_hours <= 0:
        return None
    
    if actual_hours < expected_hours:
        return "red"  # Under-worked
    elif actual_hours > expected_hours:
        return "green"  # Over-worked
    else:
        return "gray"  # On target

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

# PDF Generation Function
def generate_pdf_report(user_reports, start_date, end_date, use_threshold_filter=False, 
                       low_threshold=None, high_threshold=None):
    """Generate a styled PDF report with Tailwind-inspired design"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter),
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
    story.append(Paragraph("📊 Shift & Leave Report with Variance Analysis", title_style))
    
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
    
    # Table header style
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
    
    # Table data style
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
            
        # Section header
        story.append(Paragraph(f"{group_name} ({len(group_reports)} users)", section_header_style))
        
        # Create summary table with variance column
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
        
        # Create table
        col_widths = [2*inch, 0.7*inch, 1*inch, 1*inch, 1*inch, 1.3*inch]
        summary_table = Table(summary_data, colWidths=col_widths)
        
        # Apply styles
        summary_table.setStyle(TableStyle(table_header_style + table_data_style))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
    
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

st.title("📊 Shift & Leave Report with Variance Analysis")

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
    
    # Variance filter
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

# Helper functions
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
        # Try to get contract via user endpoint first
        resp = requests.get(f"{USERS_BASE_URL}/{user_id}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        user_data = resp.json()
        
        # Check if weekly hours are in user data
        if "weekly_hours" in user_data:
            return float(user_data.get("weekly_hours", 0)) or None
        
        # Alternative: try contracts endpoint
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
        return sorted(user_reports, key=lambda r: (r.get("variance", 0)), 
                     key=lambda r: r.get("variance", 0))
    elif sort_option == "Most Over-worked":
        return sorted(user_reports, key=lambda r: r.get("variance", 0), reverse=True)
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

def display_user_mobile(rpt, show_variance_indicator=True):
    fname = rpt["first_name"]
    lname = rpt["last_name"]
    gender_display = get_gender_display_name(rpt["gender"])[:1]
    ut = rpt["unified_total"]
    expected = rpt.get("expected_hours")
    
    if show_variance_indicator and expected is not None:
        variance_emoji = get_variance_emoji(ut, expected)
    else:
        variance_emoji = "📊"
    
    with st.expander(f"{variance_emoji} {fname} {lname} ({gender_display}) - {ut:.1f}h"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", f"{ut:.1f}h")
        with col2:
            st.metric("Shifts", f"{rpt['total_shift_hours']:.1f}h")
        with col3:
            st.metric("Leave", f"{rpt['total_leave_hours']:.1f}h")
        
        # Variance section
        if expected is not None:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Expected", f"{expected:.1f}h")
            with col2:
                variance = ut - expected
                st.metric("Variance", f"{variance:+.1f}h")
            with col3:
                variance_pct = (variance / expected * 100) if expected > 0 else 0
                st.metric("Variance %", f"{variance_pct:+.1f}%")
        
        if rpt.get("weekly_hours"):
            st.caption(f"📅 **Weekly Contract**: {rpt['weekly_hours']:.1f}h")
        
        if rpt["processed_shifts"]:
            st.caption("**Shifts:**")
            for s in rpt["processed_shifts"][:5]:
                day_name, date_str = weekday_and_ddmmyy(s["start_time_unix"])
                st.text(f"• {day_name[:3]} {date_str}: {s.get('net_work_hours', 0)}h")
            if len(rpt["processed_shifts"]) > 5:
                st.caption(f"...+{len(rpt['processed_shifts'])-5} more")
        
        if rpt["total_leave_hours"] > 0:
            st.caption(f"**Leave:** {rpt['total_leave_days']:.1f} days ({rpt['total_leave_hours']:.1f}h)")

def display_user(rpt, show_variance_indicator=True):
    fname = rpt["first_name"]
    lname = rpt["last_name"]
    uid = rpt["user_id"]
    gender = rpt["gender"]
    gender_display = get_gender_display_name(gender)
    
    st.subheader(f"{fname} {lname} ({gender_display}) (ID: {uid})")

    ut = rpt["unified_total"]
    expected = rpt.get("expected_hours")
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Hours", f"{ut:.2f}")
    with col2:
        st.metric("Shift Hours", f"{rpt['total_shift_hours']:.2f}")
    with col3:
        st.metric("Leave Hours", f"{rpt['total_leave_hours']:.2f}")
    
    # Variance section
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
    
    if rpt["processed_shifts"]:
        st.markdown("**Shifts:**")
        rows = []
        for s in rpt["processed_shifts"]:
            day_name, date_str = weekday_and_ddmmyy(s["start_time_unix"])
            loc = get_location_name(s.get("location_id"))
            role = get_role_name(s.get("role_id"))
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
        st.markdown("**Leave:**")
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

        # Fetch weekly hours
        weekly_hours = get_user_weekly_hours(uid)

        shifts_json = get_rc_shifts(start_ts, end_ts, uid)
        total_shift_hours, processed_shifts = process_user_shifts(shifts_json, start_date, end_date)

        leave_json = get_rc_leave(start_str, end_str, uid)
        total_leave_days, total_leave_hours, processed_leave_records = process_user_leave(leave_json, start_date, end_date)

        unified_total = total_shift_hours + total_leave_hours
        
        # Calculate expected hours
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

    # Apply variance filter if enabled
    if enable_variance_filter:
        user_reports = filter_by_variance(user_reports, variance_options)

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
        
        # Variance summary
        under_worked = len([r for r in user_reports if r.get("variance") and r["variance"] < 0])
        over_worked = len([r for r in user_reports if r.get("variance") and r["variance"] > 0])
        on_target = len([r for r in user_reports if r.get("variance") == 0])
        
        st.success(f"👥 {len(user_reports)} users | {summary_text} | 🔴 {under_worked} under | 🟢 {over_worked} over | 🟡 {on_target} on-target")
    
    if st.session_state.mobile_view:
        st.write(f"### All Users ({len(user_reports)})")
        for rpt in user_reports:
            display_user_mobile(rpt, show_variance_indicator=True)
    else:
        st.write(f"### All Users ({len(user_reports)})")
        for rpt in user_reports:
            display_user(rpt, show_variance_indicator=True)

    st.success("✅ Report complete!")
else:
    st.write("### 👋 Welcome!")
    st.write("Configure settings in the sidebar and click **Generate Report** to begin.")
    
    with st.expander("📱 Mobile Tips"):
        st.write("• Add this page to your home screen for app-like access")
        st.write("• Use the Mobile button above to switch to compact view")
        st.write("• Swipe right or tap ⚙️ to open settings")
        st.write("• Tables are horizontally scrollable on mobile")
    
    with st.expander("✨ Variance Analysis Features"):
        st.write("• **Weekly Hours Extraction**: Automatically pulls contracted weekly hours from user profiles")
        st.write("• **Expected Hours Calculation**: Uses formula: (days in period / 7) × weekly hours")
        st.write("• **Color-Coded Status**: 🔴 Red = Under-worked, 🟢 Green = Over-worked, 🟡 Yellow = On-target")
        st.write("• **Variance Filtering**: Filter users by their work status")
        st.write("• **Detailed Variance Metrics**: View actual hours, expected hours, and percentage variance")
    
    with st.expander("⚡ Quick Start"):
        st.write("1. Enter your API key in the sidebar")
        st.write("2. Select your date range or use period navigation")
        st.write("3. Configure filters as needed (including new Variance filter)")
        st.write("4. Click **Generate Report** to see variance analysis")
