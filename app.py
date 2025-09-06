import os
import requests
import datetime
import pytz
import pandas as pd
import streamlit as st

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

if 'current_period_year' not in st.session_state:
    current_start, current_end = get_current_period()
    st.session_state.current_period_year = current_start.year
    st.session_state.current_period_month = current_start.month

if 'mobile_view' not in st.session_state:
    st.session_state.mobile_view = False

st.title("📊 Shift & Leave Report")

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🔄 Refresh", use_container_width=True, help="Clear cache and refresh"):
        st.cache_data.clear()
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
    
    use_threshold_filter = st.checkbox("Enable Threshold Filter", value=True)
    
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
        ("Alphabetical", "Highest Hours", "Lowest Hours"),
        horizontal=True
    )
    
    if sort_option == "Alphabetical":
        sort_option = "Alphabetical (First Name)"
    elif sort_option == "Highest Hours":
        sort_option = "Highest to Lowest"
    else:
        sort_option = "Lowest to Highest"
    
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

BASE_API_URL = st.secrets.get("api_base_url", "https://api.example.com")
USERS_BASE_URL    = f"{BASE_API_URL}/v1/users"
SHIFTS_BASE_URL   = f"{BASE_API_URL}/v1/shifts"  
LOCATIONS_BASE_URL= f"{BASE_API_URL}/v1/locations"
ROLES_BASE_URL    = f"{BASE_API_URL}/v1/roles"
LEAVE_BASE_URL    = f"{BASE_API_URL}/v1/leave"

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

def display_user_mobile(rpt, show_threshold_color=False):
    fname = rpt["first_name"]
    lname = rpt["last_name"]
    gender_display = get_gender_display_name(rpt["gender"])[:1]
    ut = rpt["unified_total"]
    
    if show_threshold_color and use_threshold_filter:
        if ut < low_threshold:
            indicator = "🔴"
        elif ut > high_threshold:
            indicator = "🟠"
        else:
            indicator = "🟢"
    else:
        indicator = "📊"
    
    with st.expander(f"{indicator} {fname} {lname} ({gender_display}) - {ut:.1f}h"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", f"{ut:.1f}h")
        with col2:
            st.metric("Shifts", f"{rpt['total_shift_hours']:.1f}h")
        with col3:
            st.metric("Leave", f"{rpt['total_leave_hours']:.1f}h")
        
        if rpt["processed_shifts"]:
            st.caption("**Shifts:**")
            for s in rpt["processed_shifts"][:5]:
                day_name, date_str = weekday_and_ddmmyy(s["start_time_unix"])
                st.text(f"• {day_name[:3]} {date_str}: {s.get('net_work_hours', 0)}h")
            if len(rpt["processed_shifts"]) > 5:
                st.caption(f"...+{len(rpt["processed_shifts"])-5} more")
        
        if rpt["total_leave_hours"] > 0:
            st.caption(f"**Leave:** {rpt['total_leave_days']:.1f} days ({rpt['total_leave_hours']:.1f}h)")

def display_user(rpt, show_threshold_color=False):
    fname = rpt["first_name"]
    lname = rpt["last_name"]
    uid = rpt["user_id"]
    gender = rpt["gender"]
    gender_display = get_gender_display_name(gender)
    
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

    st.write("## 📋 Report Results")
    
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
        
