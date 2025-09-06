import os
import requests
import datetime
import pytz
import pandas as pd
import streamlit as st

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

st.sidebar.header("Report Configuration")

manual_dates = st.sidebar.checkbox("Manual Date Selection", value=False, 
                                  help="Toggle to manually select dates instead of using monthly periods")

if manual_dates:
    st.sidebar.subheader("Manual Date Range")
    default_start = datetime.date(2025, 6, 11)
    default_end = datetime.date(2025, 7, 10)
    start_date = st.sidebar.date_input("Report Start Date", value=default_start)
    end_date = st.sidebar.date_input("Report End Date", value=default_end)
    if start_date > end_date:
        st.sidebar.error("Start date must be on or before end date.")
else:
    st.sidebar.subheader("Monthly Period Navigation")
    
    current_start, current_end = get_monthly_period(
        st.session_state.current_period_year, 
        st.session_state.current_period_month
    )
    
    period_name = format_period_name(current_start, current_end)
    st.sidebar.write(f"**Current Period:** {period_name}")
    
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        if st.button("◀ Previous", help="Go to previous month period"):
            if st.session_state.current_period_month == 1:
                st.session_state.current_period_month = 12
                st.session_state.current_period_year -= 1
            else:
                st.session_state.current_period_month -= 1
            st.rerun()
    
    with col2:
        if st.button("Today", help="Go to current month period"):
            current_start, current_end = get_current_period()
            st.session_state.current_period_year = current_start.year
            st.session_state.current_period_month = current_start.month
            st.rerun()
    
    with col3:
        if st.button("Next ▶", help="Go to next month period"):
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

st.sidebar.subheader("Gender Filtering")
enable_gender_filter = st.sidebar.checkbox("Enable Gender Filter", value=False)

if enable_gender_filter:
    gender_options = st.sidebar.multiselect(
        "Select Genders to Include",
        options=['Male', 'Female', 'Terminal', 'Unknown'],
        default=['Male', 'Female', 'Terminal', 'Unknown'],
        help="Select which genders to include in the report"
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

use_threshold_filter = st.sidebar.checkbox("Enable Threshold Filtering", value=True)

if use_threshold_filter:
    low_threshold = st.sidebar.number_input("Threshold: Below this (hours)", value=156, step=1, format="%d")
    high_threshold = st.sidebar.number_input("Threshold: Above this (hours)", value=170, step=1, format="%d")
    if low_threshold > high_threshold:
        st.sidebar.error("Low threshold should be <= high threshold.")
else:
    low_threshold = None
    high_threshold = None

st.sidebar.subheader("Sorting Options")
sort_option = st.sidebar.radio(
    "Sort users by Total Hours:",
    ("Alphabetical (First Name)", "Highest to Lowest", "Lowest to Highest")
)

api_key_input = st.sidebar.text_input(
    "API Key", 
    type="password",
    help="Enter your API key or set it as an environment variable"
)

ignored_input = st.sidebar.text_input(
    "Ignored User IDs (comma-separated)", 
    value=','.join(map(str, DEFAULT_IGNORED_IDS))
)

ignored_user_ids = set()
for part in ignored_input.split(","):
    part = part.strip()
    if part.isdigit():
        ignored_user_ids.add(int(part))

generate = st.sidebar.button("Generate Report")

st.title("Unified Shift & Leave Report")

if not manual_dates:
    period_display = format_period_name(start_date, end_date)
    st.info(f"📅 **Report Period**: {period_display}")
else:
    st.info(f"📅 **Report Period**: {start_date.strftime('%d/%m/%y')} - {end_date.strftime('%d/%m/%y')}")

active_filters = []
if enable_gender_filter:
    active_filters.append(f"Gender: {', '.join(gender_options)}")
if use_threshold_filter:
    active_filters.append(f"Hours: {low_threshold}-{high_threshold}")
if ignored_user_ids:
    active_filters.append(f"Ignored: {len(ignored_user_ids)} users")

if active_filters:
    st.info(f"🔍 **Active Filters**: {' | '.join(active_filters)}")

API_KEY = None

if api_key_input:
    API_KEY = api_key_input.strip()
elif os.environ.get("API_KEY"):
    API_KEY = os.environ.get("API_KEY")

if not API_KEY:
    st.error("⚠️ **API Key Required**")
    st.write("Please provide your API key in one of these ways:")
    st.write("1. Enter it in the sidebar (recommended for testing)")
    st.write("2. Set it as an environment variable: `API_KEY`")
    st.write("3. Add it to your deployment environment")
    st.info("💡 **Tip**: You can find your API key in your account settings.")
    st.stop()

if not API_KEY.startswith(('sk-', 'rk-', 'pk-')) and len(API_KEY) < 20:
    st.warning("⚠️ **API Key Format Warning**: Your API key doesn't look like a typical API key. Please verify it's correct.")

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
def get_rotacloud_users():
    try:
        resp = requests.get(USERS_BASE_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 401:
            st.error("🔒 **Authentication Error**: Invalid API key or expired token")
            st.write("Please check your API key and try again.")
        elif resp.status_code == 403:
            st.error("🚫 **Permission Error**: Your API key doesn't have permission to access users")
        elif resp.status_code == 429:
            st.error("⏰ **Rate Limit**: Too many requests. Please wait and try again.")
        else:
            st.error(f"🌐 **HTTP Error {resp.status_code}**: {e}")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ **Timeout Error**: Request took too long. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"🔗 **Connection Error**: {e}")
        return None

@cache_data(ttl=300)
def get_rotacloud_shifts(start_ts, end_ts, user_id):
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
    except requests.exceptions.RequestException as e:
        st.warning(f"Warning: Error fetching shifts for user {user_id}: {e}")
        return None

@cache_data(ttl=300)
def get_rotacloud_leave(start_str, end_str, user_id):
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
    except requests.exceptions.RequestException as e:
        st.warning(f"Warning: Error fetching leave for user {user_id}: {e}")
        return None

@cache_data(ttl=600)
def get_location_name(location_id):
    if location_id is None:
        return "N/A"
    try:
        resp = requests.get(f"{LOCATIONS_BASE_URL}/{location_id}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("name", f"Unknown Location ({location_id})")
    except:
        return f"Error Loc ({location_id})"

@cache_data(ttl=600)
def get_role_name(role_id):
    if role_id is None:
        return "N/A"
    try:
        resp = requests.get(f"{ROLES_BASE_URL}/{role_id}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("name", f"Unknown Role ({role_id})")
    except:
        return f"Error Role ({role_id})"

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

if st.sidebar.button("🔍 Test API Connection"):
    with st.spinner("Testing API connection..."):
        success, message = test_api_connection()
        if success:
            st.sidebar.success(f"✅ {message}")
        else:
            st.sidebar.error(f"❌ {message}")

if generate:
    st.info("Generating report... this may take some time depending on number of users.")

    if start_date > end_date:
        st.error("Invalid date range: start date is after end date.")
        st.stop()

    if use_threshold_filter and low_threshold > high_threshold:
        st.error("Invalid threshold range: low threshold is greater than high threshold.")
        st.stop()

    if enable_gender_filter and not selected_gender_codes:
        st.error("No genders selected. Please select at least one gender or disable gender filtering.")
        st.stop()

    with st.spinner("Testing API connection..."):
        success, message = test_api_connection()
        if not success:
            st.error(f"❌ **API Connection Failed**: {message}")
            st.write("**Troubleshooting steps:**")
            st.write("1. Verify your API key is correct")
            st.write("2. Check if your API key has the necessary permissions")
            st.write("3. Ensure your API key hasn't expired")
            st.write("4. Try generating a new API key from your account")
            st.stop()
        else:
            st.success(f"✅ {message}")

    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")
    start_ts = date_to_unix_timestamp(start_date, hour=0, minute=0, second=0)
    end_ts   = date_to_unix_timestamp(end_date, hour=23, minute=59, second=59)
    if start_ts is None or end_ts is None:
        st.error("Date conversion error; please check dates.")
        st.stop()

    with st.spinner("Fetching users..."):
        all_users = get_rotacloud_users()
        if all_users is None:
            st.error("Could not retrieve users; aborting.")
            st.stop()

    refined = []
    for u in all_users:
        uid = u.get("id")
        if uid is None:
            continue
        if uid in ignored_user_ids:
            continue
        refined.append({
            "id": uid,
            "first_name": u.get("first_name") or "",
            "last_name": u.get("last_name") or ""
        })

    if enable_gender_filter:
        refined = filter_by_gender(refined, selected_gender_codes)
        st.write(f"Total users after gender filtering: {len(refined)}")
    else:
        st.write(f"Total users after filtering ignored IDs: {len(refined)}")

    user_reports = []
    progress_bar = st.progress(0)
    total_users = len(refined)
    
    for idx, u in enumerate(refined):
        uid = u["id"]
        fname = u["first_name"]
        lname = u["last_name"]
        gender = u.get("gender", get_user_gender(uid))

        progress_bar.progress((idx+1)/total_users)
        
        st.write(f"Processing: {fname} {lname} ({get_gender_display_name(gender)}) ({idx+1}/{total_users})")

        shifts_json = get_rotacloud_shifts(start_ts, end_ts, uid)
        total_shift_hours, processed_shifts = process_user_shifts(shifts_json, start_date, end_date)

        leave_json = get_rotacloud_leave(start_str, end_str, uid)
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

    user_reports = sort_user_reports(user_reports, sort_option)

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
            st.subheader(
                f"**Unified Total Hours (Shifts + Leave)**: "
                f"<span style='color:{color};'>{ut:.1f} hours</span>",
                unsafe_allow_html=True
            )
        else:
            ut = rpt["unified_total"]
            st.markdown(f"### Unified Total Hours (Shifts + Leave): <span style='color:orange;'>{ut:.2f} hours</span>",
                unsafe_allow_html=True)

        st.markdown(f"**Total Shift Hours in Period**: {rpt['total_shift_hours']:.2f}")
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
            st.table(df_shifts)
        else:
            st.write("No shifts found in this period.")

        if rpt["total_leave_hours"] > 0 or rpt["total_leave_days"] > 0:
            st.markdown(f"**Total Leave Days in Period**: {rpt['total_leave_days']:.1f}")
            st.markdown(f"**Total Leave Hours in Period**: {rpt['total_leave_hours']:.2f}")
            if rpt["processed_leave_records"]:
                rows = []
                for rec in rpt["processed_leave_records"]:
                    rows.append({
                        "Type ID": rec.get("type"),
                        "Status": rec.get("status"),
                        "Days in Period": f"{rec.get('days_in_report_period', 0):.1f}",
                        "Hours in Period": f"{rec.get('hours_in_report_period', 0):.2f}"
                    })
                df_leave = pd.DataFrame(rows)
                st.table(df_leave)
            else:
                st.write("No approved leave records with days/hours in period.")
        else:
            st.write("No approved leave in this period.")

        st.markdown("---")

    st.write("## Report Results")
    
    if user_reports:
        gender_summary = {}
        for rpt in user_reports:
            gender_display = get_gender_display_name(rpt["gender"])
            gender_summary[gender_display] = gender_summary.get(gender_display, 0) + 1
        
        summary_text = ", ".join([f"{count} {gender}" for gender, count in gender_summary.items()])
        st.info(f"👥 **Gender Breakdown**: {summary_text} (Total: {len(user_reports)} users)")
    
    if use_threshold_filter:
        below = [rpt for rpt in user_reports if rpt["unified_total"] < low_threshold]
        above = [rpt for rpt in user_reports if rpt["unified_total"] > high_threshold]
        between = [rpt for rpt in user_reports if low_threshold <= rpt["unified_total"] <= high_threshold]

        st.header(f"Users with Unified Total < {low_threshold} hours (count: {len(below)})")
        if not below:
            st.write("None in this group.")
        else:
            for rpt in below:
                display_user(rpt, show_threshold_color=True)

        st.header(f"Users with Unified Total > {high_threshold} hours (count: {len(above)})")
        if not above:
            st.write("None in this group.")
        else:
            for rpt in above:
                display_user(rpt, show_threshold_color=True)

        st.header(f"Users with Unified Total between {low_threshold} and {high_threshold} hours (inclusive) (count: {len(between)})")
        if not between:
            st.write("None in this group.")
        else:
            for rpt in between:
                display_user(rpt, show_threshold_color=True)
    else:
        st.header(f"All Users - Sorted by: {sort_option} (count: {len(user_reports)})")
        if not user_reports:
            st.write("No users found.")
        else:
            for rpt in user_reports:
                display_user(rpt, show_threshold_color=False)

    st.success("Report generation complete.")
else:
    st.write("Adjust configuration in the sidebar, then click **Generate Report**.") 
    st.write("**Quick Start:**")
    st.write("1. Enter your API key in the sidebar")
    st.write("2. Test the API connection")
    st.write("3. Use navigation buttons to select period or enable manual date selection")
    st.write("4. Configure gender filtering if needed")
    st.write("5. Adjust thresholds if needed")
    st.write("6. Choose sorting preference")
    st.write("7. Toggle threshold filtering if desired")
    st.write("8. Click 'Generate Report'")
