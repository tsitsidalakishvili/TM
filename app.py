import os

import altair as alt
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth

try:
    from streamlit.errors import StreamlitSecretNotFoundError
except Exception:
    class StreamlitSecretNotFoundError(Exception):
        """Fallback for older Streamlit versions."""

st.set_page_config(page_title="Freedom Square CRM (Short)", layout="wide")

load_dotenv()


def _secrets_file_exists():
    app_secrets = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
    user_secrets = os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml")
    return os.path.exists(app_secrets) or os.path.exists(user_secrets)


def get_config(key, default=None):
    try:
        if _secrets_file_exists():
            if key in st.secrets:
                return st.secrets.get(key, default)
    except (StreamlitSecretNotFoundError, FileNotFoundError):
        pass
    value = os.getenv(key)
    return value if value is not None else default


NEO4J_URI = get_config("NEO4J_URI")
NEO4J_USER = get_config("NEO4J_USER") or get_config("NEO4J_USERNAME") or "neo4j"
NEO4J_PASSWORD = get_config("NEO4J_PASSWORD")
NEO4J_DATABASE = get_config("NEO4J_DATABASE", "neo4j")

driver = None
_auth_rate_limited = False


def _session_execute_read(session, func, *args):
    if hasattr(session, "execute_read"):
        return session.execute_read(func, *args)
    return session.read_transaction(func, *args)


def _session_execute_write(session, func, *args):
    if hasattr(session, "execute_write"):
        return session.execute_write(func, *args)
    return session.write_transaction(func, *args)


def init_driver():
    global driver, _auth_rate_limited
    if _auth_rate_limited:
        return False
    if not NEO4J_URI or not NEO4J_PASSWORD:
        driver = None
        return False
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session(database=NEO4J_DATABASE) as session:
            _session_execute_read(session, lambda tx: list(tx.run("RETURN 1")))
        return True
    except Exception as exc:
        error_str = str(exc)
        if "AuthenticationRateLimit" in error_str or "authentication details too many times" in error_str:
            _auth_rate_limited = True
            st.error("Neo4j authentication rate limit reached. Please wait a few minutes.")
        else:
            st.error(f"Could not initialize Neo4j driver: {exc}")
        driver = None
        return False


def _run_read(tx, query, params):
    result = tx.run(query, params or {})
    return [r.data() for r in result]


def run_query(query, params=None, silent=False):
    if driver is None:
        if not silent:
            st.warning("Neo4j driver not available. Check connection settings.")
        return pd.DataFrame()
    if _auth_rate_limited:
        if not silent:
            st.error("Neo4j authentication rate limit active. Please wait before retrying.")
        return pd.DataFrame()
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            data = _session_execute_read(session, _run_read, query, params)
            return pd.DataFrame(data)
    except Exception as exc:
        if not silent:
            st.error(f"Neo4j query failed: {exc}")
        return pd.DataFrame()


def _run_write(tx, query, params):
    tx.run(query, params or {})


def run_write(query, params=None):
    if driver is None:
        st.warning("Neo4j driver not available. Check connection settings.")
        return False
    if _auth_rate_limited:
        st.error("Neo4j authentication rate limit active. Please wait before retrying.")
        return False
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            _session_execute_write(session, _run_write, query, params)
        return True
    except Exception as exc:
        st.error(f"Neo4j write failed: {exc}")
        return False


def upsert_person(payload):
    query = """
    MERGE (p:Person {email: $email})
    ON CREATE SET p.personId = randomUUID(), p.createdAt = datetime()
    SET p.firstName = $firstName,
        p.lastName = $lastName,
        p.gender = $gender,
        p.age = $age,
        p.phone = $phone,
        p.lat = $lat,
        p.lon = $lon,
        p.effortHours = coalesce($effortHours, p.effortHours),
        p.eventsAttendedCount = coalesce($eventsAttendedCount, p.eventsAttendedCount),
        p.referralCount = coalesce($referralCount, p.referralCount)
    WITH p
    MERGE (st:SupporterType {name: $supporterType})
    MERGE (p)-[:CLASSIFIED_AS]->(st)
    WITH p
    FOREACH (_ IN CASE WHEN $address IS NULL OR $address = '' THEN [] ELSE [1] END |
        MERGE (a:Address {fullAddress: $address})
        ON CREATE SET a.latitude = $lat, a.longitude = $lon
        ON MATCH SET a.latitude = coalesce($lat, a.latitude),
                    a.longitude = coalesce($lon, a.longitude)
        MERGE (p)-[:LIVES_AT]->(a)
    )
    """
    return run_write(query, payload)


def clean_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def split_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = str(value).split(",")
    cleaned = []
    for item in items:
        text = clean_text(item)
        if text:
            cleaned.append(text)
    return cleaned


def format_list_label(values, limit=6):
    items = [str(v).strip() for v in values or [] if str(v).strip()]
    items = sorted(set(items))
    if not items:
        return "None"
    if len(items) > limit:
        return ", ".join(items[:limit]) + f" (+{len(items) - limit} more)"
    return ", ".join(items)


def normalize_supporter_type(value, default_type="Supporter"):
    text = clean_text(value)
    if not text:
        return default_type
    if "member" in text.lower():
        return "Member"
    return "Supporter"


def _normalize_column(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "")
    )


def _get_column(df, candidates):
    normalized = {_normalize_column(col): col for col in df.columns}
    for cand in candidates:
        key = _normalize_column(cand)
        if key in normalized:
            return normalized[key]
    return None


def _build_import_rows(df, default_type):
    if df.empty:
        return []
    col_email = _get_column(df, ["email", "primary_email", "e_mail", "e-mail", "email_address"])
    col_email_secondary = _get_column(
        df, ["secondary_email", "alternate_email", "alt_email"]
    )
    if not col_email:
        return []
    col_first = _get_column(df, ["first_name", "firstname", "first"])
    col_last = _get_column(df, ["last_name", "lastname", "last"])
    col_gender = _get_column(df, ["gender", "sex"])
    col_age = _get_column(df, ["age"])
    col_phone = _get_column(df, ["phone", "primary_phone", "mobile"])
    col_phone_secondary = _get_column(df, ["secondary_phone", "alt_phone"])
    col_address = _get_column(df, ["address", "fulladdress", "full_address"])
    col_lat = _get_column(df, ["lat", "latitude"])
    col_lon = _get_column(df, ["lon", "lng", "longitude"])
    col_type = _get_column(df, ["supporter_type", "type", "group"])
    col_effort = _get_column(df, ["effort_hours", "volunteer_hours", "hours", "time_spent"])
    col_events = _get_column(df, ["events_attended", "events_attended_count", "event_attended", "event_attend_count"])
    col_refs = _get_column(df, ["referral_count", "references", "referrals", "recruits"])
    col_education = _get_column(df, ["education", "education_level"])
    col_skills = _get_column(df, ["skills", "skill_list", "skill"])

    rows = []
    for _, row in df.iterrows():
        email = clean_text(row.get(col_email))
        if not email and col_email_secondary:
            email = clean_text(row.get(col_email_secondary))
        if not email:
            continue
        age_val = pd.to_numeric(row.get(col_age), errors="coerce") if col_age else None
        age = int(age_val) if age_val is not None and not pd.isna(age_val) and age_val > 0 else None
        lat_val = pd.to_numeric(row.get(col_lat), errors="coerce") if col_lat else None
        lon_val = pd.to_numeric(row.get(col_lon), errors="coerce") if col_lon else None
        lat = float(lat_val) if lat_val is not None and not pd.isna(lat_val) else None
        lon = float(lon_val) if lon_val is not None and not pd.isna(lon_val) else None
        supporter_type = clean_text(row.get(col_type)) if col_type else None
        effort_val = pd.to_numeric(row.get(col_effort), errors="coerce") if col_effort else None
        effort_hours = float(effort_val) if effort_val is not None and not pd.isna(effort_val) else None
        events_val = pd.to_numeric(row.get(col_events), errors="coerce") if col_events else None
        events_attended = int(events_val) if events_val is not None and not pd.isna(events_val) else None
        refs_val = pd.to_numeric(row.get(col_refs), errors="coerce") if col_refs else None
        referrals = int(refs_val) if refs_val is not None and not pd.isna(refs_val) else None
        education = clean_text(row.get(col_education)) if col_education else None
        skills = split_list(row.get(col_skills)) if col_skills else []
        rows.append(
            {
                "email": email,
                "firstName": clean_text(row.get(col_first)) if col_first else None,
                "lastName": clean_text(row.get(col_last)) if col_last else None,
                "gender": clean_text(row.get(col_gender)) if col_gender else None,
                "age": age,
                "phone": clean_text(row.get(col_phone)) if col_phone else (clean_text(row.get(col_phone_secondary)) if col_phone_secondary else None),
                "address": clean_text(row.get(col_address)) if col_address else None,
                "lat": lat,
                "lon": lon,
                "effortHours": effort_hours,
                "eventsAttendedCount": events_attended,
                "referralCount": referrals,
                "education": education,
                "skills": skills,
                "supporterType": normalize_supporter_type(supporter_type, default_type),
            }
        )
    return rows


def nominatim_search(query, limit=5):
    if not query:
        return []
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": limit},
            headers={"User-Agent": "freedom-square-crm-short"},
            timeout=10,
        )
        if response.ok:
            return response.json()
    except Exception:
        return []
    return []


def bulk_upsert_people(rows):
    if not rows:
        return False
    query = """
    UNWIND $rows AS row
    WITH row
    WHERE row.email IS NOT NULL AND trim(row.email) <> ""
    MERGE (p:Person {email: row.email})
    ON CREATE SET p.personId = randomUUID(), p.createdAt = datetime()
    SET p.firstName = row.firstName,
        p.lastName = row.lastName,
        p.gender = row.gender,
        p.age = row.age,
        p.phone = row.phone,
        p.lat = row.lat,
        p.lon = row.lon,
        p.effortHours = coalesce(row.effortHours, p.effortHours),
        p.eventsAttendedCount = coalesce(row.eventsAttendedCount, p.eventsAttendedCount),
        p.referralCount = coalesce(row.referralCount, p.referralCount)
    WITH p, row
    FOREACH (_ IN CASE WHEN row.education IS NULL OR row.education = '' THEN [] ELSE [1] END |
        MERGE (ed:EducationLevel {name: row.education})
        MERGE (p)-[:HAS_EDUCATION]->(ed)
    )
    FOREACH (skill IN coalesce(row.skills, []) |
        MERGE (sk:Skill {name: skill})
        MERGE (p)-[:CAN_CONTRIBUTE_WITH]->(sk)
    )
    MERGE (st:SupporterType {name: coalesce(row.supporterType, 'Supporter')})
    MERGE (p)-[:CLASSIFIED_AS]->(st)
    WITH p, row
    FOREACH (_ IN CASE WHEN row.address IS NULL OR row.address = '' THEN [] ELSE [1] END |
        MERGE (a:Address {fullAddress: row.address})
        ON CREATE SET a.latitude = row.lat, a.longitude = row.lon
        ON MATCH SET a.latitude = coalesce(row.lat, a.latitude),
                    a.longitude = coalesce(row.lon, a.longitude)
        MERGE (p)-[:LIVES_AT]->(a)
    )
    """
    return run_write(query, {"rows": rows})


def classify_group(types):
    for value in types or []:
        if value and "member" in str(value).lower():
            return "Member"
    return "Supporter"


def _education_score(level):
    if not level:
        return 0.0
    text = str(level).lower()
    if "phd" in text or "doctor" in text:
        return 3.0
    if "master" in text:
        return 2.0
    if "bachelor" in text:
        return 1.0
    if "high" in text:
        return 0.5
    return 0.0


def pick_education(levels):
    best_label = None
    best_score = 0.0
    for level in levels or []:
        score = _education_score(level)
        if score > best_score:
            best_score = score
            best_label = level
    return best_label or "Unspecified", best_score


def calc_rating(effort_score):
    score = max(0.0, float(effort_score or 0))
    if score >= 120:
        return 5
    if score >= 80:
        return 4
    if score >= 40:
        return 3
    if score >= 10:
        return 2
    return 1


def rating_stars(value):
    filled = max(0, min(5, int(value)))
    return "⭐" * filled + "☆" * (5 - filled)


def rating_color(value):
    if value >= 4:
        return [46, 204, 113, 190]
    if value >= 3:
        return [241, 196, 15, 190]
    return [231, 76, 60, 190]


@st.cache_data(ttl=60)
def load_supporter_summary():
    df = run_query(
        """
        MATCH (p:Person)
        OPTIONAL MATCH (p)-[:IS_SUPPORTER]->(s:Supporter)
        OPTIONAL MATCH (p)-[:CLASSIFIED_AS]->(st:SupporterType)
        OPTIONAL MATCH (p)-[:HAS_ACTIVITY]->(a:Activity)
        OPTIONAL MATCH (p)-[r:REGISTERED_FOR]->(:Event)
        OPTIONAL MATCH (p)-[:CAN_CONTRIBUTE_WITH]->(sk:Skill)
        OPTIONAL MATCH (p)-[:HAS_EDUCATION]->(ed:EducationLevel)
        OPTIONAL MATCH (p)<-[:REFERRED_BY]-(refP:Person)
        OPTIONAL MATCH (s)-[:RECRUITED]->(sr:Supporter)
        WITH p, s,
             collect(DISTINCT st.name) AS types,
             collect(DISTINCT ed.name) AS educationLevels,
             count(DISTINCT a) AS activityCount,
             count(DISTINCT r) AS eventJoinCount,
             count(DISTINCT CASE WHEN r.status = 'Attended' THEN r ELSE NULL END) AS eventAttendRelCount,
             collect(DISTINCT sk.name) AS skills,
             count(DISTINCT refP) AS referredCount,
             count(DISTINCT sr) AS recruitedCount
        RETURN
          p.email AS email,
          p.firstName AS firstName,
          p.lastName AS lastName,
          coalesce(p.gender, 'Unspecified') AS gender,
          p.age AS age,
          types,
          activityCount,
          eventJoinCount,
          eventAttendRelCount,
          skills,
          educationLevels,
          coalesce(p.eventsAttendedCount, 0) AS eventAttendProp,
          coalesce(p.referralCount, 0) AS referralProp,
          referredCount,
          recruitedCount,
          coalesce(p.effortHours, p.volunteerHours, s.volunteer_hours, s.volunteerHours, 0) AS effortHours,
          coalesce(p.donationTotal, 0) AS donationTotal
        """,
        silent=True,
    )
    if df.empty:
        return df
    df["types"] = df["types"].apply(lambda v: v or [])
    df["group"] = df["types"].apply(classify_group)
    df["activityCount"] = pd.to_numeric(df["activityCount"], errors="coerce").fillna(0).astype(int)
    df["eventJoinCount"] = pd.to_numeric(df["eventJoinCount"], errors="coerce").fillna(0).astype(int)
    df["eventAttendRelCount"] = pd.to_numeric(df["eventAttendRelCount"], errors="coerce").fillna(0).astype(int)
    df["skills"] = df["skills"].apply(lambda v: v or [])
    df["skillCount"] = df["skills"].apply(lambda v: len([x for x in v if x]))
    df["skillsLabel"] = df["skills"].apply(format_list_label)
    df["eventAttendProp"] = pd.to_numeric(df["eventAttendProp"], errors="coerce").fillna(0).astype(int)
    df["referredCount"] = pd.to_numeric(df["referredCount"], errors="coerce").fillna(0).astype(int)
    df["recruitedCount"] = pd.to_numeric(df["recruitedCount"], errors="coerce").fillna(0).astype(int)
    df["referralProp"] = pd.to_numeric(df["referralProp"], errors="coerce").fillna(0).astype(int)
    df["eventAttendCount"] = df["eventAttendRelCount"] + df["eventAttendProp"]
    df["referralCount"] = df["referredCount"] + df["recruitedCount"] + df["referralProp"]
    df["joinCount"] = df["activityCount"] + df["eventJoinCount"]
    df["effortHours"] = pd.to_numeric(df["effortHours"], errors="coerce").fillna(0.0)
    df["donationTotal"] = pd.to_numeric(df["donationTotal"], errors="coerce").fillna(0.0)
    education_values = df["educationLevels"].apply(pick_education)
    df["educationLevel"] = education_values.apply(lambda value: value[0])
    df["educationScore"] = education_values.apply(lambda value: value[1])
    df["effortScore"] = df["effortHours"] + df["eventAttendCount"] + df["referralCount"]
    df["rating"] = df["effortScore"].apply(calc_rating)
    df["ratingStars"] = df["rating"].apply(rating_stars)
    full_name = (df["firstName"].fillna("") + " " + df["lastName"].fillna("")).str.strip()
    df["fullName"] = full_name.mask(full_name == "", df["email"])
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df


@st.cache_data(ttl=60)
def load_map_data():
    df = run_query(
        """
        MATCH (p:Person)
        OPTIONAL MATCH (p)-[:LIVES_AT]->(a:Address)
        WITH p,
             coalesce(p.lat, a.latitude) AS lat,
             coalesce(p.lon, a.longitude) AS lon
        WHERE lat IS NOT NULL AND lon IS NOT NULL
        OPTIONAL MATCH (p)-[:IS_SUPPORTER]->(s:Supporter)
        OPTIONAL MATCH (p)-[:CLASSIFIED_AS]->(st:SupporterType)
        OPTIONAL MATCH (p)-[:HAS_ACTIVITY]->(a:Activity)
        OPTIONAL MATCH (p)-[r:REGISTERED_FOR]->(:Event)
        OPTIONAL MATCH (p)-[:CAN_CONTRIBUTE_WITH]->(sk:Skill)
        OPTIONAL MATCH (p)-[:HAS_EDUCATION]->(ed:EducationLevel)
        OPTIONAL MATCH (p)<-[:REFERRED_BY]-(refP:Person)
        OPTIONAL MATCH (s)-[:RECRUITED]->(sr:Supporter)
        WITH p, s, lat, lon,
             collect(DISTINCT st.name) AS types,
             collect(DISTINCT ed.name) AS educationLevels,
             count(DISTINCT a) AS activityCount,
             count(DISTINCT r) AS eventJoinCount,
             count(DISTINCT CASE WHEN r.status = 'Attended' THEN r ELSE NULL END) AS eventAttendRelCount,
             collect(DISTINCT sk.name) AS skills,
             count(DISTINCT refP) AS referredCount,
             count(DISTINCT sr) AS recruitedCount
        RETURN
          lat,
          lon,
          p.email AS email,
          p.firstName AS firstName,
          p.lastName AS lastName,
          coalesce(p.gender, 'Unspecified') AS gender,
          types,
          activityCount,
          eventJoinCount,
          eventAttendRelCount,
          skills,
          educationLevels,
          coalesce(p.eventsAttendedCount, 0) AS eventAttendProp,
          coalesce(p.referralCount, 0) AS referralProp,
          referredCount,
          recruitedCount,
          coalesce(p.effortHours, p.volunteerHours, s.volunteer_hours, s.volunteerHours, 0) AS effortHours,
          coalesce(p.donationTotal, 0) AS donationTotal
        """,
        silent=True,
    )
    if df.empty:
        return df
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df["types"] = df["types"].apply(lambda v: v or [])
    df["group"] = df["types"].apply(classify_group)
    df["activityCount"] = pd.to_numeric(df["activityCount"], errors="coerce").fillna(0).astype(int)
    df["eventJoinCount"] = pd.to_numeric(df["eventJoinCount"], errors="coerce").fillna(0).astype(int)
    df["eventAttendRelCount"] = pd.to_numeric(df["eventAttendRelCount"], errors="coerce").fillna(0).astype(int)
    df["skills"] = df["skills"].apply(lambda v: v or [])
    df["skillCount"] = df["skills"].apply(lambda v: len([x for x in v if x]))
    df["skillsLabel"] = df["skills"].apply(format_list_label)
    df["eventAttendProp"] = pd.to_numeric(df["eventAttendProp"], errors="coerce").fillna(0).astype(int)
    df["referredCount"] = pd.to_numeric(df["referredCount"], errors="coerce").fillna(0).astype(int)
    df["recruitedCount"] = pd.to_numeric(df["recruitedCount"], errors="coerce").fillna(0).astype(int)
    df["referralProp"] = pd.to_numeric(df["referralProp"], errors="coerce").fillna(0).astype(int)
    df["eventAttendCount"] = df["eventAttendRelCount"] + df["eventAttendProp"]
    df["referralCount"] = df["referredCount"] + df["recruitedCount"] + df["referralProp"]
    df["joinCount"] = df["activityCount"] + df["eventJoinCount"]
    df["effortHours"] = pd.to_numeric(df["effortHours"], errors="coerce").fillna(0.0)
    df["donationTotal"] = pd.to_numeric(df["donationTotal"], errors="coerce").fillna(0.0)
    education_values = df["educationLevels"].apply(pick_education)
    df["educationLevel"] = education_values.apply(lambda value: value[0])
    df["educationScore"] = education_values.apply(lambda value: value[1])
    df["effortScore"] = df["effortHours"] + df["eventAttendCount"] + df["referralCount"]
    df["rating"] = df["effortScore"].apply(calc_rating)
    df["ratingStars"] = df["rating"].apply(rating_stars)
    name = (df["firstName"].fillna("") + " " + df["lastName"].fillna("")).str.strip()
    df["name"] = name.mask(name == "", df["email"])
    df["pointSize"] = (6 + df["effortScore"].clip(lower=0) * 0.2).clip(4, 60)
    df["color"] = df["group"].map(
        {"Supporter": [51, 136, 255, 180], "Member": [142, 68, 173, 180]}
    )
    df["color"] = df["color"].apply(
        lambda value: value if isinstance(value, list) else [120, 120, 120, 180]
    )
    df["ratingColor"] = df["rating"].apply(rating_color)
    return df


def answer_chat(question, df_summary):
    text = question.lower().strip()
    if df_summary.empty:
        return "No supporter data available yet.", None

    if "gender" in text:
        counts = (
            df_summary["gender"]
            .fillna("Unspecified")
            .value_counts()
            .rename_axis("gender")
            .reset_index(name="count")
        )
        return "Here is the gender breakdown.", counts

    if "age" in text:
        age_series = df_summary["age"].dropna()
        if age_series.empty:
            return "No age data available.", None
        stats = (
            age_series.describe()
            .loc[["count", "mean", "min", "max"]]
            .round(2)
            .to_frame("value")
            .reset_index()
            .rename(columns={"index": "metric"})
        )
        return "Age summary (count, mean, min, max).", stats

    if "member" in text or "supporter" in text:
        counts = (
            df_summary["group"]
            .value_counts()
            .rename_axis("group")
            .reset_index(name="count")
        )
        return "Supporter vs member counts.", counts

    if "effort" in text or "hours" in text:
        top = (
            df_summary.sort_values("effortScore", ascending=False)
            .head(10)[["fullName", "email", "group", "effortHours", "eventAttendCount", "referralCount", "ratingStars"]]
        )
        return "Top supporters by effort score.", top

    if "top" in text or "active" in text or "join" in text:
        top = (
            df_summary.sort_values("effortScore", ascending=False)
            .head(10)[["fullName", "email", "group", "effortHours", "eventAttendCount", "referralCount", "ratingStars"]]
        )
        return "Top supporters by effort score.", top

    return (
        "Try asking about gender, age, top joiners, or member counts.",
        None,
    )


def sort_people(df, sort_by):
    if df.empty:
        return df
    if sort_by == "Effort score":
        return df.sort_values("effortScore", ascending=False)
    if sort_by == "Effort hours":
        return df.sort_values("effortHours", ascending=False)
    if sort_by == "Join count":
        return df.sort_values("joinCount", ascending=False)
    if sort_by == "Rating":
        return df.sort_values("rating", ascending=False)
    return df.sort_values("fullName")


init_driver()

st.title("Freedom Square CRM (Short Version)")

if driver is None:
    st.error("Missing or invalid Neo4j credentials. Set NEO4J_URI and NEO4J_PASSWORD in .env.")
    st.stop()

tab_intro, tab_supporters, tab_members, tab_map, tab_import = st.tabs(
    ["Introduction", "Supporters", "Members", "Map", "Import/Export"]
)


with tab_intro:
    st.subheader("Introduction")
    st.write("Short CRM view focused on supporters, members, activity, and map insights.")
    df_summary = load_supporter_summary()
    if df_summary.empty:
        st.info("No supporters found.")
    else:
        total_people = len(df_summary)
        total_supporters = int((df_summary["group"] == "Supporter").sum())
        total_members = int((df_summary["group"] == "Member").sum())
        avg_effort = float(df_summary["effortScore"].mean()) if total_people else 0.0
        metrics = st.columns(4)
        metrics[0].metric("Total people", f"{total_people:,}")
        metrics[1].metric("Supporters", f"{total_supporters:,}")
        metrics[2].metric("Members", f"{total_members:,}")
        metrics[3].metric("Avg effort score", f"{avg_effort:.1f}")

        st.markdown("### Statistics")
        group_counts = (
            df_summary["group"]
            .value_counts()
            .rename_axis("group")
            .reset_index(name="count")
        )
        group_chart = (
            alt.Chart(group_counts)
            .mark_bar()
            .encode(x="group:N", y="count:Q", tooltip=["group:N", "count:Q"])
        )

        gender_counts = (
            df_summary["gender"]
            .fillna("Unspecified")
            .value_counts()
            .rename_axis("gender")
            .reset_index(name="count")
        )
        gender_chart = (
            alt.Chart(gender_counts)
            .mark_bar()
            .encode(x="gender:N", y="count:Q", tooltip=["gender:N", "count:Q"])
        )

        rating_counts = (
            df_summary["rating"]
            .value_counts()
            .sort_index()
            .rename_axis("rating")
            .reset_index(name="count")
        )
        rating_chart = (
            alt.Chart(rating_counts)
            .mark_bar()
            .encode(x="rating:O", y="count:Q", tooltip=["rating:O", "count:Q"])
        )

        stat_cols = st.columns(2)
        with stat_cols[0]:
            st.markdown("**Supporter vs Member**")
            st.altair_chart(group_chart, use_container_width=True)
        with stat_cols[1]:
            st.markdown("**Gender distribution**")
            st.altair_chart(gender_chart, use_container_width=True)

        stat_cols = st.columns(2)
        with stat_cols[0]:
            st.markdown("**Rating distribution**")
            st.altair_chart(rating_chart, use_container_width=True)
        with stat_cols[1]:
            st.markdown("**Age distribution**")
            age_df = df_summary.dropna(subset=["age"])
            if age_df.empty:
                st.caption("No age data available.")
            else:
                age_chart = (
                    alt.Chart(age_df)
                    .mark_bar()
                    .encode(
                        alt.X("age:Q", bin=alt.Bin(maxbins=12)),
                        y="count()",
                        tooltip=["count()"],
                    )
                )
                st.altair_chart(age_chart, use_container_width=True)

        st.markdown("**Effort score summary**")
        effort_stats = (
            df_summary["effortScore"]
            .describe()
            .loc[["mean", "min", "max"]]
            .round(2)
            .to_frame("value")
            .reset_index()
            .rename(columns={"index": "metric"})
        )
        st.dataframe(effort_stats, use_container_width=True)

    st.markdown("---")
    st.subheader("Chatbox")
    st.caption("Ask about supporters, members, gender, age, or top joiners.")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("table") is not None:
                st.dataframe(message["table"], use_container_width=True)

    prompt = st.chat_input("Type your question")
    if prompt:
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        df_summary = load_supporter_summary()
        response, table = answer_chat(prompt, df_summary)
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": response, "table": table}
        )
        with st.chat_message("assistant"):
            st.markdown(response)
            if table is not None:
                st.dataframe(table, use_container_width=True)


with tab_supporters:
    st.subheader("Supporters")
    form_col, list_col = st.columns([1, 2])
    with form_col:
        st.markdown("**Search address**")
        search_query = st.text_input("Search address", key="supporter_address_search")
        if st.button("Find address", key="supporter_address_button"):
            results = nominatim_search(search_query)
            st.session_state["supporter_address_results"] = results
        if "supporter_address_results" not in st.session_state:
            st.session_state["supporter_address_results"] = []

        options = st.session_state["supporter_address_results"]
        labels = [item.get("display_name", "") for item in options]
        selected_label = st.selectbox(
            "Matches", [""] + labels, key="supporter_address_select"
        )
        if selected_label:
            idx = labels.index(selected_label)
            selected = options[idx]
            st.session_state["supporter_address_value"] = selected.get("display_name")
            st.session_state["supporter_lat_value"] = selected.get("lat")
            st.session_state["supporter_lon_value"] = selected.get("lon")

        st.markdown("**New supporter**")
        with st.form("new_supporter_form"):
            email = st.text_input("Email *", key="supporter_email")
            name_cols = st.columns(2)
            first_name = name_cols[0].text_input("First name", key="supporter_first")
            last_name = name_cols[1].text_input("Last name", key="supporter_last")
            gender = st.selectbox(
                "Gender", ["Unspecified", "Female", "Male", "Other"], key="supporter_gender"
            )
            age = st.number_input(
                "Age", min_value=0, max_value=120, value=0, step=1, key="supporter_age"
            )
            phone = st.text_input("Phone", key="supporter_phone")
            effort_hours = st.number_input(
                "Effort hours", min_value=0.0, value=0.0, step=1.0, key="supporter_effort"
            )
            address = st.text_input(
                "Address",
                value=st.session_state.get("supporter_address_value", ""),
                key="supporter_address",
            )
            lat_text = st.text_input(
                "Latitude",
                value=str(st.session_state.get("supporter_lat_value", "") or ""),
                key="supporter_lat",
            )
            lon_text = st.text_input(
                "Longitude",
                value=str(st.session_state.get("supporter_lon_value", "") or ""),
                key="supporter_lon",
            )
            submit = st.form_submit_button("Save supporter")

        if submit:
            cleaned_email = clean_text(email)
            if not cleaned_email:
                st.error("Email is required.")
            else:
                try:
                    lat = float(lat_text) if clean_text(lat_text) else None
                    lon = float(lon_text) if clean_text(lon_text) else None
                except ValueError:
                    st.error("Latitude and longitude must be numbers.")
                    lat = lon = None
                else:
                    payload = {
                        "email": cleaned_email,
                        "firstName": clean_text(first_name),
                        "lastName": clean_text(last_name),
                        "gender": None if gender == "Unspecified" else gender,
                        "age": int(age) if age > 0 else None,
                        "phone": clean_text(phone),
                        "effortHours": float(effort_hours) if effort_hours > 0 else None,
                        "eventsAttendedCount": None,
                        "referralCount": None,
                        "lat": lat,
                        "lon": lon,
                        "address": clean_text(address),
                        "supporterType": "Supporter",
                    }
                    if upsert_person(payload):
                        load_supporter_summary.clear()
                        load_map_data.clear()
                        st.success("Supporter saved.")

    with list_col:
        df_summary = load_supporter_summary()
        supporters = df_summary[df_summary["group"] == "Supporter"] if not df_summary.empty else df_summary
        sort_by = st.selectbox(
            "Sort supporters by",
            ["Effort score", "Effort hours", "Join count", "Rating", "Name (A-Z)"],
            key="supporter_sort",
        )
        supporters = sort_people(supporters, sort_by)
        if supporters.empty:
            st.info("No supporters found.")
        else:
            display_df = supporters[
                [
                    "fullName",
                    "email",
                    "effortHours",
                    "eventAttendCount",
                    "referralCount",
                    "effortScore",
                    "joinCount",
                    "skillCount",
                    "educationLevel",
                    "ratingStars",
                    "gender",
                    "age",
                ]
            ].rename(
                columns={
                    "fullName": "Name",
                    "email": "Email",
                    "effortHours": "Effort Hours",
                    "eventAttendCount": "Events Attended",
                    "referralCount": "Referrals",
                    "effortScore": "Effort Score",
                    "joinCount": "Joined",
                    "skillCount": "Skills",
                    "educationLevel": "Education",
                    "ratingStars": "Rating",
                    "gender": "Gender",
                    "age": "Age",
                }
            )
            st.dataframe(display_df, use_container_width=True)


with tab_members:
    st.subheader("Members")
    form_col, list_col = st.columns([1, 2])
    with form_col:
        st.markdown("**Search address**")
        search_query = st.text_input("Search address", key="member_address_search")
        if st.button("Find address", key="member_address_button"):
            results = nominatim_search(search_query)
            st.session_state["member_address_results"] = results
        if "member_address_results" not in st.session_state:
            st.session_state["member_address_results"] = []

        options = st.session_state["member_address_results"]
        labels = [item.get("display_name", "") for item in options]
        selected_label = st.selectbox(
            "Matches", [""] + labels, key="member_address_select"
        )
        if selected_label:
            idx = labels.index(selected_label)
            selected = options[idx]
            st.session_state["member_address_value"] = selected.get("display_name")
            st.session_state["member_lat_value"] = selected.get("lat")
            st.session_state["member_lon_value"] = selected.get("lon")

        st.markdown("**New member**")
        with st.form("new_member_form"):
            email = st.text_input("Email *", key="member_email")
            name_cols = st.columns(2)
            first_name = name_cols[0].text_input("First name", key="member_first")
            last_name = name_cols[1].text_input("Last name", key="member_last")
            gender = st.selectbox(
                "Gender", ["Unspecified", "Female", "Male", "Other"], key="member_gender"
            )
            age = st.number_input(
                "Age", min_value=0, max_value=120, value=0, step=1, key="member_age"
            )
            phone = st.text_input("Phone", key="member_phone")
            effort_hours = st.number_input(
                "Effort hours", min_value=0.0, value=0.0, step=1.0, key="member_effort"
            )
            address = st.text_input(
                "Address",
                value=st.session_state.get("member_address_value", ""),
                key="member_address",
            )
            lat_text = st.text_input(
                "Latitude",
                value=str(st.session_state.get("member_lat_value", "") or ""),
                key="member_lat",
            )
            lon_text = st.text_input(
                "Longitude",
                value=str(st.session_state.get("member_lon_value", "") or ""),
                key="member_lon",
            )
            submit = st.form_submit_button("Save member")

        if submit:
            cleaned_email = clean_text(email)
            if not cleaned_email:
                st.error("Email is required.")
            else:
                try:
                    lat = float(lat_text) if clean_text(lat_text) else None
                    lon = float(lon_text) if clean_text(lon_text) else None
                except ValueError:
                    st.error("Latitude and longitude must be numbers.")
                    lat = lon = None
                else:
                    payload = {
                        "email": cleaned_email,
                        "firstName": clean_text(first_name),
                        "lastName": clean_text(last_name),
                        "gender": None if gender == "Unspecified" else gender,
                        "age": int(age) if age > 0 else None,
                        "phone": clean_text(phone),
                        "effortHours": float(effort_hours) if effort_hours > 0 else None,
                        "eventsAttendedCount": None,
                        "referralCount": None,
                        "lat": lat,
                        "lon": lon,
                        "address": clean_text(address),
                        "supporterType": "Member",
                    }
                    if upsert_person(payload):
                        load_supporter_summary.clear()
                        load_map_data.clear()
                        st.success("Member saved.")

    with list_col:
        df_summary = load_supporter_summary()
        members = df_summary[df_summary["group"] == "Member"] if not df_summary.empty else df_summary
        sort_by = st.selectbox(
            "Sort members by",
            ["Effort score", "Effort hours", "Join count", "Rating", "Name (A-Z)"],
            key="member_sort",
        )
        members = sort_people(members, sort_by)
        if members.empty:
            st.info("No members found.")
        else:
            display_df = members[
                [
                    "fullName",
                    "email",
                    "effortHours",
                    "eventAttendCount",
                    "referralCount",
                    "effortScore",
                    "joinCount",
                    "skillCount",
                    "educationLevel",
                    "ratingStars",
                    "gender",
                    "age",
                ]
            ].rename(
                columns={
                    "fullName": "Name",
                    "email": "Email",
                    "effortHours": "Effort Hours",
                    "eventAttendCount": "Events Attended",
                    "referralCount": "Referrals",
                    "effortScore": "Effort Score",
                    "joinCount": "Joined",
                    "skillCount": "Skills",
                    "educationLevel": "Education",
                    "ratingStars": "Rating",
                    "gender": "Gender",
                    "age": "Age",
                }
            )
            st.dataframe(display_df, use_container_width=True)


with tab_map:
    st.subheader("Map")
    df_geo = load_map_data()
    if df_geo.empty:
        st.info("No supporters with latitude/longitude found.")
    else:
        sidebar_col, map_col = st.columns([1, 4])
        with sidebar_col:
            st.markdown("**Filters**")
            show_supporters = st.checkbox("Show supporters", value=True)
            show_members = st.checkbox("Show members", value=True)

            education_options = sorted(
                [
                    value
                    for value in df_geo["educationLevel"].dropna().unique().tolist()
                    if str(value).strip() and str(value).lower() != "unspecified"
                ]
            )
            selected_education = st.multiselect(
                "Education", education_options, default=[]
            )
            skill_options = sorted(
                {
                    str(skill).strip()
                    for skills in df_geo["skills"]
                    for skill in (skills or [])
                    if str(skill).strip()
                }
            )
            selected_skills = st.multiselect("Skills", skill_options, default=[])
            gender_options = sorted(
                [
                    value
                    for value in df_geo["gender"].dropna().unique().tolist()
                    if str(value).strip() and str(value).lower() != "unspecified"
                ]
            )
            selected_gender = st.multiselect("Gender", gender_options, default=[])
            min_effort = st.number_input(
                "Minimum effort hours", min_value=0.0, value=0.0, step=1.0
            )
            min_events = st.number_input(
                "Minimum events attended", min_value=0, value=0, step=1
            )
            min_referrals = st.number_input(
                "Minimum referrals", min_value=0, value=0, step=1
            )

        df_filtered = df_geo.copy()
        if not show_supporters:
            df_filtered = df_filtered[df_filtered["group"] != "Supporter"]
        if not show_members:
            df_filtered = df_filtered[df_filtered["group"] != "Member"]
        if selected_education:
            df_filtered = df_filtered[df_filtered["educationLevel"].isin(selected_education)]
        if selected_skills:
            df_filtered = df_filtered[
                df_filtered["skills"].apply(
                    lambda skills: any(skill in (skills or []) for skill in selected_skills)
                )
            ]
        if selected_gender:
            df_filtered = df_filtered[df_filtered["gender"].isin(selected_gender)]
        if min_effort > 0:
            df_filtered = df_filtered[df_filtered["effortHours"] >= min_effort]
        if min_events > 0:
            df_filtered = df_filtered[df_filtered["eventAttendCount"] >= min_events]
        if min_referrals > 0:
            df_filtered = df_filtered[df_filtered["referralCount"] >= min_referrals]

        with map_col:
            if df_filtered.empty:
                st.info("No map points for the selected filter.")
            else:
                st.caption("Hover points for details. Use the console to open a small profile.")
                legend_cols = st.columns(2)
                legend_cols[0].markdown(
                    "<div style='display:flex;align-items:center;gap:6px;'>"
                    "<span style='width:14px;height:14px;background:#3388ff;display:inline-block;border-radius:3px;'></span>"
                    "<span style='font-size:12px;'>Supporter</span></div>",
                    unsafe_allow_html=True,
                )
                legend_cols[1].markdown(
                    "<div style='display:flex;align-items:center;gap:6px;'>"
                    "<span style='width:14px;height:14px;background:#8e44ad;display:inline-block;border-radius:3px;'></span>"
                    "<span style='font-size:12px;'>Member</span></div>",
                    unsafe_allow_html=True,
                )

                scatter = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_filtered,
                    get_position=["lon", "lat"],
                    get_fill_color="color",
                    get_radius="pointSize",
                    pickable=True,
                )

                view_state = pdk.ViewState(
                    latitude=df_filtered["lat"].mean(),
                    longitude=df_filtered["lon"].mean(),
                    zoom=11,
                    pitch=20,
                )

                layers = [scatter]

                deck = pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    tooltip={
                        "text": "Name: {name}\nGroup: {group}\nEffort Score: {effortScore}\nEffort Hours: {effortHours}\nEvents Attended: {eventAttendCount}\nReferrals: {referralCount}\nSkills: {skillsLabel}\nEducation: {educationLevel}\nRating: {ratingStars}"
                    },
                )
                st.pydeck_chart(deck, use_container_width=True)



with tab_import:
    st.subheader("Import / Export")
    st.markdown("**Import supporters or members (CSV)**")
    upload = st.file_uploader("Upload CSV", type=["csv"])
    default_type = st.selectbox(
        "Default type (if missing)", ["Supporter", "Member", "Both"], key="import_default_type"
    )
    default_type_value = default_type
    if default_type == "Both":
        st.caption("Rows without a supporter_type will default to Supporter.")
        default_type_value = "Supporter"

    if upload is not None:
        try:
            df_upload = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            df_upload = pd.DataFrame()

        if not df_upload.empty:
            st.caption("Preview")
            st.dataframe(df_upload.head(10), use_container_width=True)

            if st.button("Import CSV"):
                rows = _build_import_rows(df_upload, default_type_value)
                if not rows:
                    st.error("No valid rows found. Ensure the CSV has an email column.")
                elif bulk_upsert_people(rows):
                    load_supporter_summary.clear()
                    load_map_data.clear()
                    st.success(f"Imported {len(rows)} rows.")

    st.markdown("---")
    st.markdown("**Export current data (CSV)**")
    df_export = load_supporter_summary()
    if df_export.empty:
        st.info("No data available to export.")
    else:
        export_df = df_export[
            [
                "fullName",
                "email",
                "group",
                "effortScore",
                "effortHours",
                "eventAttendCount",
                "referralCount",
                "joinCount",
                "skillCount",
                "educationLevel",
                "ratingStars",
                "gender",
                "age",
            ]
        ].rename(
            columns={
                "fullName": "name",
                "email": "email",
                "group": "group",
                "effortScore": "effort_score",
                "effortHours": "effort_hours",
                "eventAttendCount": "events_attended",
                "referralCount": "referrals",
                "joinCount": "joined",
                "skillCount": "skills_count",
                "educationLevel": "education",
                "ratingStars": "rating",
                "gender": "gender",
                "age": "age",
            }
        )
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="supporters_export.csv",
            mime="text/csv",
        )



