import os
from datetime import datetime
from io import StringIO

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import streamlit.components.v1 as components
import altair as alt
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv
import pydeck as pdk
import requests
from streamlit_js_eval import get_geolocation
from pyvis.network import Network

# ----------------------------------
# Load environment
# ----------------------------------
load_dotenv()

def get_config(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets.get(key, default)
    except StreamlitSecretNotFoundError:
        pass
    value = os.getenv(key)
    return value if value is not None else default

NEO4J_URI = get_config("NEO4J_URI")
NEO4J_USER = (
    get_config("NEO4J_USER")
    or get_config("NEO4J_USERNAME")
    or "neo4j"
)
NEO4J_PASSWORD = get_config("NEO4J_PASSWORD")
NEO4J_DATABASE = get_config("NEO4J_DATABASE", "neo4j")
NATION_SLUG = get_config("NATION_SLUG", "default")

if not NEO4J_URI or not NEO4J_PASSWORD:
    st.error("⚠️ Missing Neo4j credentials. Set NEO4J_URI and NEO4J_PASSWORD in Streamlit secrets or .env.")

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
    
    # Allow explicit override for sandbox/Aura connections if needed
    uri = NEO4J_URI
    override_uri = get_config("NEO4J_OVERRIDE_URI")
    if override_uri:
        uri = override_uri

    try:
        driver = GraphDatabase.driver(uri, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
        # Test connection
        with driver.session(database=NEO4J_DATABASE) as session:
            _session_execute_read(session, lambda tx: list(tx.run("RETURN 1")))
        return True
    except Exception as e:
        error_str = str(e)
        # Check for authentication rate limit
        if "AuthenticationRateLimit" in error_str or "authentication details too many times" in error_str:
            _auth_rate_limited = True
            st.error("🔒 Neo4j authentication rate limit reached. Please wait a few minutes before trying again, or reset your Neo4j password.")
            driver = None
            return False
        st.error(f"Could not initialize Neo4j driver: {e}")
        driver = None
        return False

def apply_runtime_config():
    global NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
    NEO4J_URI = st.session_state.get("neo4j_uri", NEO4J_URI)
    NEO4J_USER = st.session_state.get("neo4j_user", NEO4J_USER)
    NEO4J_PASSWORD = st.session_state.get("neo4j_password", NEO4J_PASSWORD)
    NEO4J_DATABASE = st.session_state.get("neo4j_database", NEO4J_DATABASE)

def reset_driver():
    global driver, _auth_rate_limited
    if driver is not None:
        try:
            driver.close()
        except Exception:
            pass
    driver = None
    _auth_rate_limited = False
    apply_runtime_config()
    return init_driver()

init_driver()

# ----------------------------------
# Helpers
# ----------------------------------
def yes_no_to_bool(v):
    if pd.isna(v):
        return False
    return str(v).strip().lower() in ["yes", "true", "1", "დიახ"]

def split_list(v):
    if pd.isna(v):
        return []
    return [x.strip() for x in str(v).split(",") if x.strip()]

def clean_text(value):
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None

def normalize_str_list(values):
    cleaned = []
    for v in values or []:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            cleaned.append(s)
    return sorted(set(cleaned))

def combine_date_time(date_val, time_val):
    if not date_val:
        return None
    if time_val is None:
        return datetime.combine(date_val, datetime.min.time())
    try:
        return datetime.combine(date_val, time_val)
    except Exception:
        return None

def to_iso(dt_value):
    if not dt_value:
        return None
    try:
        return dt_value.isoformat(timespec="seconds")
    except Exception:
        return None

def get_distinct_values(label, prop="name"):
    if driver is None:
        return []
    query = f"""
    MATCH (n:{label})
    WHERE n.{prop} IS NOT NULL
    RETURN DISTINCT n.{prop} AS value
    ORDER BY value
    """
    df = run_query(query, silent=True)
    if df.empty or "value" not in df.columns:
        return []
    return [str(v) for v in df["value"].dropna().tolist() if str(v).strip()]

def load_schema_markdown():
    schema_path = os.path.join(os.path.dirname(__file__), "GRAPH_SCHEMA.md")
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def ensure_nation(slug):
    if driver is None:
        return False
    return run_write("""
    MERGE (n:Nation {slug: $slug})
    ON CREATE SET n.name = $name, n.createdAt = datetime()
    """, {"slug": slug, "name": "Default Nation"})

def get_nation(slug):
    if driver is None:
        return None
    df = run_query("""
    MATCH (n:Nation {slug: $slug})
    RETURN n
    """, {"slug": slug}, silent=True)
    if df.empty:
        return None
    node = df.iloc[0].get("n", {})
    return node if isinstance(node, dict) else None

def pick_first(row, candidates):
    """Return the first existing column value from candidates or None."""
    for name in candidates:
        if name in row and not pd.isna(row[name]):
            return row[name]
    return None

def hex_to_rgb(hex_color, alpha=200):
    """Convert #RRGGBB or #RRGGBBAA to [r, g, b, a]."""
    if not isinstance(hex_color, str) or not hex_color.startswith("#"):
        return [51, 136, 255, alpha]
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return [r, g, b, alpha]
    if len(h) == 8:
        r, g, b, a = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16)
        return [r, g, b, a]
    return [51, 136, 255, alpha]

def nominatim_search(query, limit=5):
    if not query:
        return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": limit},
            headers={"User-Agent": "freedom-square-crm"}
        )
        if r.ok:
            return r.json()
    except Exception:
        return []
    return []

def nominatim_reverse(lat, lon):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 18},
            headers={"User-Agent": "freedom-square-crm"}
        )
        if r.ok:
            return r.json()
    except Exception:
        return None
    return None

def _run_read(tx, query, params):
    result = tx.run(query, params or {})
    return [r.data() for r in result]


def run_query(query, params=None, silent=False):
    global _auth_rate_limited
    if driver is None:
        if not silent:
            st.error("Neo4j driver not available. Check connection settings.")
        return pd.DataFrame()
    if _auth_rate_limited:
        if not silent:
            st.error("🔒 Neo4j authentication rate limit active. Please wait before retrying.")
        return pd.DataFrame()
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            data = _session_execute_read(session, _run_read, query, params)
            return pd.DataFrame(data)
    except Exception as e:
        error_str = str(e)
        # Check for authentication rate limit
        if "AuthenticationRateLimit" in error_str or "authentication details too many times" in error_str:
            _auth_rate_limited = True
            if not silent:
                st.error("🔒 Neo4j authentication rate limit reached. Please wait a few minutes before trying again.")
        elif not silent:
            st.error(f"Neo4j query failed: {e}")
        return pd.DataFrame()

def _run_write(tx, query, params):
    tx.run(query, params or {})


def run_write(query, params=None):
    global _auth_rate_limited
    if driver is None:
        st.error("Neo4j driver not available. Check connection settings.")
        return False
    if _auth_rate_limited:
        st.error("🔒 Neo4j authentication rate limit active. Please wait before retrying.")
        return False
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            _session_execute_write(session, _run_write, query, params)
        return True
    except Exception as e:
        error_str = str(e)
        # Check for authentication rate limit
        if "AuthenticationRateLimit" in error_str or "authentication details too many times" in error_str:
            _auth_rate_limited = True
            st.error("🔒 Neo4j authentication rate limit reached. Please wait a few minutes before trying again.")
        else:
            st.error(f"Neo4j write failed: {e}")
        return False

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="🟣 Freedom Square – Supporter Graph CRM",
    layout="wide"
)

st.title("🟣 Freedom Square – Supporter Graph CRM")
st.caption("Graph-based supporter management with explainable analytics")
st.session_state.setdefault("nation_slug", NATION_SLUG)

tab_dashboard, tab_form, tab_supporters, tab_import, tab_relationships, tab_activities, tab_events, tab_contributions, tab_goals, tab_comms, tab_finance, tab_maps, tab_analytics, tab_settings, tab_schema, tab_chatbot, tab_cleanup = st.tabs([
    "🏠 Dashboard",
    "📝 New Supporter",
    "👥 Supporters",
    "📤 Import/Export",
    "🧭 Relationships",
    "🗒 Activities",
    "📅 Events",
    "💰 Contributions",
    "🎯 Goals & Paths",
    "📣 Communication",
    "💳 Finance",
    "🗺️ Maps",
    "📊 Analytics",
    "⚙️ Settings",
    "🧱 Schema",
    "💬 Chatbot",
    "🧹 Cleanup"
])

# ----------------------------------
# Tab 0 — Dashboard
# ----------------------------------
with tab_dashboard:
    st.subheader("Nation dashboard")
    if driver is None:
        st.warning("Neo4j driver not available. Check connection settings.")
    else:
        slug = st.session_state.get("nation_slug", NATION_SLUG)
        slug_input = st.text_input("Nation slug", value=slug)
        if slug_input.strip() and slug_input.strip() != slug:
            st.session_state["nation_slug"] = slug_input.strip()
            slug = slug_input.strip()

        ensure_nation(slug)
        nation = get_nation(slug) or {}

        df_overview = run_query("""
        MATCH (p:Person)
        OPTIONAL MATCH (p)-[:CLASSIFIED_AS]->(st:SupporterType)
        WITH p, collect(DISTINCT st.name) AS types
        RETURN count(p) AS total,
               sum(CASE WHEN any(t IN types WHERE toLower(t) CONTAINS 'volunteer') THEN 1 ELSE 0 END) AS volunteers,
               sum(CASE WHEN any(t IN types WHERE toLower(t) CONTAINS 'donor') THEN 1 ELSE 0 END) AS donors
        """, silent=True)

        total_supporters = int(df_overview["total"].iloc[0]) if not df_overview.empty else 0
        total_volunteers = int(df_overview["volunteers"].iloc[0]) if not df_overview.empty else 0
        total_donors = int(df_overview["donors"].iloc[0]) if not df_overview.empty else 0

        df_direct = run_query("""
        MATCH (p:Person)
        RETURN sum(coalesce(p.donationTotal, 0)) AS directTotal
        """, silent=True)
        df_rel = run_query("""
        MATCH ()-[d:DONATED_TO]->()
        RETURN sum(coalesce(d.amount, 0)) AS relTotal
        """, silent=True)
        df_contrib = run_query("""
        MATCH ()-[:MADE_CONTRIBUTION]->(c:Contribution)
        RETURN sum(coalesce(c.amount, 0)) AS contribTotal
        """, silent=True)
        direct_total = float(df_direct["directTotal"].iloc[0]) if not df_direct.empty and df_direct["directTotal"].iloc[0] is not None else 0.0
        rel_total = float(df_rel["relTotal"].iloc[0]) if not df_rel.empty and df_rel["relTotal"].iloc[0] is not None else 0.0
        contrib_total = float(df_contrib["contribTotal"].iloc[0]) if not df_contrib.empty and df_contrib["contribTotal"].iloc[0] is not None else 0.0

        df_counts = run_query("""
        MATCH (n:Nation {slug: $slug})
        OPTIONAL MATCH (n)-[:HAS_BROADCASTER]->(b:Broadcaster)
        OPTIONAL MATCH (n)-[:HAS_GOAL]->(g:Goal)
        OPTIONAL MATCH (n)-[:HAS_PATH]->(p:Path)
        RETURN count(DISTINCT b) AS broadcasters,
               count(DISTINCT g) AS goals,
               count(DISTINCT p) AS paths
        """, {"slug": slug}, silent=True)
        broadcasters_count = int(df_counts["broadcasters"].iloc[0]) if not df_counts.empty else 0
        goals_count = int(df_counts["goals"].iloc[0]) if not df_counts.empty else 0
        paths_count = int(df_counts["paths"].iloc[0]) if not df_counts.empty else 0

        metric_cols = st.columns(4)
        metric_cols[0].metric("Total supporters", f"{total_supporters:,}")
        metric_cols[1].metric("Volunteers", f"{total_volunteers:,}")
        metric_cols[2].metric("Donors", f"{total_donors:,}")
        metric_cols[3].metric("Donation total", f"{(direct_total + rel_total + contrib_total):,.2f}")

        metric_cols2 = st.columns(3)
        metric_cols2[0].metric("Broadcasters", f"{broadcasters_count:,}")
        metric_cols2[1].metric("Goals", f"{goals_count:,}")
        metric_cols2[2].metric("Paths", f"{paths_count:,}")

        st.markdown("**Nation summary**")
        summary_cols = st.columns(2)
        with summary_cols[0]:
            st.write(f"Name: {nation.get('name', 'Default Nation')}")
            st.write(f"Website: {nation.get('website', '—')}")
            st.write(f"Owner email: {nation.get('ownerEmail', '—')}")
        with summary_cols[1]:
            st.write(f"Contact email: {nation.get('contactEmail', '—')}")
            st.write(f"Phone: {nation.get('phone', '—')}")
            st.write(f"Address: {nation.get('address', '—')}")

        st.markdown("**Setup checklist**")
        with st.form("setup_checklist_form"):
            imported_people = st.checkbox("Imported people", value=bool(nation.get("importedPeople", False)))
            connected_broadcasters = st.checkbox("Connected broadcasters", value=bool(nation.get("connectedBroadcasters", False)))
            added_payment = st.checkbox("Added payment processor", value=bool(nation.get("addedPaymentProcessor", False)))
            contact_billing = st.checkbox("Configured contact & billing", value=bool(nation.get("configuredBilling", False)))
            defined_goals = st.checkbox("Defined goals and paths", value=bool(nation.get("definedGoals", False)))
            save_setup = st.form_submit_button("Save checklist")
            if save_setup:
                run_write("""
                MATCH (n:Nation {slug: $slug})
                SET n.importedPeople = $importedPeople,
                    n.connectedBroadcasters = $connectedBroadcasters,
                    n.addedPaymentProcessor = $addedPaymentProcessor,
                    n.configuredBilling = $configuredBilling,
                    n.definedGoals = $definedGoals,
                    n.updatedAt = datetime()
                """, {
                    "slug": slug,
                    "importedPeople": imported_people,
                    "connectedBroadcasters": connected_broadcasters,
                    "addedPaymentProcessor": added_payment,
                    "configuredBilling": contact_billing,
                    "definedGoals": defined_goals
                })
                st.success("Checklist updated.")

        st.markdown("**Recent supporters**")
        df_recent = run_query("""
        MATCH (p:Person)
        RETURN p.firstName AS firstName,
               p.lastName AS lastName,
               p.email AS email,
               p.createdAt AS createdAt
        ORDER BY p.createdAt DESC
        LIMIT 10
        """, silent=True)
        if df_recent.empty:
            st.info("No supporters found yet.")
        else:
            st.dataframe(df_recent)

# ----------------------------------
# Tab 1 — Form
# ----------------------------------
with tab_form:
    st.subheader("Add a supporter (single entry form)")
    st.caption("Fill the fields, submit, preview, then add to Neo4j.")

    default_areas = ["Field Organizing", "Events", "Fundraising", "Communications", "Tech", "Policy"]
    default_skills = ["Communication", "Fundraising", "Data", "Design", "Engineering", "Organizing"]
    supporter_types = ["Interested", "Volunteer", "Donor", "Active Supporter"]
    existing_tags = get_distinct_values("Tag")

    # Session defaults for address/lat/lon
    st.session_state.setdefault("form_address", "")
    st.session_state.setdefault("form_lat", 0.0)
    st.session_state.setdefault("form_lon", 0.0)
    st.session_state.setdefault("addr_results", [])

    with st.form("supporter_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            first_name = st.text_input("First Name")
            email = st.text_input("Email")
            phone = st.text_input("Phone", value="")
            gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
            age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
            supporter_type = st.selectbox("Supporter Type", supporter_types, index=0)
            time_availability = st.selectbox("Time Availability", ["", "Weekends", "Evenings", "Full-time", "Ad-hoc"])
            agrees = st.checkbox("Agrees with Manifesto", value=False)
            interested_membership = st.checkbox("Interested in Party Membership", value=False)
            facebook_member = st.checkbox("Facebook Group Member", value=False)
            referrer_email = st.text_input("Referred By (email)", value="")
        with col_b:
            last_name = st.text_input("Last Name")
            search_query = st.text_input("Search address (free OSM)", key="addr_search_query")
            btn_search, btn_geo = st.columns(2)
            with btn_search:
                if st.form_submit_button("🔍 Search address", type="secondary", help="Uses OpenStreetMap Nominatim"):
                    st.session_state.addr_results = nominatim_search(search_query, limit=5)
            with btn_geo:
                if st.form_submit_button("📍 Use current location", type="secondary", help="Requires browser location permission"):
                    loc = get_geolocation()
                    if loc and "coords" in loc:
                        lat = loc["coords"]["latitude"]
                        lon = loc["coords"]["longitude"]
                        st.session_state.form_lat = lat
                        st.session_state.form_lon = lon
                        rev = nominatim_reverse(lat, lon)
                        if rev:
                            st.session_state.form_address = rev.get("display_name", st.session_state.form_address)
                        st.success("Location captured.")
                    else:
                        st.warning("Location not available (permission denied or unsupported).")

            if st.session_state.addr_results:
                options = [f"{o['display_name']} ({o['lat']}, {o['lon']})" for o in st.session_state.addr_results]
                choice = st.selectbox("Select a suggested address", options)
                sel = st.session_state.addr_results[options.index(choice)]
                st.session_state.form_address = sel["display_name"]
                st.session_state.form_lat = float(sel["lat"])
                st.session_state.form_lon = float(sel["lon"])
                st.info("Address selected and lat/lon filled.")

            address = st.text_input("Address", key="form_address")
            latitude = st.number_input("Latitude", value=st.session_state.get("form_lat", 0.0), format="%.6f", key="form_lat")
            longitude = st.number_input("Longitude", value=st.session_state.get("form_lon", 0.0), format="%.6f", key="form_lon")
            involvement = st.multiselect("Preferred Areas of Involvement", default_areas, default=[])
            skills_sel = st.multiselect("Skills", default_skills, default=[])
            donation_total = st.number_input("Donation Total (optional)", min_value=0.0, value=0.0, step=10.0)
            tags_selected = st.multiselect("Tags (existing)", existing_tags, default=[])
            tags_custom = st.text_input("Add tags (comma separated)")
        tags_combined = normalize_str_list(tags_selected + split_list(tags_custom))
        about = st.text_area("About / Motivation", value="")

        save = st.form_submit_button("💾 Save to Neo4j and preview")

        if save:
            row = {
                "First Name": first_name,
                "Last Name": last_name,
                "Email": email,
                "Phone": phone,
                "Address": address,
                "Latitude": latitude if latitude != 0.0 else None,
                "Longitude": longitude if longitude != 0.0 else None,
                "Gender": gender,
                "Age": age if age else None,
                "About You / Motivation": about,
                "Time Availability": time_availability,
                "Agrees with Manifesto": "Yes" if agrees else "No",
                "Interested in Party Membership": "Yes" if interested_membership else "No",
                "Facebook Group Member": "Yes" if facebook_member else "No",
                "Supporter Type": supporter_type,
                "Preferred Areas of Involvement": ", ".join(involvement),
                "How You Can Help": ", ".join(skills_sel),
                "Tags": ", ".join(tags_combined),
                "Referred By Email": referrer_email,
                "Donation Total": donation_total if donation_total else None,
            }
            st.success("Preview new supporter")
            st.dataframe(pd.DataFrame([row]))

            if driver is None:
                st.error("Neo4j driver not available. Check connection settings.")
            elif not email.strip():
                st.error("Email is required to upsert the supporter.")
            else:
                lat_val = float(st.session_state.get("form_lat", latitude)) if st.session_state.get("form_lat", latitude) not in [0.0, None] else None
                lon_val = float(st.session_state.get("form_lon", longitude)) if st.session_state.get("form_lon", longitude) not in [0.0, None] else None

                ok = run_write("""
                MERGE (p:Person {email: $email})
                ON CREATE SET
                  p.personId = randomUUID(),
                  p.createdAt = datetime(),
                  p.firstName = $firstName,
                  p.lastName = $lastName,
                  p.phone = $phone,
                  p.gender = $gender,
                  p.age = $age,
                  p.about = $about,
                  p.timeAvailability = $timeAvailability,
                  p.agreesWithManifesto = $agreesWithManifesto,
                  p.interestedInMembership = $interestedInMembership,
                  p.facebookGroupMember = $facebookGroupMember,
                  p.lat = $lat,
                  p.lon = $lon,
                  p.donationTotal = $donationTotal
                ON MATCH SET
                  p.firstName = coalesce($firstName, p.firstName),
                  p.lastName = coalesce($lastName, p.lastName),
                  p.phone = coalesce($phone, p.phone),
                  p.gender = coalesce($gender, p.gender),
                  p.age = coalesce($age, p.age),
                  p.about = coalesce($about, p.about),
                  p.timeAvailability = coalesce($timeAvailability, p.timeAvailability),
                  p.agreesWithManifesto = coalesce($agreesWithManifesto, p.agreesWithManifesto),
                  p.interestedInMembership = coalesce($interestedInMembership, p.interestedInMembership),
                  p.facebookGroupMember = coalesce($facebookGroupMember, p.facebookGroupMember),
                  p.lat = coalesce($lat, p.lat),
                  p.lon = coalesce($lon, p.lon),
                  p.donationTotal = coalesce($donationTotal, p.donationTotal)

                WITH p

                MERGE (a:Address {fullAddress: $address})
                ON CREATE SET
                  a.latitude = $lat,
                  a.longitude = $lon
                ON MATCH SET
                  a.latitude = coalesce($lat, a.latitude),
                  a.longitude = coalesce($lon, a.longitude)

                MERGE (p)-[:LIVES_AT]->(a)

                MERGE (st:SupporterType {name: $supporterType})
                MERGE (p)-[:CLASSIFIED_AS]->(st)

                FOREACH (area IN $involvementAreas |
                  MERGE (ia:InvolvementArea {name: area})
                  MERGE (p)-[:INTERESTED_IN]->(ia)
                )

                FOREACH (skill IN $skills |
                  MERGE (sk:Skill {name: skill})
                  MERGE (p)-[:CAN_CONTRIBUTE_WITH]->(sk)
                )

                FOREACH (tag IN $tags |
                  MERGE (t:Tag {name: tag})
                  MERGE (p)-[:HAS_TAG]->(t)
                )

                FOREACH (_ IN CASE
                  WHEN $referrerEmail IS NULL OR $referrerEmail = '' OR $referrerEmail = $email THEN []
                  ELSE [1]
                END |
                  MERGE (ref:Person {email: $referrerEmail})
                  ON CREATE SET ref.personId = randomUUID()
                  MERGE (p)-[:REFERRED_BY]->(ref)
                )
                """, {
                    "email": email,
                    "firstName": first_name,
                    "lastName": last_name,
                    "phone": phone,
                    "address": address,
                    "gender": gender,
                    "age": age if age else None,
                    "about": about,
                    "timeAvailability": time_availability,
                    "agreesWithManifesto": agrees,
                    "interestedInMembership": interested_membership,
                    "facebookGroupMember": facebook_member,
                    "supporterType": supporter_type,
                    "lat": lat_val,
                    "lon": lon_val,
                    "donationTotal": donation_total if donation_total else None,
                    "involvementAreas": involvement,
                    "skills": skills_sel,
                    "tags": tags_combined,
                    "referrerEmail": referrer_email.strip()
                })

                if ok:
                    confirm = run_query("""
                    MATCH (p:Person {email: $email})
                    OPTIONAL MATCH (p)-[:CLASSIFIED_AS]->(st:SupporterType)
                    RETURN p.email AS email, p.firstName AS firstName, p.lastName AS lastName, coalesce(st.name,'Unspecified') AS type
                    """, {"email": email}, silent=True)
                    st.success("Added/updated supporter in Neo4j.")
                    if not confirm.empty:
                        st.dataframe(confirm)

# ----------------------------------
# Tab 1 — Supporters
# ----------------------------------
with tab_supporters:
    st.subheader("Supporter directory")
    st.caption("Filter, export, and edit supporter profiles.")

    if driver is None:
        st.warning("Neo4j driver not available. Check connection settings.")
    else:
        type_options = get_distinct_values("SupporterType")
        if not type_options:
            type_options = ["Interested", "Volunteer", "Donor", "Active Supporter"]
        tag_options = get_distinct_values("Tag")
        area_options = normalize_str_list(get_distinct_values("InvolvementArea") + [
            "Field Organizing", "Events", "Fundraising", "Communications", "Tech", "Policy"
        ])
        skill_options = normalize_str_list(get_distinct_values("Skill") + [
            "Communication", "Fundraising", "Data", "Design", "Engineering", "Organizing"
        ])

        filter_cols = st.columns(4)
        with filter_cols[0]:
            q = st.text_input("Search name or email", value="")
        with filter_cols[1]:
            type_filter = st.multiselect("Supporter Type", type_options, default=[])
        with filter_cols[2]:
            tag_filter = st.multiselect("Tags", tag_options, default=[])
        with filter_cols[3]:
            area_filter = st.multiselect("Involvement Areas", area_options, default=[])

        filter_cols2 = st.columns(3)
        with filter_cols2[0]:
            skill_filter = st.multiselect("Skills", skill_options, default=[])
        with filter_cols2[1]:
            manifesto_choice = st.selectbox("Agrees with Manifesto", ["Any", "Yes", "No", "Unspecified"], index=0)
        with filter_cols2[2]:
            membership_choice = st.selectbox("Membership Interest", ["Any", "Yes", "No", "Unspecified"], index=0)

        manifesto_filter = None
        if manifesto_choice == "Yes":
            manifesto_filter = True
        elif manifesto_choice == "No":
            manifesto_filter = False
        elif manifesto_choice == "Unspecified":
            manifesto_filter = "UNSPECIFIED"

        membership_filter = None
        if membership_choice == "Yes":
            membership_filter = True
        elif membership_choice == "No":
            membership_filter = False
        elif membership_choice == "Unspecified":
            membership_filter = "UNSPECIFIED"

        df_supporters = run_query("""
        MATCH (p:Person)
        OPTIONAL MATCH (p)-[:CLASSIFIED_AS]->(st:SupporterType)
        OPTIONAL MATCH (p)-[:HAS_TAG]->(t:Tag)
        OPTIONAL MATCH (p)-[:INTERESTED_IN]->(ia:InvolvementArea)
        OPTIONAL MATCH (p)-[:CAN_CONTRIBUTE_WITH]->(sk:Skill)
        WITH p,
             collect(DISTINCT st.name) AS types,
             collect(DISTINCT t.name) AS tags,
             collect(DISTINCT ia.name) AS areas,
             collect(DISTINCT sk.name) AS skills
        WHERE ($q IS NULL OR $q = ""
               OR toLower(p.email) CONTAINS toLower($q)
               OR toLower(p.firstName) CONTAINS toLower($q)
               OR toLower(p.lastName) CONTAINS toLower($q))
          AND (size($types) = 0 OR any(t IN types WHERE t IN $types))
          AND (size($tags) = 0 OR any(t IN tags WHERE t IN $tags))
          AND (size($areas) = 0 OR any(a IN areas WHERE a IN $areas))
          AND (size($skills) = 0 OR any(s IN skills WHERE s IN $skills))
          AND (
            $manifestoFilter IS NULL
            OR ($manifestoFilter = "UNSPECIFIED" AND p.agreesWithManifesto IS NULL)
            OR ($manifestoFilter <> "UNSPECIFIED" AND p.agreesWithManifesto = $manifestoFilter)
          )
          AND (
            $membershipFilter IS NULL
            OR ($membershipFilter = "UNSPECIFIED" AND p.interestedInMembership IS NULL)
            OR ($membershipFilter <> "UNSPECIFIED" AND p.interestedInMembership = $membershipFilter)
          )
        RETURN p.email AS email,
               p.firstName AS firstName,
               p.lastName AS lastName,
               p.phone AS phone,
               p.gender AS gender,
               p.age AS age,
               p.timeAvailability AS timeAvailability,
               p.agreesWithManifesto AS agreesWithManifesto,
               p.interestedInMembership AS interestedInMembership,
               types,
               tags,
               areas,
               skills
        ORDER BY p.lastName, p.firstName
        LIMIT 500
        """, {
            "q": q,
            "types": type_filter,
            "tags": tag_filter,
            "areas": area_filter,
            "skills": skill_filter,
            "manifestoFilter": manifesto_filter,
            "membershipFilter": membership_filter
        }, silent=True)

        if df_supporters.empty:
            st.info("No supporters found with the current filters.")
        else:
            st.metric("Supporters found", len(df_supporters))

            display_df = df_supporters.copy()
            display_df["types"] = display_df["types"].apply(lambda v: ", ".join(v) if isinstance(v, list) else "")
            display_df["tags"] = display_df["tags"].apply(lambda v: ", ".join(v) if isinstance(v, list) else "")
            display_df["areas"] = display_df["areas"].apply(lambda v: ", ".join(v) if isinstance(v, list) else "")
            display_df["skills"] = display_df["skills"].apply(lambda v: ", ".join(v) if isinstance(v, list) else "")
            st.dataframe(display_df)

            csv_buffer = StringIO()
            display_df.to_csv(csv_buffer, index=False)
            export_name = f"supporters_export_{datetime.utcnow().strftime('%Y%m%d')}.csv"
            st.download_button(
                "Download filtered CSV",
                csv_buffer.getvalue(),
                file_name=export_name,
                mime="text/csv"
            )

            st.markdown("**Edit supporter**")
            email_options = [e for e in display_df["email"].dropna().tolist() if str(e).strip()]
            selected_email = st.selectbox("Select supporter by email", [""] + email_options)

            if selected_email:
                details = run_query("""
                MATCH (p:Person {email: $email})
                OPTIONAL MATCH (p)-[:CLASSIFIED_AS]->(st:SupporterType)
                OPTIONAL MATCH (p)-[:HAS_TAG]->(t:Tag)
                OPTIONAL MATCH (p)-[:INTERESTED_IN]->(ia:InvolvementArea)
                OPTIONAL MATCH (p)-[:CAN_CONTRIBUTE_WITH]->(sk:Skill)
                OPTIONAL MATCH (p)-[:REFERRED_BY]->(ref:Person)
                WITH p,
                     collect(DISTINCT st.name) AS types,
                     collect(DISTINCT t.name) AS tags,
                     collect(DISTINCT ia.name) AS areas,
                     collect(DISTINCT sk.name) AS skills,
                     collect(DISTINCT ref.email) AS referrers
                RETURN p.firstName AS firstName,
                       p.lastName AS lastName,
                       p.phone AS phone,
                       p.gender AS gender,
                       p.age AS age,
                       p.about AS about,
                       p.timeAvailability AS timeAvailability,
                       p.agreesWithManifesto AS agreesWithManifesto,
                       p.interestedInMembership AS interestedInMembership,
                       p.facebookGroupMember AS facebookGroupMember,
                       p.lat AS lat,
                       p.lon AS lon,
                       p.donationTotal AS donationTotal,
                       types,
                       tags,
                       areas,
                       skills,
                       referrers
                """, {"email": selected_email}, silent=True)

                if details.empty:
                    st.info("No details found for the selected supporter.")
                else:
                    d = details.iloc[0].to_dict()
                    current_type = d.get("types")[0] if isinstance(d.get("types"), list) and d.get("types") else ""
                    current_tags = d.get("tags") if isinstance(d.get("tags"), list) else []
                    current_areas = d.get("areas") if isinstance(d.get("areas"), list) else []
                    current_skills = d.get("skills") if isinstance(d.get("skills"), list) else []
                    current_referrer = d.get("referrers")[0] if isinstance(d.get("referrers"), list) and d.get("referrers") else ""

                    with st.form("edit_supporter_form"):
                        edit_cols = st.columns(2)
                        with edit_cols[0]:
                            gender_options = ["", "Male", "Female", "Other"]
                            gender_value = clean_text(d.get("gender")) or ""
                            if gender_value not in gender_options:
                                gender_value = ""
                            try:
                                age_value = int(float(d.get("age") or 0))
                            except Exception:
                                age_value = 0
                            edit_first = st.text_input("First Name", value=clean_text(d.get("firstName")) or "")
                            edit_last = st.text_input("Last Name", value=clean_text(d.get("lastName")) or "")
                            edit_phone = st.text_input("Phone", value=clean_text(d.get("phone")) or "")
                            edit_gender = st.selectbox("Gender", gender_options, index=gender_options.index(gender_value))
                            edit_age = st.number_input("Age", min_value=0, max_value=120, value=age_value, step=1)
                            edit_type = st.selectbox("Supporter Type", type_options, index=type_options.index(current_type) if current_type in type_options else 0)
                        with edit_cols[1]:
                            time_options = ["", "Weekends", "Evenings", "Full-time", "Ad-hoc"]
                            time_value = clean_text(d.get("timeAvailability")) or ""
                            if time_value not in time_options:
                                time_value = ""
                            edit_time = st.selectbox("Time Availability", time_options, index=time_options.index(time_value))
                            edit_manifesto = st.checkbox("Agrees with Manifesto", value=bool(d.get("agreesWithManifesto")) if d.get("agreesWithManifesto") is not None else False)
                            edit_membership = st.checkbox("Interested in Party Membership", value=bool(d.get("interestedInMembership")) if d.get("interestedInMembership") is not None else False)
                            edit_facebook = st.checkbox("Facebook Group Member", value=bool(d.get("facebookGroupMember")) if d.get("facebookGroupMember") is not None else False)
                            edit_referrer = st.text_input("Referred By (email)", value=clean_text(current_referrer) or "")
                            edit_tags_selected = st.multiselect("Tags (existing)", tag_options, default=current_tags)
                            edit_tags_custom = st.text_input("Add tags (comma separated)", value="")

                        edit_about = st.text_area("About / Motivation", value=clean_text(d.get("about")) or "")
                        edit_areas = st.multiselect("Preferred Areas of Involvement", area_options, default=current_areas)
                        edit_skills = st.multiselect("Skills", skill_options, default=current_skills)

                        replace_relationships = st.checkbox("Replace tags/areas/skills", value=True)
                        save_edit = st.form_submit_button("Update supporter")

                        if save_edit:
                            edit_tags = normalize_str_list(edit_tags_selected + split_list(edit_tags_custom))

                            ok = run_write("""
                            MERGE (p:Person {email: $email})
                            ON CREATE SET p.personId = randomUUID()
                            SET p.firstName = coalesce($firstName, p.firstName),
                                p.lastName = coalesce($lastName, p.lastName),
                                p.phone = coalesce($phone, p.phone),
                                p.gender = coalesce($gender, p.gender),
                                p.age = coalesce($age, p.age),
                                p.about = coalesce($about, p.about),
                                p.timeAvailability = coalesce($timeAvailability, p.timeAvailability),
                                p.agreesWithManifesto = $agreesWithManifesto,
                                p.interestedInMembership = $interestedInMembership,
                                p.facebookGroupMember = $facebookGroupMember
                            """, {
                                "email": selected_email,
                                "firstName": clean_text(edit_first),
                                "lastName": clean_text(edit_last),
                                "phone": clean_text(edit_phone),
                                "gender": clean_text(edit_gender),
                                "age": edit_age if edit_age else None,
                                "about": clean_text(edit_about),
                                "timeAvailability": clean_text(edit_time),
                                "agreesWithManifesto": edit_manifesto,
                                "interestedInMembership": edit_membership,
                                "facebookGroupMember": edit_facebook
                            })

                            if ok:
                                run_write("""
                                MATCH (p:Person {email: $email})
                                OPTIONAL MATCH (p)-[r:CLASSIFIED_AS]->(:SupporterType)
                                DELETE r
                                """, {"email": selected_email})

                                if replace_relationships:
                                    run_write("""
                                    MATCH (p:Person {email: $email})
                                    OPTIONAL MATCH (p)-[t:HAS_TAG]->(:Tag)
                                    OPTIONAL MATCH (p)-[a:INTERESTED_IN]->(:InvolvementArea)
                                    OPTIONAL MATCH (p)-[s:CAN_CONTRIBUTE_WITH]->(:Skill)
                                    DELETE t, a, s
                                    """, {"email": selected_email})

                                run_write("""
                                MATCH (p:Person {email: $email})
                                MERGE (st:SupporterType {name: $supporterType})
                                MERGE (p)-[:CLASSIFIED_AS]->(st)

                                FOREACH (area IN $areas |
                                  MERGE (ia:InvolvementArea {name: area})
                                  MERGE (p)-[:INTERESTED_IN]->(ia)
                                )

                                FOREACH (skill IN $skills |
                                  MERGE (sk:Skill {name: skill})
                                  MERGE (p)-[:CAN_CONTRIBUTE_WITH]->(sk)
                                )

                                FOREACH (tag IN $tags |
                                  MERGE (t:Tag {name: tag})
                                  MERGE (p)-[:HAS_TAG]->(t)
                                )
                                """, {
                                    "email": selected_email,
                                    "supporterType": edit_type,
                                    "areas": edit_areas,
                                    "skills": edit_skills,
                                    "tags": edit_tags
                                })

                                run_write("""
                                MATCH (p:Person {email: $email})
                                OPTIONAL MATCH (p)-[r:REFERRED_BY]->(:Person)
                                DELETE r
                                """, {"email": selected_email})

                                if clean_text(edit_referrer) and clean_text(edit_referrer) != selected_email:
                                    run_write("""
                                    MATCH (p:Person {email: $email})
                                    MERGE (ref:Person {email: $referrer})
                                    ON CREATE SET ref.personId = randomUUID()
                                    MERGE (p)-[:REFERRED_BY]->(ref)
                                    """, {
                                        "email": selected_email,
                                        "referrer": clean_text(edit_referrer)
                                    })

                                st.success("Supporter updated.")

                    with st.expander("Danger zone: delete supporter"):
                        confirm_delete = st.checkbox("I understand this will delete the supporter", value=False)
                        if st.button("Delete supporter", disabled=not confirm_delete):
                            run_write("MATCH (p:Person {email: $email}) DETACH DELETE p", {"email": selected_email})
                            st.warning("Supporter deleted.")

# ----------------------------------
# Tab 3 — Relationships
# ----------------------------------
with tab_relationships:
    st.subheader("Relationship mapping")
    st.caption("Connect supporters to campaigns or to referrers.")

    if driver is None:
        st.warning("Neo4j driver not available. Check connection settings.")
    else:
        rel_cols = st.columns(2)
        with rel_cols[0]:
            st.markdown("**Link supporter to campaign**")
            with st.form("link_campaign_form"):
                rel_email = st.text_input("Supporter email", value="")
                campaign_name = st.text_input("Campaign name", value="")
                rel_type = st.selectbox("Relationship type", ["SUPPORTS", "VOLUNTEERS_FOR", "DONATED_TO"])
                donation_amount = None
                if rel_type == "DONATED_TO":
                    donation_amount = st.number_input("Donation amount", min_value=0.0, value=0.0, step=10.0)
                link_campaign = st.form_submit_button("Link campaign")

                if link_campaign:
                    if not clean_text(rel_email) or not clean_text(campaign_name):
                        st.error("Supporter email and campaign name are required.")
                    else:
                        query = f"""
                        MERGE (p:Person {{email: $email}})
                        ON CREATE SET p.personId = randomUUID()
                        MERGE (c:Campaign {{name: $campaign}})
                        MERGE (p)-[r:{rel_type}]->(c)
                        ON CREATE SET r.createdAt = datetime()
                        """
                        params = {
                            "email": clean_text(rel_email),
                            "campaign": clean_text(campaign_name)
                        }
                        if rel_type == "DONATED_TO":
                            query += "SET r.amount = $amount"
                            params["amount"] = donation_amount
                        ok = run_write(query, params)
                        if ok:
                            st.success("Relationship saved.")

        with rel_cols[1]:
            st.markdown("**Link supporter to referrer**")
            with st.form("link_referral_form"):
                supporter_email = st.text_input("Supporter email", value="", key="ref_supporter_email")
                referrer_email = st.text_input("Referrer email", value="", key="ref_referrer_email")
                link_referrer = st.form_submit_button("Link referrer")

                if link_referrer:
                    if not clean_text(supporter_email) or not clean_text(referrer_email):
                        st.error("Both emails are required.")
                    elif clean_text(supporter_email) == clean_text(referrer_email):
                        st.error("Supporter and referrer cannot be the same.")
                    else:
                        ok = run_write("""
                        MERGE (p:Person {email: $email})
                        ON CREATE SET p.personId = randomUUID()
                        MERGE (ref:Person {email: $referrer})
                        ON CREATE SET ref.personId = randomUUID()
                        MERGE (p)-[:REFERRED_BY]->(ref)
                        """, {
                            "email": clean_text(supporter_email),
                            "referrer": clean_text(referrer_email)
                        })
                        if ok:
                            st.success("Referral saved.")

        st.markdown("**Relationship map**")
        map_email = st.text_input("Supporter email to visualize", value="")
        if st.button("Render relationship map"):
            if not clean_text(map_email):
                st.warning("Enter a supporter email.")
            else:
                df_rel = run_query("""
                MATCH (p:Person {email: $email})
                OPTIONAL MATCH (p)-[:REFERRED_BY]->(ref:Person)
                OPTIONAL MATCH (child:Person)-[:REFERRED_BY]->(p)
                OPTIONAL MATCH (p)-[:SUPPORTS]->(c:Campaign)
                OPTIONAL MATCH (p)-[:VOLUNTEERS_FOR]->(vc:Campaign)
                OPTIONAL MATCH (p)-[:DONATED_TO]->(dc:Campaign)
                RETURN p.email AS email,
                       collect(DISTINCT ref.email) AS referrers,
                       collect(DISTINCT child.email) AS referred,
                       collect(DISTINCT c.name) AS supports,
                       collect(DISTINCT vc.name) AS volunteersFor,
                       collect(DISTINCT dc.name) AS donatedTo
                """, {"email": clean_text(map_email)}, silent=True)

                if df_rel.empty:
                    st.info("No relationships found for that supporter.")
                else:
                    rel = df_rel.iloc[0].to_dict()
                    net = Network(height="420px", width="100%", bgcolor="#ffffff", font_color="#111111")
                    center_id = f"person:{rel.get('email')}"
                    net.add_node(center_id, label=rel.get("email"), color="#6c5ce7", shape="dot")

                    for ref in rel.get("referrers") or []:
                        ref_id = f"ref:{ref}"
                        net.add_node(ref_id, label=ref, color="#e17055")
                        net.add_edge(center_id, ref_id, label="REFERRED_BY")

                    for child in rel.get("referred") or []:
                        child_id = f"child:{child}"
                        net.add_node(child_id, label=child, color="#00b894")
                        net.add_edge(child_id, center_id, label="REFERRED_BY")

                    for campaign in rel.get("supports") or []:
                        cid = f"support:{campaign}"
                        net.add_node(cid, label=campaign, color="#0984e3", shape="box")
                        net.add_edge(center_id, cid, label="SUPPORTS")

                    for campaign in rel.get("volunteersFor") or []:
                        cid = f"volunteer:{campaign}"
                        net.add_node(cid, label=campaign, color="#00cec9", shape="box")
                        net.add_edge(center_id, cid, label="VOLUNTEERS_FOR")

                    for campaign in rel.get("donatedTo") or []:
                        cid = f"donor:{campaign}"
                        net.add_node(cid, label=campaign, color="#fdcb6e", shape="box")
                        net.add_edge(center_id, cid, label="DONATED_TO")

                    components.html(net.generate_html(), height=450, scrolling=True)

# ----------------------------------
# Tab — Activities
# ----------------------------------
with tab_activities:
    st.subheader("Activities")
    st.caption("Log calls, meetings, emails, and other interactions.")

    if driver is None:
        st.warning("Neo4j driver not available. Check connection settings.")
    else:
        activity_cols = st.columns(2)
        with activity_cols[0]:
            with st.form("activity_form"):
                email = st.text_input("Supporter email", value="")
                activity_type = st.selectbox(
                    "Activity type",
                    ["Phone Call", "Email", "Meeting", "SMS", "Canvass", "Note", "Other"]
                )
                custom_type = st.text_input("Custom type (optional)", value="")
                subject = st.text_input("Subject", value="")
                status = st.selectbox("Status", ["Completed", "Scheduled", "Cancelled"])
                act_date = st.date_input("Activity date", value=datetime.utcnow().date())
                act_time = st.time_input(
                    "Activity time",
                    value=datetime.utcnow().time().replace(second=0, microsecond=0)
                )
                details = st.text_area("Details", value="")
                campaign_options = get_distinct_values("Campaign")
                campaign_name = st.selectbox("Related campaign (optional)", [""] + campaign_options)
                save_activity = st.form_submit_button("Save activity")

                if save_activity:
                    if not clean_text(email):
                        st.error("Supporter email is required.")
                    else:
                        dt_val = combine_date_time(act_date, act_time)
                        activity_date = to_iso(dt_val) or datetime.utcnow().isoformat(timespec="seconds")
                        activity_type_final = clean_text(custom_type) or activity_type
                        ok = run_write("""
                        MERGE (p:Person {email: $email})
                        ON CREATE SET p.personId = randomUUID()
                        CREATE (a:Activity {
                          activityId: randomUUID(),
                          type: $type,
                          subject: $subject,
                          status: $status,
                          details: $details,
                          activityDate: datetime($activityDate),
                          createdAt: datetime()
                        })
                        MERGE (p)-[:HAS_ACTIVITY]->(a)
                        FOREACH (_ IN CASE
                          WHEN $campaign IS NULL OR $campaign = '' THEN []
                          ELSE [1]
                        END |
                          MERGE (c:Campaign {name: $campaign})
                          MERGE (a)-[:RELATED_TO]->(c)
                        )
                        """, {
                            "email": clean_text(email),
                            "type": clean_text(activity_type_final) or "Activity",
                            "subject": clean_text(subject),
                            "status": clean_text(status),
                            "details": clean_text(details),
                            "activityDate": activity_date,
                            "campaign": clean_text(campaign_name)
                        })
                        if ok:
                            st.success("Activity saved.")

        with activity_cols[1]:
            st.markdown("**Recent activities**")
            df_activities = run_query("""
            MATCH (p:Person)-[:HAS_ACTIVITY]->(a:Activity)
            OPTIONAL MATCH (a)-[:RELATED_TO]->(c:Campaign)
            RETURN p.email AS email,
                   a.type AS type,
                   a.subject AS subject,
                   a.status AS status,
                   a.activityDate AS activityDate,
                   c.name AS campaign,
                   a.details AS details
            ORDER BY a.activityDate DESC
            LIMIT 200
            """, silent=True)
            if df_activities.empty:
                st.info("No activities logged yet.")
            else:
                st.dataframe(df_activities)
                csv_buffer = StringIO()
                df_activities.to_csv(csv_buffer, index=False)
                st.download_button(
                    "Download activities CSV",
                    csv_buffer.getvalue(),
                    file_name="activities_export.csv",
                    mime="text/csv"
                )

# ----------------------------------
# Tab — Events
# ----------------------------------
with tab_events:
    st.subheader("Events")
    st.caption("Create events and register supporters.")

    if driver is None:
        st.warning("Neo4j driver not available. Check connection settings.")
    else:
        event_cols = st.columns(2)
        with event_cols[0]:
            with st.form("event_form"):
                event_name = st.text_input("Event name", value="")
                start_date = st.date_input("Start date", value=datetime.utcnow().date())
                start_time = st.time_input(
                    "Start time",
                    value=datetime.utcnow().time().replace(second=0, microsecond=0),
                    key="event_start_time"
                )
                end_date = st.date_input("End date", value=datetime.utcnow().date(), key="event_end_date")
                end_time = st.time_input(
                    "End time",
                    value=datetime.utcnow().time().replace(second=0, microsecond=0),
                    key="event_end_time"
                )
                location = st.text_input("Location", value="")
                status = st.selectbox("Status", ["Planned", "Ongoing", "Completed", "Cancelled"])
                capacity = st.number_input("Capacity", min_value=0, value=0, step=1)
                notes = st.text_area("Notes", value="")
                save_event = st.form_submit_button("Save event")

                if save_event:
                    if not clean_text(event_name):
                        st.error("Event name is required.")
                    else:
                        start_iso = to_iso(combine_date_time(start_date, start_time))
                        end_iso = to_iso(combine_date_time(end_date, end_time))
                        event_key = clean_text(event_name).lower().replace(" ", "-")
                        if start_iso:
                            event_key = f"{event_key}-{start_iso[:10]}"
                        run_write("""
                        MERGE (e:Event {eventKey: $eventKey})
                        ON CREATE SET e.eventId = randomUUID(),
                                      e.createdAt = datetime()
                        SET e.name = $name,
                            e.startDate = datetime($startDate),
                            e.endDate = datetime($endDate),
                            e.location = $location,
                            e.status = $status,
                            e.capacity = $capacity,
                            e.notes = $notes,
                            e.updatedAt = datetime()
                        """, {
                            "eventKey": event_key,
                            "name": clean_text(event_name),
                            "startDate": start_iso or datetime.utcnow().isoformat(timespec="seconds"),
                            "endDate": end_iso or datetime.utcnow().isoformat(timespec="seconds"),
                            "location": clean_text(location),
                            "status": clean_text(status),
                            "capacity": int(capacity) if capacity else 0,
                            "notes": clean_text(notes)
                        })
                        st.success("Event saved.")

        with event_cols[1]:
            df_events = run_query("""
            MATCH (e:Event)
            RETURN e.eventKey AS eventKey,
                   e.name AS name,
                   e.startDate AS startDate,
                   e.endDate AS endDate,
                   e.location AS location,
                   e.status AS status,
                   e.capacity AS capacity
            ORDER BY e.startDate DESC
            """, silent=True)

            if df_events.empty:
                st.info("No events created yet.")
            else:
                st.dataframe(df_events.drop(columns=["eventKey"]))
                df_events["startLabel"] = df_events["startDate"].astype(str).replace("None", "")
                df_events["label"] = df_events["name"].fillna("Event") + " • " + df_events["startLabel"]
                event_key_map = dict(zip(df_events["label"], df_events["eventKey"]))
                selected_event_label = st.selectbox("Event to manage", [""] + df_events["label"].tolist())

                if selected_event_label:
                    event_key = event_key_map.get(selected_event_label)
                    with st.form("event_registration_form"):
                        reg_email = st.text_input("Supporter email", value="")
                        reg_status = st.selectbox("Registration status", ["Registered", "Attended", "No Show", "Cancelled"])
                        reg_role = st.selectbox("Role", ["Attendee", "Volunteer", "Speaker", "Organizer"])
                        register = st.form_submit_button("Register supporter")

                        if register:
                            if not clean_text(reg_email):
                                st.error("Supporter email is required.")
                            else:
                                ok = run_write("""
                                MATCH (e:Event {eventKey: $eventKey})
                                MERGE (p:Person {email: $email})
                                ON CREATE SET p.personId = randomUUID()
                                MERGE (p)-[r:REGISTERED_FOR]->(e)
                                SET r.status = $status,
                                    r.role = $role,
                                    r.registeredAt = datetime()
                                """, {
                                    "eventKey": event_key,
                                    "email": clean_text(reg_email),
                                    "status": clean_text(reg_status),
                                    "role": clean_text(reg_role)
                                })
                                if ok:
                                    st.success("Supporter registered.")

                    st.markdown("**Registered supporters**")
                    df_regs = run_query("""
                    MATCH (e:Event {eventKey: $eventKey})
                    OPTIONAL MATCH (p:Person)-[r:REGISTERED_FOR]->(e)
                    RETURN p.email AS email,
                           r.status AS status,
                           r.role AS role,
                           r.registeredAt AS registeredAt
                    ORDER BY r.registeredAt DESC
                    """, {"eventKey": event_key}, silent=True)
                    if df_regs.empty or df_regs["email"].isna().all():
                        st.info("No registrations yet.")
                    else:
                        st.dataframe(df_regs.dropna(subset=["email"]))

# ----------------------------------
# Tab — Contributions
# ----------------------------------
with tab_contributions:
    st.subheader("Contributions")
    st.caption("Record donations and payment details.")

    if driver is None:
        st.warning("Neo4j driver not available. Check connection settings.")
    else:
        contrib_cols = st.columns(2)
        with contrib_cols[0]:
            processor_df = run_query("""
            MATCH (pp:PaymentProcessor)
            RETURN pp.provider AS provider,
                   pp.accountId AS accountId,
                   pp.currency AS currency,
                   pp.liveMode AS liveMode
            ORDER BY pp.provider, pp.accountId
            """, silent=True)
            processor_map = {}
            processor_options = [""]
            if not processor_df.empty:
                for _, row in processor_df.iterrows():
                    label = f"{row.get('provider')} • {row.get('accountId')}"
                    processor_map[label] = {
                        "provider": row.get("provider"),
                        "accountId": row.get("accountId")
                    }
                    processor_options.append(label)

            campaign_options = get_distinct_values("Campaign")

            with st.form("contribution_form"):
                contrib_email = st.text_input("Supporter email", value="")
                amount = st.number_input("Amount", min_value=0.0, value=0.0, step=10.0)
                currency = st.text_input("Currency", value="USD")
                status = st.selectbox("Status", ["Completed", "Pending", "Failed", "Refunded"])
                recv_date = st.date_input("Receive date", value=datetime.utcnow().date())
                recv_time = st.time_input(
                    "Receive time",
                    value=datetime.utcnow().time().replace(second=0, microsecond=0),
                    key="contrib_time"
                )
                processor_label = st.selectbox("Payment processor (optional)", processor_options)
                campaign_name = st.selectbox("Campaign attribution (optional)", [""] + campaign_options)
                source = st.text_input("Source (optional)", value="")
                note = st.text_area("Notes (optional)", value="")
                save_contrib = st.form_submit_button("Save contribution")

                if save_contrib:
                    if not clean_text(contrib_email):
                        st.error("Supporter email is required.")
                    elif amount <= 0:
                        st.error("Amount must be greater than zero.")
                    else:
                        receive_iso = to_iso(combine_date_time(recv_date, recv_time)) or datetime.utcnow().isoformat(timespec="seconds")
                        processor = processor_map.get(processor_label, {})
                        ok = run_write("""
                        MERGE (p:Person {email: $email})
                        ON CREATE SET p.personId = randomUUID()
                        CREATE (c:Contribution {
                          contributionId: randomUUID(),
                          amount: $amount,
                          currency: $currency,
                          status: $status,
                          receiveDate: datetime($receiveDate),
                          source: $source,
                          note: $note,
                          createdAt: datetime()
                        })
                        MERGE (p)-[:MADE_CONTRIBUTION]->(c)
                        FOREACH (_ IN CASE
                          WHEN $processorProvider IS NULL OR $processorAccountId IS NULL THEN []
                          ELSE [1]
                        END |
                          MERGE (pp:PaymentProcessor {provider: $processorProvider, accountId: $processorAccountId})
                          MERGE (c)-[:PROCESSED_BY]->(pp)
                        )
                        FOREACH (_ IN CASE
                          WHEN $campaign IS NULL OR $campaign = '' THEN []
                          ELSE [1]
                        END |
                          MERGE (camp:Campaign {name: $campaign})
                          MERGE (c)-[:ATTRIBUTED_TO]->(camp)
                        )
                        """, {
                            "email": clean_text(contrib_email),
                            "amount": float(amount),
                            "currency": clean_text(currency) or "USD",
                            "status": clean_text(status),
                            "receiveDate": receive_iso,
                            "source": clean_text(source),
                            "note": clean_text(note),
                            "processorProvider": processor.get("provider"),
                            "processorAccountId": processor.get("accountId"),
                            "campaign": clean_text(campaign_name)
                        })
                        if ok:
                            st.success("Contribution saved.")

        with contrib_cols[1]:
            st.markdown("**Recent contributions**")
            df_contribs = run_query("""
            MATCH (p:Person)-[:MADE_CONTRIBUTION]->(c:Contribution)
            OPTIONAL MATCH (c)-[:PROCESSED_BY]->(pp:PaymentProcessor)
            OPTIONAL MATCH (c)-[:ATTRIBUTED_TO]->(camp:Campaign)
            RETURN p.email AS email,
                   c.amount AS amount,
                   c.currency AS currency,
                   c.status AS status,
                   c.receiveDate AS receiveDate,
                   pp.provider AS provider,
                   pp.accountId AS accountId,
                   camp.name AS campaign,
                   c.source AS source
            ORDER BY c.receiveDate DESC
            LIMIT 200
            """, silent=True)
            if df_contribs.empty:
                st.info("No contributions recorded yet.")
            else:
                st.dataframe(df_contribs)
                csv_buffer = StringIO()
                df_contribs.to_csv(csv_buffer, index=False)
                st.download_button(
                    "Download contributions CSV",
                    csv_buffer.getvalue(),
                    file_name="contributions_export.csv",
                    mime="text/csv"
                )

# ----------------------------------
# Tab 4 — Goals & Paths
# ----------------------------------
with tab_goals:
    st.subheader("Goals and paths")
    st.caption("Track measurable goals and define paths that move supporters through steps.")

    if driver is None:
        st.warning("Neo4j driver not available. Check connection settings.")
    else:
        slug = st.session_state.get("nation_slug", NATION_SLUG)
        ensure_nation(slug)

        goal_cols = st.columns(2)
        with goal_cols[0]:
            st.markdown("**Create or update a goal**")
            with st.form("goal_form"):
                goal_name = st.text_input("Goal name", value="")
                goal_desc = st.text_area("Goal description", value="")
                goal_unit = st.text_input("Unit (e.g., contacts, donors, volunteers)", value="")
                target_value = st.number_input("Target value", min_value=0.0, value=0.0, step=1.0)
                current_value = st.number_input("Current value", min_value=0.0, value=0.0, step=1.0)
                save_goal = st.form_submit_button("Save goal")
                if save_goal:
                    if not clean_text(goal_name):
                        st.error("Goal name is required.")
                    else:
                        goal_key = clean_text(goal_name).lower()
                        run_write("""
                        MERGE (g:Goal {goalKey: $goalKey})
                        ON CREATE SET g.goalId = randomUUID(),
                                      g.createdAt = datetime()
                        SET g.name = $name,
                            g.description = $description,
                            g.unit = $unit,
                            g.targetValue = $targetValue,
                            g.currentValue = $currentValue,
                            g.updatedAt = datetime()
                        WITH g
                        MATCH (n:Nation {slug: $slug})
                        MERGE (n)-[:HAS_GOAL]->(g)
                        """, {
                            "goalKey": goal_key,
                            "name": clean_text(goal_name),
                            "description": clean_text(goal_desc),
                            "unit": clean_text(goal_unit),
                            "targetValue": target_value,
                            "currentValue": current_value,
                            "slug": slug
                        })
                        st.success("Goal saved.")

        with goal_cols[1]:
            st.markdown("**Create or update a path**")
            goals_df = run_query("""
            MATCH (n:Nation {slug: $slug})-[:HAS_GOAL]->(g:Goal)
            RETURN g.name AS name, g.goalKey AS goalKey
            ORDER BY g.name
            """, {"slug": slug}, silent=True)
            goal_names = goals_df["name"].tolist() if not goals_df.empty else []
            goal_key_map = dict(zip(goals_df["name"], goals_df["goalKey"])) if not goals_df.empty else {}

            with st.form("path_form"):
                path_name = st.text_input("Path name", value="")
                path_desc = st.text_area("Path description", value="")
                path_steps = st.text_input("Steps (comma separated)", value="")
                path_goal = st.selectbox("Link to goal (optional)", [""] + goal_names, index=0)
                save_path = st.form_submit_button("Save path")
                if save_path:
                    if not clean_text(path_name):
                        st.error("Path name is required.")
                    else:
                        path_key = clean_text(path_name).lower()
                        steps_list = normalize_str_list(split_list(path_steps))
                        run_write("""
                        MERGE (p:Path {pathKey: $pathKey})
                        ON CREATE SET p.pathId = randomUUID(),
                                      p.createdAt = datetime()
                        SET p.name = $name,
                            p.description = $description,
                            p.steps = $steps,
                            p.updatedAt = datetime()
                        WITH p
                        MATCH (n:Nation {slug: $slug})
                        MERGE (n)-[:HAS_PATH]->(p)
                        WITH p
                        OPTIONAL MATCH (g:Goal {goalKey: $goalKey})
                        FOREACH (_ IN CASE WHEN g IS NULL THEN [] ELSE [1] END |
                          MERGE (g)-[:HAS_PATH]->(p)
                        )
                        """, {
                            "pathKey": path_key,
                            "name": clean_text(path_name),
                            "description": clean_text(path_desc),
                            "steps": steps_list,
                            "slug": slug,
                            "goalKey": goal_key_map.get(path_goal)
                        })
                        st.success("Path saved.")

        st.markdown("**Assign supporter to path**")
        paths_df = run_query("""
        MATCH (n:Nation {slug: $slug})-[:HAS_PATH]->(p:Path)
        RETURN p.name AS name, p.pathKey AS pathKey, p.steps AS steps
        ORDER BY p.name
        """, {"slug": slug}, silent=True)
        path_names = paths_df["name"].tolist() if not paths_df.empty else []
        path_key_map = dict(zip(paths_df["name"], paths_df["pathKey"])) if not paths_df.empty else {}
        path_steps_map = dict(zip(paths_df["name"], paths_df["steps"])) if not paths_df.empty else {}

        with st.form("assign_path_form"):
            supporter_email = st.text_input("Supporter email", value="")
            selected_path = st.selectbox("Path", [""] + path_names, index=0)
            step_options = [""] + (path_steps_map.get(selected_path) or [])
            step_value = st.selectbox("Current step", step_options, index=0)
            status_value = st.selectbox("Status", ["Not started", "In progress", "Completed"])
            assign_path = st.form_submit_button("Assign to path")
            if assign_path:
                if not clean_text(supporter_email) or not clean_text(selected_path):
                    st.error("Supporter email and path are required.")
                else:
                    run_write("""
                    MATCH (p:Person {email: $email})
                    MATCH (path:Path {pathKey: $pathKey})
                    MERGE (p)-[r:IN_PATH]->(path)
                    SET r.status = $status,
                        r.step = $step,
                        r.updatedAt = datetime()
                    """, {
                        "email": clean_text(supporter_email),
                        "pathKey": path_key_map.get(selected_path),
                        "status": status_value,
                        "step": clean_text(step_value)
                    })
                    st.success("Supporter assigned to path.")

        st.markdown("**Current goals and paths**")
        df_goals = run_query("""
        MATCH (n:Nation {slug: $slug})-[:HAS_GOAL]->(g:Goal)
        OPTIONAL MATCH (g)-[:HAS_PATH]->(p:Path)
        RETURN g.name AS goal,
               g.currentValue AS current,
               g.targetValue AS target,
               g.unit AS unit,
               collect(DISTINCT p.name) AS paths
        ORDER BY g.name
        """, {"slug": slug}, silent=True)
        if df_goals.empty:
            st.info("No goals defined yet.")
        else:
            df_goals["paths"] = df_goals["paths"].apply(lambda v: ", ".join(v) if isinstance(v, list) else "")
            st.dataframe(df_goals)

# ----------------------------------
# Tab 5 — Communication
# ----------------------------------
with tab_comms:
    st.subheader("Broadcasters")
    st.caption("Broadcasters are official voices for public communication.")

    if driver is None:
        st.warning("Neo4j driver not available. Check connection settings.")
    else:
        slug = st.session_state.get("nation_slug", NATION_SLUG)
        ensure_nation(slug)

        with st.form("broadcaster_form"):
            b_name = st.text_input("Broadcaster name", value="")
            b_role = st.text_input("Role or title", value="")
            b_email = st.text_input("Email", value="")
            b_twitter = st.text_input("Twitter handle", value="")
            b_facebook = st.text_input("Facebook page", value="")
            b_phone = st.text_input("Virtual phone number", value="")
            b_notes = st.text_area("Notes", value="")
            save_broadcaster = st.form_submit_button("Save broadcaster")
            if save_broadcaster:
                if not clean_text(b_name) and not clean_text(b_email):
                    st.error("Provide a name or email for the broadcaster.")
                else:
                    key = f"email:{clean_text(b_email)}" if clean_text(b_email) else f"name:{clean_text(b_name)}"
                    run_write("""
                    MERGE (b:Broadcaster {broadcasterKey: $key})
                    ON CREATE SET b.broadcasterId = randomUUID(),
                                  b.createdAt = datetime()
                    SET b.name = $name,
                        b.role = $role,
                        b.email = $email,
                        b.twitter = $twitter,
                        b.facebook = $facebook,
                        b.phone = $phone,
                        b.notes = $notes,
                        b.updatedAt = datetime()
                    WITH b
                    MATCH (n:Nation {slug: $slug})
                    MERGE (n)-[:HAS_BROADCASTER]->(b)
                    """, {
                        "key": key,
                        "name": clean_text(b_name),
                        "role": clean_text(b_role),
                        "email": clean_text(b_email),
                        "twitter": clean_text(b_twitter),
                        "facebook": clean_text(b_facebook),
                        "phone": clean_text(b_phone),
                        "notes": clean_text(b_notes),
                        "slug": slug
                    })
                    st.success("Broadcaster saved.")

        df_broadcasters = run_query("""
        MATCH (n:Nation {slug: $slug})-[:HAS_BROADCASTER]->(b:Broadcaster)
        RETURN b.broadcasterKey AS key,
               b.name AS name,
               b.role AS role,
               b.email AS email,
               b.twitter AS twitter,
               b.facebook AS facebook,
               b.phone AS phone
        ORDER BY b.name
        """, {"slug": slug}, silent=True)
        if df_broadcasters.empty:
            st.info("No broadcasters yet.")
        else:
            st.dataframe(df_broadcasters.drop(columns=["key"]))
            delete_key = st.selectbox("Delete broadcaster", [""] + df_broadcasters["key"].tolist())
            if delete_key and st.button("Remove broadcaster"):
                run_write("""
                MATCH (b:Broadcaster {broadcasterKey: $key})
                DETACH DELETE b
                """, {"key": delete_key})
                st.warning("Broadcaster removed.")

# ----------------------------------
# Tab 6 — Finance
# ----------------------------------
with tab_finance:
    st.subheader("Finance settings")
    st.caption("Connect a payment processor and review donation totals.")

    if driver is None:
        st.warning("Neo4j driver not available. Check connection settings.")
    else:
        slug = st.session_state.get("nation_slug", NATION_SLUG)
        ensure_nation(slug)

        with st.form("payment_form"):
            provider = st.selectbox("Provider", ["Stripe", "PayPal", "Other"])
            account_id = st.text_input("Account ID", value="")
            currency = st.text_input("Currency", value="USD")
            live_mode = st.checkbox("Live mode", value=False)
            processor_notes = st.text_area("Notes", value="")
            save_payment = st.form_submit_button("Save payment processor")
            if save_payment:
                if not clean_text(account_id):
                    st.error("Account ID is required.")
                else:
                    run_write("""
                    MERGE (pp:PaymentProcessor {provider: $provider, accountId: $accountId})
                    ON CREATE SET pp.processorId = randomUUID(),
                                  pp.createdAt = datetime()
                    SET pp.currency = $currency,
                        pp.liveMode = $liveMode,
                        pp.notes = $notes,
                        pp.updatedAt = datetime()
                    WITH pp
                    MATCH (n:Nation {slug: $slug})
                    MERGE (n)-[:USES_PAYMENT_PROCESSOR]->(pp)
                    """, {
                        "provider": provider,
                        "accountId": clean_text(account_id),
                        "currency": clean_text(currency),
                        "liveMode": live_mode,
                        "notes": clean_text(processor_notes),
                        "slug": slug
                    })
                    st.success("Payment processor saved.")

        df_processors = run_query("""
        MATCH (n:Nation {slug: $slug})-[:USES_PAYMENT_PROCESSOR]->(pp:PaymentProcessor)
        RETURN pp.provider AS provider,
               pp.accountId AS accountId,
               pp.currency AS currency,
               pp.liveMode AS liveMode
        ORDER BY pp.provider
        """, {"slug": slug}, silent=True)
        if df_processors.empty:
            st.info("No payment processors configured.")
        else:
            st.dataframe(df_processors)

        st.markdown("**Donation totals**")
        df_direct = run_query("""
        MATCH (p:Person)
        RETURN sum(coalesce(p.donationTotal, 0)) AS directTotal
        """, silent=True)
        df_rel = run_query("""
        MATCH ()-[d:DONATED_TO]->()
        RETURN sum(coalesce(d.amount, 0)) AS relTotal
        """, silent=True)
        df_contrib = run_query("""
        MATCH ()-[:MADE_CONTRIBUTION]->(c:Contribution)
        RETURN sum(coalesce(c.amount, 0)) AS contribTotal
        """, silent=True)
        direct_total = float(df_direct["directTotal"].iloc[0]) if not df_direct.empty and df_direct["directTotal"].iloc[0] is not None else 0.0
        rel_total = float(df_rel["relTotal"].iloc[0]) if not df_rel.empty and df_rel["relTotal"].iloc[0] is not None else 0.0
        contrib_total = float(df_contrib["contribTotal"].iloc[0]) if not df_contrib.empty and df_contrib["contribTotal"].iloc[0] is not None else 0.0
        st.metric("Total donations", f"{(direct_total + rel_total + contrib_total):,.2f}")

# ----------------------------------
# Tab 7 — Settings
# ----------------------------------
with tab_settings:
    st.subheader("Nation settings")
    st.caption("Configure contact, billing, and core nation details.")

    if driver is None:
        st.warning("Neo4j driver not available. Check connection settings.")
    else:
        slug = st.session_state.get("nation_slug", NATION_SLUG)
        ensure_nation(slug)
        nation = get_nation(slug) or {}

        with st.form("nation_settings_form"):
            new_slug = st.text_input("Nation slug", value=slug)
            name = st.text_input("Nation name", value=nation.get("name", "Default Nation"))
            website = st.text_input("Website", value=nation.get("website", ""))
            timezone = st.text_input("Timezone", value=nation.get("timezone", ""))
            contact_name = st.text_input("Contact name", value=nation.get("contactName", ""))
            contact_email = st.text_input("Contact email", value=nation.get("contactEmail", ""))
            phone = st.text_input("Phone", value=nation.get("phone", ""))
            address = st.text_area("Address", value=nation.get("address", ""))
            billing_email = st.text_input("Billing email", value=nation.get("billingEmail", ""))
            billing_address = st.text_area("Billing address", value=nation.get("billingAddress", ""))
            owner_email = st.text_input("Nation owner email", value=nation.get("ownerEmail", ""))
            save_settings = st.form_submit_button("Save settings")

            if save_settings:
                if not clean_text(new_slug):
                    st.error("Nation slug is required.")
                else:
                    run_write("""
                    MERGE (n:Nation {slug: $slug})
                    ON CREATE SET n.createdAt = datetime()
                    SET n.name = $name,
                        n.website = $website,
                        n.timezone = $timezone,
                        n.contactName = $contactName,
                        n.contactEmail = $contactEmail,
                        n.phone = $phone,
                        n.address = $address,
                        n.billingEmail = $billingEmail,
                        n.billingAddress = $billingAddress,
                        n.ownerEmail = $ownerEmail,
                        n.updatedAt = datetime()
                    """, {
                        "slug": clean_text(new_slug),
                        "name": clean_text(name),
                        "website": clean_text(website),
                        "timezone": clean_text(timezone),
                        "contactName": clean_text(contact_name),
                        "contactEmail": clean_text(contact_email),
                        "phone": clean_text(phone),
                        "address": clean_text(address),
                        "billingEmail": clean_text(billing_email),
                        "billingAddress": clean_text(billing_address),
                        "ownerEmail": clean_text(owner_email)
                    })
                    st.session_state["nation_slug"] = clean_text(new_slug)
                    st.success("Nation settings saved.")

        st.markdown("**Neo4j connection diagnostics**")
        with st.form("neo4j_connection_form"):
            st.session_state.setdefault("neo4j_uri", NEO4J_URI or "")
            st.session_state.setdefault("neo4j_user", NEO4J_USER or "neo4j")
            st.session_state.setdefault("neo4j_password", NEO4J_PASSWORD or "")
            st.session_state.setdefault("neo4j_database", NEO4J_DATABASE or "neo4j")

            neo4j_uri = st.text_input("NEO4J_URI", key="neo4j_uri")
            neo4j_user = st.text_input("NEO4J_USER", key="neo4j_user")
            neo4j_password = st.text_input("NEO4J_PASSWORD", type="password", key="neo4j_password")
            neo4j_database = st.text_input("NEO4J_DATABASE", key="neo4j_database")
            test_connection = st.form_submit_button("Test connection")

            if test_connection:
                ok = reset_driver()
                if ok and driver is not None:
                    df_ping = run_query("RETURN 1 AS ok", silent=True)
                    if not df_ping.empty:
                        st.success("Neo4j connection OK.")
                    else:
                        st.warning("Connected, but query returned no data.")
                else:
                    st.error("Neo4j driver not available. Check credentials and network access.")

# ----------------------------------
# Tab 8 — Schema
# ----------------------------------
with tab_schema:
    st.subheader("Graph schema (Phase 1)")
    schema_md = load_schema_markdown()
    if schema_md:
        st.markdown(schema_md)
    else:
        st.info("Schema file not found. Create GRAPH_SCHEMA.md to document the graph model.")

# ----------------------------------
# Tab 2 — Import/Export
# ----------------------------------
with tab_import:
    st.subheader("Import supporters")
    st.caption("Use the template columns for best results. CSV and Excel are supported.")

    template_df = pd.DataFrame([{
        "First Name": "",
        "Last Name": "",
        "Email": "",
        "Phone": "",
        "Address": "",
        "Latitude": "",
        "Longitude": "",
        "Gender": "",
        "Age": "",
        "About You / Motivation": "",
        "Time Availability": "",
        "Agrees with Manifesto": "",
        "Interested in Party Membership": "",
        "Facebook Group Member": "",
        "Supporter Type": "",
        "Preferred Areas of Involvement": "",
        "How You Can Help": "",
        "Tags": "",
        "Referred By Email": "",
        "Donation Total": ""
    }])
    template_csv = template_df.to_csv(index=False)
    st.download_button(
        "Download CSV template",
        template_csv,
        file_name="supporters_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload supporter file (.xlsx or .csv)",
        type=["xlsx", "csv"]
    )

    df = None
    if uploaded_file:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"Loaded {len(df)} supporters")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    if df is not None and st.button("🚀 Import to Neo4j"):
        if driver is None:
            st.error("Neo4j driver not available. Check connection settings.")
        else:
            try:
                with st.spinner("Importing data into Neo4j…"):

                    run_write("""
                    CREATE CONSTRAINT person_email_unique IF NOT EXISTS
                    FOR (p:Person) REQUIRE p.email IS UNIQUE
                    """)

                    run_write("""
                    CREATE CONSTRAINT district_name_unique IF NOT EXISTS
                    FOR (d:District) REQUIRE d.name IS UNIQUE
                    """)

                    run_write("""
                    CREATE INDEX address_geo IF NOT EXISTS
                    FOR (a:Address)
                    ON (a.latitude, a.longitude)
                    """)

                skipped = 0
                for _, row in df.iterrows():
                    email_val = pick_first(row, ["Email", "email", "E-mail"])
                    if not email_val or str(email_val).strip() == "":
                        skipped += 1
                        continue

                    lat_raw = pick_first(row, ["Latitude", "latitude", "lat"])
                    lon_raw = pick_first(row, ["Longitude", "longitude", "lon"])

                    try:
                        lat_val = float(lat_raw) if lat_raw is not None else None
                    except Exception:
                        lat_val = None
                    try:
                        lon_val = float(lon_raw) if lon_raw is not None else None
                    except Exception:
                        lon_val = None

                    donation_raw = pick_first(row, ["Donation Total", "Donations", "Donation"])
                    try:
                        donation_total = float(donation_raw) if donation_raw is not None else None
                    except Exception:
                        donation_total = None

                    ok = run_write("""
                    MERGE (p:Person {email: $email})
                    ON CREATE SET
                      p.personId = randomUUID(),
                      p.createdAt = datetime(),
                      p.firstName = $firstName,
                      p.lastName = $lastName,
                      p.phone = $phone,
                      p.gender = $gender,
                      p.age = $age,
                      p.about = $about,
                      p.timeAvailability = $timeAvailability,
                      p.agreesWithManifesto = $agreesWithManifesto,
                      p.interestedInMembership = $interestedInMembership,
                      p.facebookGroupMember = $facebookGroupMember,
                      p.lat = $lat,
                      p.lon = $lon,
                      p.donationTotal = $donationTotal
                        ON MATCH SET
                          p.firstName = coalesce($firstName, p.firstName),
                          p.lastName = coalesce($lastName, p.lastName),
                          p.phone = coalesce($phone, p.phone),
                          p.gender = coalesce($gender, p.gender),
                          p.age = coalesce($age, p.age),
                          p.about = coalesce($about, p.about),
                          p.timeAvailability = coalesce($timeAvailability, p.timeAvailability),
                          p.agreesWithManifesto = coalesce($agreesWithManifesto, p.agreesWithManifesto),
                          p.interestedInMembership = coalesce($interestedInMembership, p.interestedInMembership),
                          p.facebookGroupMember = coalesce($facebookGroupMember, p.facebookGroupMember),
                          p.lat = coalesce($lat, p.lat),
                          p.lon = coalesce($lon, p.lon),
                          p.donationTotal = coalesce($donationTotal, p.donationTotal)

                        WITH p

                        MERGE (a:Address {fullAddress: $address})
                        ON CREATE SET
                          a.latitude = $lat,
                          a.longitude = $lon
                        ON MATCH SET
                          a.latitude = coalesce($lat, a.latitude),
                          a.longitude = coalesce($lon, a.longitude)

                        MERGE (p)-[:LIVES_AT]->(a)

                    WITH p

                    MERGE (st:SupporterType {name: $supporterType})
                    MERGE (p)-[:CLASSIFIED_AS]->(st)

                    FOREACH (area IN $involvementAreas |
                      MERGE (ia:InvolvementArea {name: area})
                      MERGE (p)-[:INTERESTED_IN]->(ia)
                    )

                    FOREACH (skill IN $skills |
                      MERGE (sk:Skill {name: skill})
                      MERGE (p)-[:CAN_CONTRIBUTE_WITH]->(sk)
                    )

                    FOREACH (tag IN $tags |
                      MERGE (t:Tag {name: tag})
                      MERGE (p)-[:HAS_TAG]->(t)
                    )

                    FOREACH (_ IN CASE
                      WHEN $referrerEmail IS NULL OR $referrerEmail = '' OR $referrerEmail = $email THEN []
                      ELSE [1]
                    END |
                      MERGE (ref:Person {email: $referrerEmail})
                      ON CREATE SET ref.personId = randomUUID()
                      MERGE (p)-[:REFERRED_BY]->(ref)
                    )
                    """, {
                        "email": str(email_val).strip(),
                        "firstName": pick_first(row, ["First Name", "first_name", "FirstName"]),
                        "lastName": pick_first(row, ["Last Name", "last_name", "LastName"]),
                        "phone": pick_first(row, ["Phone", "phone", "Mobile"]),
                        "address": pick_first(row, ["Address", "address", "Full Address"]),
                        "gender": pick_first(row, ["Gender", "gender"]),
                        "age": pick_first(row, ["Age", "age"]),
                        "about": pick_first(row, ["About You / Motivation", "About", "Motivation"]),
                        "timeAvailability": pick_first(row, ["Time Availability", "Availability"]),
                        "agreesWithManifesto": yes_no_to_bool(pick_first(row, ["Agrees with Manifesto", "Manifesto"])),
                        "interestedInMembership": yes_no_to_bool(pick_first(row, ["Interested in Party Membership", "Membership Interest"])),
                        "facebookGroupMember": yes_no_to_bool(pick_first(row, ["Facebook Group Member", "Facebook Member"])),
                        "supporterType": pick_first(row, ["Supporter Type", "SupporterType", "Type"]) or "Interested",
                        "lat": lat_val,
                        "lon": lon_val,
                        "donationTotal": donation_total,
                        "involvementAreas": split_list(pick_first(row, ["Preferred Areas of Involvement", "Involvement Areas", "Areas"])),
                        "skills": split_list(pick_first(row, ["How You Can Help", "Skills", "Skill"])),
                        "tags": split_list(pick_first(row, ["Tags", "Tag", "Supporter Tags"])),
                        "referrerEmail": pick_first(row, ["Referred By Email", "Referrer Email", "Referred By"])
                    })

                    if not ok:
                        st.stop()

                if skipped:
                    st.warning(f"Skipped {skipped} rows without an email.")
                st.success("✅ Import completed successfully")
            except Exception as e:
                st.error(f"Import failed: {e}")

# ----------------------------------
# Tab 4 — Maps
# ----------------------------------
with tab_maps:
    st.subheader("Maps (requires Neo4j connectivity & geo data)")
    st.caption("Shows supporters with lat/lon. Make sure NEO4J_URI points to the correct Bolt port (7687).")

    map_mode = "Scatter (pydeck + tooltip)"

    if driver is None:
        st.warning("Neo4j driver not available. Check connection settings.")
    else:
        df_geo = run_query("""
        MATCH (p:Person)
        WHERE p.lat IS NOT NULL AND p.lon IS NOT NULL
        OPTIONAL MATCH (p)-[:LIVES_AT]->(a:Address)
        OPTIONAL MATCH (p)-[:CLASSIFIED_AS]->(st:SupporterType)
        OPTIONAL MATCH (p)-[:CAN_CONTRIBUTE_WITH]->(s:Skill)
        WITH p, a, st, count(s) AS skillCount
        OPTIONAL MATCH (p)-[:MADE_CONTRIBUTION]->(c:Contribution)
        WITH p, a, st, skillCount, sum(c.amount) AS donationTotal
        RETURN
          coalesce(p.lat, a.latitude) AS lat,
          coalesce(p.lon, a.longitude) AS lon,
          coalesce(st.name, 'Unspecified') AS type,
          p.email AS email,
          skillCount,
          donationTotal,
          coalesce(p.pointSize,
                   6 +
                   coalesce(skillCount,0) * 3 +
                   coalesce(donationTotal,0) * 0.01) AS pointSize
        """, silent=True)

        if df_geo.empty:
            st.info("No supporters with latitude/longitude found. Import data with lat/lon first.")
        else:
            # Normalize and filter invalid coords
            df_geo["lat"] = pd.to_numeric(df_geo["lat"], errors="coerce")
            df_geo["lon"] = pd.to_numeric(df_geo["lon"], errors="coerce")
            df_geo = df_geo.dropna(subset=["lat", "lon"])

            st.metric("Mapped supporters", len(df_geo))

            color_map = {
                "Interested": "#00aaff",
                "Active Supporter": "#8a2be2",
                "Volunteer": "#2ecc71",
                "Donor": "#f39c12"
            }
            df_geo["color"] = df_geo["type"].map(color_map).fillna("#3388ff")
            df_geo["color_rgba"] = df_geo["color"].apply(hex_to_rgb)
            df_geo["pointSize"] = pd.to_numeric(df_geo.get("pointSize", 6), errors="coerce").fillna(6).clip(lower=1, upper=80)

            # Legend
            legend_cols = st.columns(len(color_map))
            for (label, color), col in zip(color_map.items(), legend_cols):
                col.markdown(
                    f"<div style='display:flex;align-items:center;gap:6px;'>"
                    f"<span style='width:14px;height:14px;background:{color};display:inline-block;border-radius:3px;'></span>"
                    f"<span style='font-size:12px;'>{label}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            if map_mode == "Scatter (pydeck + tooltip)":
                scatter = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_geo,
                    get_position=["lon", "lat"],
                    get_fill_color="color_rgba",
                    get_radius="pointSize",
                    pickable=True,
                )
                view_state = pdk.ViewState(
                    latitude=df_geo["lat"].mean(),
                    longitude=df_geo["lon"].mean(),
                    zoom=14,
                    pitch=20
                )
                deck = pdk.Deck(
                    layers=[scatter],
                    initial_view_state=view_state,
                    tooltip={"text": "Type: {type}\nEmail: {email}\nLat: {lat}\nLon: {lon}"}
                )
                st.pydeck_chart(deck)
            # Single mode only; no other branches

# ----------------------------------
# Tab 5 — Analytics
# ----------------------------------
with tab_analytics:
    st.subheader("Analytics")
    if driver is None:
        st.warning("Neo4j driver not available. Check connection settings.")
    else:
        df_total = run_query("""
        MATCH (p:Person)
        RETURN count(p) AS total
        """, silent=True)
        total_supporters = int(df_total["total"].iloc[0]) if not df_total.empty else 0

        df_manifesto = run_query("""
        MATCH (p:Person)
        RETURN CASE
            WHEN p.agreesWithManifesto IS NULL THEN 'Unspecified'
            WHEN p.agreesWithManifesto THEN 'Yes'
            ELSE 'No'
        END AS agrees, count(p) AS count
        """, silent=True)

        df_membership = run_query("""
        MATCH (p:Person)
        RETURN CASE
            WHEN p.interestedInMembership IS NULL THEN 'Unspecified'
            WHEN p.interestedInMembership THEN 'Yes'
            ELSE 'No'
        END AS interested, count(p) AS count
        """, silent=True)

        df_facebook = run_query("""
        MATCH (p:Person)
        RETURN CASE
            WHEN p.facebookGroupMember IS NULL THEN 'Unspecified'
            WHEN p.facebookGroupMember THEN 'Yes'
            ELSE 'No'
        END AS facebook, count(p) AS count
        """, silent=True)

        manifesto_yes = int(df_manifesto.loc[df_manifesto["agrees"] == "Yes", "count"].sum()) if not df_manifesto.empty else 0
        membership_yes = int(df_membership.loc[df_membership["interested"] == "Yes", "count"].sum()) if not df_membership.empty else 0
        facebook_yes = int(df_facebook.loc[df_facebook["facebook"] == "Yes", "count"].sum()) if not df_facebook.empty else 0

        manifesto_pct = (manifesto_yes / total_supporters * 100) if total_supporters else 0
        membership_pct = (membership_yes / total_supporters * 100) if total_supporters else 0
        facebook_pct = (facebook_yes / total_supporters * 100) if total_supporters else 0

        metric_cols = st.columns(4)
        metric_cols[0].metric("Total Supporters", f"{total_supporters:,}")
        metric_cols[1].metric("Manifesto Agree (%)", f"{manifesto_pct:.1f}%")
        metric_cols[2].metric("Membership Interest (%)", f"{membership_pct:.1f}%")
        metric_cols[3].metric("Facebook Group Member (%)", f"{facebook_pct:.1f}%")

        df_types = run_query("""
        MATCH (p:Person)-[:CLASSIFIED_AS]->(st:SupporterType)
        RETURN st.name AS type, count(p) AS count
        """, silent=True)
        if not df_types.empty:
            type_cols = st.columns(2)
            with type_cols[0]:
                st.markdown("**Supporters by Type (Share)**")
                type_share = (
                    alt.Chart(df_types)
                    .mark_arc(innerRadius=55)
                    .encode(
                        theta=alt.Theta("count:Q"),
                        color=alt.Color("type:N"),
                        tooltip=["type:N", "count:Q"],
                    )
                )
                st.altair_chart(type_share, use_container_width=True)
            with type_cols[1]:
                st.markdown("**Supporters by Type (Counts)**")
                type_bar = (
                    alt.Chart(df_types)
                    .mark_bar()
                    .encode(
                        x=alt.X("type:N", sort="-y"),
                        y=alt.Y("count:Q"),
                        tooltip=["type:N", "count:Q"],
                    )
                )
                st.altair_chart(type_bar, use_container_width=True)
        else:
            st.info("No supporter type data found.")

        df_gender = run_query("""
        MATCH (p:Person)
        RETURN coalesce(p.gender,'Unspecified') AS gender, count(p) AS count
        """, silent=True)

        df_type_gender = run_query("""
        MATCH (p:Person)-[:CLASSIFIED_AS]->(st:SupporterType)
        RETURN st.name AS type, coalesce(p.gender,'Unspecified') AS gender, count(p) AS count
        """, silent=True)

        gender_cols = st.columns(2)
        with gender_cols[0]:
            if not df_gender.empty:
                st.markdown("**Gender Distribution (Share)**")
                gender_pie = (
                    alt.Chart(df_gender)
                    .mark_arc(innerRadius=45)
                    .encode(
                        theta=alt.Theta("count:Q"),
                        color=alt.Color("gender:N"),
                        tooltip=["gender:N", "count:Q"],
                    )
                )
                st.altair_chart(gender_pie, use_container_width=True)
        with gender_cols[1]:
            if not df_type_gender.empty:
                st.markdown("**Gender by Supporter Type**")
                gender_stack = (
                    alt.Chart(df_type_gender)
                    .mark_bar()
                    .encode(
                        x=alt.X("type:N", title="Supporter Type"),
                        y=alt.Y("count:Q"),
                        color=alt.Color("gender:N"),
                        tooltip=["type:N", "gender:N", "count:Q"],
                    )
                )
                st.altair_chart(gender_stack, use_container_width=True)

        df_time = run_query("""
        MATCH (p:Person)
        RETURN coalesce(p.timeAvailability,'Unspecified') AS availability, count(p) AS count
        """, silent=True)
        if not df_time.empty:
            st.markdown("**Time Availability**")
            time_bar = (
                alt.Chart(df_time)
                .mark_bar()
                .encode(
                    y=alt.Y("availability:N", sort="-x"),
                    x=alt.X("count:Q"),
                    tooltip=["availability:N", "count:Q"],
                )
            )
            st.altair_chart(time_bar, use_container_width=True)

        indicator_cols = st.columns(3)
        with indicator_cols[0]:
            if not df_manifesto.empty:
                st.markdown("**Agrees With Manifesto**")
                manifesto_chart = (
                    alt.Chart(df_manifesto)
                    .mark_arc(innerRadius=40)
                    .encode(
                        theta=alt.Theta("count:Q"),
                        color=alt.Color("agrees:N"),
                        tooltip=["agrees:N", "count:Q"],
                    )
                )
                st.altair_chart(manifesto_chart, use_container_width=True)
        with indicator_cols[1]:
            if not df_membership.empty:
                st.markdown("**Interested in Party Membership**")
                membership_chart = (
                    alt.Chart(df_membership)
                    .mark_arc(innerRadius=40)
                    .encode(
                        theta=alt.Theta("count:Q"),
                        color=alt.Color("interested:N"),
                        tooltip=["interested:N", "count:Q"],
                    )
                )
                st.altair_chart(membership_chart, use_container_width=True)
        with indicator_cols[2]:
            if not df_facebook.empty:
                st.markdown("**Facebook Group Member**")
                facebook_chart = (
                    alt.Chart(df_facebook)
                    .mark_arc(innerRadius=40)
                    .encode(
                        theta=alt.Theta("count:Q"),
                        color=alt.Color("facebook:N"),
                        tooltip=["facebook:N", "count:Q"],
                    )
                )
                st.altair_chart(facebook_chart, use_container_width=True)

        df_age = run_query("""
        MATCH (p:Person)
        WHERE p.age IS NOT NULL
        RETURN p.age AS age
        """, silent=True)
        if not df_age.empty:
            ages = pd.to_numeric(df_age["age"], errors="coerce")
            ages = ages[ages > 0]
            if not ages.empty:
                st.markdown("**Age Distribution**")
                age_chart = (
                    alt.Chart(pd.DataFrame({"age": ages}))
                    .mark_area(opacity=0.4, line={"color": "#6b6b6b"})
                    .encode(
                        x=alt.X("age:Q", bin=alt.Bin(maxbins=12), title="Age"),
                        y=alt.Y("count():Q", title="Supporters"),
                        tooltip=["count():Q"],
                    )
                )
                st.altair_chart(age_chart, use_container_width=True)

        top_cols = st.columns(2)
        with top_cols[0]:
            df_involve = run_query("""
            MATCH (p:Person)-[:INTERESTED_IN]->(ia:InvolvementArea)
            RETURN ia.name AS area, count(p) AS count
            ORDER BY count DESC
            LIMIT 10
            """, silent=True)
            if not df_involve.empty:
                st.markdown("**Top Involvement Areas**")
                involve_bar = (
                    alt.Chart(df_involve)
                    .mark_bar()
                    .encode(
                        y=alt.Y("area:N", sort="-x"),
                        x=alt.X("count:Q"),
                        tooltip=["area:N", "count:Q"],
                    )
                )
                st.altair_chart(involve_bar, use_container_width=True)
        with top_cols[1]:
            df_skills = run_query("""
            MATCH (p:Person)-[:CAN_CONTRIBUTE_WITH]->(s:Skill)
            RETURN s.name AS skill, count(p) AS count
            ORDER BY count DESC
            LIMIT 10
            """, silent=True)
            if not df_skills.empty:
                st.markdown("**Top Skills**")
                skill_bar = (
                    alt.Chart(df_skills)
                    .mark_bar()
                    .encode(
                        y=alt.Y("skill:N", sort="-x"),
                        x=alt.X("count:Q"),
                        tooltip=["skill:N", "count:Q"],
                    )
                )
                st.altair_chart(skill_bar, use_container_width=True)

# ----------------------------------
# Tab 7 — Chatbot
# ----------------------------------
with tab_chatbot:
    st.subheader("Chatbot (stub)")
    st.info("Ask graph questions; this stub executes a simple match on supporter type keywords.")
    q = st.text_input("Question", placeholder="e.g., Show interested supporters")
    def render_results_header(title="Results"):
        header_cols = st.columns([3, 1])
        with header_cols[0]:
            st.markdown(f"**{title}**")
        with header_cols[1]:
            st.caption("In progress")

    if st.button("Ask"):
        if not q.strip():
            st.warning("Enter a question.")
        else:
            q_lower = q.lower()
            if "gender" in q_lower and "interested" in q_lower:
                df_q = run_query("""
                MATCH (p:Person)-[:CLASSIFIED_AS]->(st:SupporterType)
                WHERE toLower(st.name) CONTAINS 'interested'
                RETURN p.firstName AS firstName, p.lastName AS lastName, p.email AS email,
                       st.name AS type, coalesce(p.gender, 'Unspecified') AS gender
                LIMIT 200
                """, silent=True)
                if df_q.empty:
                    st.info("No matching supporters found.")
                else:
                    render_results_header("Results")
                    st.dataframe(df_q)
                    st.markdown("**Gender ratio among interested**")
                    gender_counts = (
                        df_q["gender"]
                        .value_counts(dropna=False)
                        .rename_axis("gender")
                        .reset_index(name="count")
                    )
                    gender_counts["ratio"] = (
                        gender_counts["count"] / gender_counts["count"].sum()
                    ).round(3)
                    st.bar_chart(gender_counts.set_index("gender")["count"])
                    st.dataframe(gender_counts)
            elif "interested" in q_lower:
                df_q = run_query("""
                MATCH (p:Person)-[:CLASSIFIED_AS]->(st:SupporterType)
                WHERE toLower(st.name) CONTAINS 'interested'
                RETURN p.firstName AS firstName, p.lastName AS lastName, p.email AS email,
                       st.name AS type, coalesce(p.gender, 'Unspecified') AS gender
                LIMIT 50
                """, silent=True)
                if df_q.empty:
                    st.info("No matching supporters found.")
                else:
                    render_results_header("Results")
                    st.dataframe(df_q)
                    st.markdown("**Charts**")
                    chart_cols = st.columns(2)
                    with chart_cols[0]:
                        if "type" in df_q.columns:
                            type_counts = (
                                df_q["type"]
                                .value_counts(dropna=False)
                                .rename_axis("type")
                                .reset_index(name="count")
                            )
                            st.bar_chart(type_counts.set_index("type"))
                    with chart_cols[1]:
                        if "gender" in df_q.columns:
                            gender_counts = (
                                df_q["gender"]
                                .value_counts(dropna=False)
                                .rename_axis("gender")
                                .reset_index(name="count")
                            )
                            st.bar_chart(gender_counts.set_index("gender"))
            else:
                st.info("No intent matched; try mentioning 'interested'.")

# ----------------------------------
# Tab 8 — Cleanup
# ----------------------------------
with tab_cleanup:
    st.subheader("Danger zone")
    st.caption("Remove all data from Neo4j (irreversible).")
    if st.button("🧹 DELETE ALL DATA (DANGEROUS)"):
        run_write("MATCH (n) DETACH DELETE n")
        st.warning("Database cleared")
