import os
import pandas as pd
import streamlit as st
import altair as alt
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv
import pydeck as pdk
import requests
from streamlit_js_eval import get_geolocation

# ----------------------------------
# Load environment
# ----------------------------------
load_dotenv()

def get_config(key, default=None):
    if key in st.secrets:
        return st.secrets.get(key, default)
    value = os.getenv(key)
    return value if value is not None else default

NEO4J_URI = get_config("NEO4J_URI")
NEO4J_USER = get_config("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = get_config("NEO4J_PASSWORD")
NEO4J_DATABASE = get_config("NEO4J_DATABASE", "neo4j")

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
    
    # Handle sandbox URIs - convert to direct bolt if needed
    uri = NEO4J_URI
    if "bolt.neo4jsandbox.com:443" in uri:
        uri = uri.replace("bolt.neo4jsandbox.com:443", "bolt://3.236.197.115").replace("neo4j+s://", "bolt://")

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

tab_form, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 New Supporter",
    "📤 Import",
    "🗺️ Maps",
    "📊 Analytics",
    "💬 Chatbot",
    "🧹 Cleanup"
])

# ----------------------------------
# Tab 0 — Form
# ----------------------------------
with tab_form:
    st.subheader("Add a supporter (single entry form)")
    st.caption("Fill the fields, submit, preview, then add to Neo4j.")

    default_areas = ["Field Organizing", "Events", "Fundraising", "Communications", "Tech", "Policy"]
    default_skills = ["Communication", "Fundraising", "Data", "Design", "Engineering", "Organizing"]
    supporter_types = ["Interested", "Volunteer", "Donor", "Active Supporter"]

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
                    "skills": skills_sel
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
# Tab 1 — Import
# ----------------------------------
with tab1:
    uploaded_file = st.file_uploader(
        "Upload Excel (.xlsx)",
        type=["xlsx"]
    )

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.success(f"Loaded {len(df)} supporters")
        st.dataframe(df.head())

    if uploaded_file and st.button("🚀 Import to Neo4j"):
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

                for _, row in df.iterrows():

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

                        ok = run_write("""
                    MERGE (p:Person {email: $email})
                    ON CREATE SET
                      p.personId = randomUUID(),
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
                          p.lon = $lon
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
                          p.lon = coalesce($lon, p.lon)

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
                    """, {
                        "email": row["Email"],
                        "firstName": row["First Name"],
                        "lastName": row["Last Name"],
                        "phone": row["Phone"],
                        "address": row["Address"],
                        "gender": row["Gender"],
                        "age": row["Age"],
                        "about": row["About You / Motivation"],
                        "timeAvailability": row["Time Availability"],
                        "agreesWithManifesto": yes_no_to_bool(row["Agrees with Manifesto"]),
                        "interestedInMembership": yes_no_to_bool(row["Interested in Party Membership"]),
                        "facebookGroupMember": yes_no_to_bool(row["Facebook Group Member"]),
                        "supporterType": row.get("Supporter Type", "Interested"),
                            "lat": lat_val,
                            "lon": lon_val,
                        "involvementAreas": split_list(row["Preferred Areas of Involvement"]),
                        "skills": split_list(row["How You Can Help"])
                    })

                        if not ok:
                            st.stop()

                st.success("✅ Import completed successfully")
            except Exception as e:
                st.error(f"Import failed: {e}")

# ----------------------------------
# Tab 2 — Maps
# ----------------------------------
with tab2:
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
        OPTIONAL MATCH (p)-[d:DONATED]->(:Donation)
        WITH p, a, st, skillCount, sum(d.amount) AS donationTotal
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
                    zoom=12,
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
# Tab 3 — Analytics
# ----------------------------------
with tab3:
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
# Tab 4 — Chatbot
# ----------------------------------
with tab4:
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
# Tab 5 — Cleanup
# ----------------------------------
with tab5:
    st.subheader("Danger zone")
    st.caption("Remove all data from Neo4j (irreversible).")
    if st.button("🧹 DELETE ALL DATA (DANGEROUS)"):
        run_write("MATCH (n) DETACH DELETE n")
        st.warning("Database cleared")
