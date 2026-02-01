import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv

load_dotenv()


def get_config(key, default=None):
    value = os.getenv(key)
    return value if value is not None else default


NEO4J_URI = get_config("NEO4J_URI")
NEO4J_USER = get_config("NEO4J_USER") or get_config("NEO4J_USERNAME") or "neo4j"
NEO4J_PASSWORD = get_config("NEO4J_PASSWORD")
NEO4J_DATABASE = get_config("NEO4J_DATABASE", "neo4j")

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        if not NEO4J_URI or not NEO4J_PASSWORD:
            raise RuntimeError("Missing Neo4j credentials")
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD)
        )
    return _driver


def run_query(query, params=None):
    with get_driver().session(database=NEO4J_DATABASE) as session:
        result = session.run(query, params or {})
        return [r.data() for r in result]


def run_write(query, params=None):
    with get_driver().session(database=NEO4J_DATABASE) as session:
        session.run(query, params or {})


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


class SupporterIn(BaseModel):
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    about: Optional[str] = None
    time_availability: Optional[str] = None
    agrees_with_manifesto: Optional[bool] = None
    interested_in_membership: Optional[bool] = None
    facebook_group_member: Optional[bool] = None
    supporter_type: Optional[str] = Field(None, description="Supporter type label")
    tags: List[str] = Field(default_factory=list)
    involvement_areas: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    address: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    donation_total: Optional[float] = None
    referrer_email: Optional[str] = None


class SupporterOut(SupporterIn):
    pass


app = FastAPI(title="Supporter CRM API", version="0.1.0")


@app.on_event("shutdown")
def shutdown_event():
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


@app.get("/health")
def health_check():
    return {"status": "ok"}


def upsert_supporter(supporter: SupporterIn):
    tags = normalize_str_list(supporter.tags)
    areas = normalize_str_list(supporter.involvement_areas)
    skills = normalize_str_list(supporter.skills)
    supporter_type = clean_text(supporter.supporter_type) or "Interested"
    referrer = clean_text(supporter.referrer_email)

    run_write("""
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
    FOREACH (_ IN CASE WHEN $address IS NULL OR $address = '' THEN [] ELSE [1] END |
      MERGE (a:Address {fullAddress: $address})
      ON CREATE SET a.latitude = $lat, a.longitude = $lon
      ON MATCH SET a.latitude = coalesce($lat, a.latitude),
                  a.longitude = coalesce($lon, a.longitude)
      MERGE (p)-[:LIVES_AT]->(a)
    )

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
        "email": supporter.email,
        "firstName": clean_text(supporter.first_name),
        "lastName": clean_text(supporter.last_name),
        "phone": clean_text(supporter.phone),
        "gender": clean_text(supporter.gender),
        "age": supporter.age,
        "about": clean_text(supporter.about),
        "timeAvailability": clean_text(supporter.time_availability),
        "agreesWithManifesto": supporter.agrees_with_manifesto,
        "interestedInMembership": supporter.interested_in_membership,
        "facebookGroupMember": supporter.facebook_group_member,
        "supporterType": supporter_type,
        "lat": supporter.lat,
        "lon": supporter.lon,
        "donationTotal": supporter.donation_total,
        "address": clean_text(supporter.address),
        "involvementAreas": areas,
        "skills": skills,
        "tags": tags,
        "referrerEmail": referrer
    })


def replace_relationships(email: str, supporter: SupporterIn):
    tags = normalize_str_list(supporter.tags)
    areas = normalize_str_list(supporter.involvement_areas)
    skills = normalize_str_list(supporter.skills)
    supporter_type = clean_text(supporter.supporter_type) or "Interested"
    referrer = clean_text(supporter.referrer_email)

    run_write("""
    MATCH (p:Person {email: $email})
    OPTIONAL MATCH (p)-[r:CLASSIFIED_AS]->(:SupporterType)
    OPTIONAL MATCH (p)-[t:HAS_TAG]->(:Tag)
    OPTIONAL MATCH (p)-[a:INTERESTED_IN]->(:InvolvementArea)
    OPTIONAL MATCH (p)-[s:CAN_CONTRIBUTE_WITH]->(:Skill)
    OPTIONAL MATCH (p)-[rb:REFERRED_BY]->(:Person)
    DELETE r, t, a, s, rb
    """, {"email": email})

    run_write("""
    MATCH (p:Person {email: $email})
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
        "supporterType": supporter_type,
        "involvementAreas": areas,
        "skills": skills,
        "tags": tags,
        "referrerEmail": referrer
    })


@app.get("/supporters", response_model=List[SupporterOut])
def list_supporters(
    q: Optional[str] = None,
    tags: Optional[List[str]] = Query(None),
    types: Optional[List[str]] = Query(None),
    limit: int = 200
):
    tags = normalize_str_list(tags or [])
    types = normalize_str_list(types or [])
    try:
        records = run_query("""
        MATCH (p:Person)
        OPTIONAL MATCH (p)-[:CLASSIFIED_AS]->(st:SupporterType)
        OPTIONAL MATCH (p)-[:HAS_TAG]->(t:Tag)
        OPTIONAL MATCH (p)-[:INTERESTED_IN]->(ia:InvolvementArea)
        OPTIONAL MATCH (p)-[:CAN_CONTRIBUTE_WITH]->(sk:Skill)
        OPTIONAL MATCH (p)-[:LIVES_AT]->(a:Address)
        OPTIONAL MATCH (p)-[:REFERRED_BY]->(ref:Person)
        WITH p, a,
             collect(DISTINCT st.name) AS supporterTypes,
             collect(DISTINCT t.name) AS tags,
             collect(DISTINCT ia.name) AS areas,
             collect(DISTINCT sk.name) AS skills,
             collect(DISTINCT ref.email) AS referrers
        WHERE ($q IS NULL OR $q = ""
           OR toLower(p.email) CONTAINS toLower($q)
           OR toLower(p.firstName) CONTAINS toLower($q)
           OR toLower(p.lastName) CONTAINS toLower($q))
          AND (size($tags) = 0 OR any(t IN tags WHERE t IN $tags))
          AND (size($types) = 0 OR any(t IN supporterTypes WHERE t IN $types))
        RETURN p.email AS email,
               p.firstName AS firstName,
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
               a.fullAddress AS address,
               supporterTypes AS supporterTypes,
               tags AS tags,
               areas AS involvementAreas,
               skills AS skills,
               referrers AS referrers
        ORDER BY p.lastName, p.firstName
        LIMIT $limit
        """, {
            "q": q,
            "tags": tags,
            "types": types,
            "limit": limit
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    results = []
    for row in records:
        supporter_type = row.get("supporterTypes")[0] if row.get("supporterTypes") else None
        referrer = row.get("referrers")[0] if row.get("referrers") else None
        results.append(SupporterOut(
            email=row.get("email"),
            first_name=row.get("firstName"),
            last_name=row.get("lastName"),
            phone=row.get("phone"),
            gender=row.get("gender"),
            age=row.get("age"),
            about=row.get("about"),
            time_availability=row.get("timeAvailability"),
            agrees_with_manifesto=row.get("agreesWithManifesto"),
            interested_in_membership=row.get("interestedInMembership"),
            facebook_group_member=row.get("facebookGroupMember"),
            supporter_type=supporter_type,
            tags=row.get("tags") or [],
            involvement_areas=row.get("involvementAreas") or [],
            skills=row.get("skills") or [],
            address=row.get("address"),
            lat=row.get("lat"),
            lon=row.get("lon"),
            donation_total=row.get("donationTotal"),
            referrer_email=referrer
        ))
    return results


@app.get("/supporters/{email}", response_model=SupporterOut)
def get_supporter(email: str):
    try:
        records = run_query("""
        MATCH (p:Person {email: $email})
        OPTIONAL MATCH (p)-[:CLASSIFIED_AS]->(st:SupporterType)
        OPTIONAL MATCH (p)-[:HAS_TAG]->(t:Tag)
        OPTIONAL MATCH (p)-[:INTERESTED_IN]->(ia:InvolvementArea)
        OPTIONAL MATCH (p)-[:CAN_CONTRIBUTE_WITH]->(sk:Skill)
        OPTIONAL MATCH (p)-[:LIVES_AT]->(a:Address)
        OPTIONAL MATCH (p)-[:REFERRED_BY]->(ref:Person)
        RETURN p.email AS email,
               p.firstName AS firstName,
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
               a.fullAddress AS address,
               collect(DISTINCT st.name) AS supporterTypes,
               collect(DISTINCT t.name) AS tags,
               collect(DISTINCT ia.name) AS involvementAreas,
               collect(DISTINCT sk.name) AS skills,
               collect(DISTINCT ref.email) AS referrers
        """, {"email": email})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not records:
        raise HTTPException(status_code=404, detail="Supporter not found")

    row = records[0]
    supporter_type = row.get("supporterTypes")[0] if row.get("supporterTypes") else None
    referrer = row.get("referrers")[0] if row.get("referrers") else None
    return SupporterOut(
        email=row.get("email"),
        first_name=row.get("firstName"),
        last_name=row.get("lastName"),
        phone=row.get("phone"),
        gender=row.get("gender"),
        age=row.get("age"),
        about=row.get("about"),
        time_availability=row.get("timeAvailability"),
        agrees_with_manifesto=row.get("agreesWithManifesto"),
        interested_in_membership=row.get("interestedInMembership"),
        facebook_group_member=row.get("facebookGroupMember"),
        supporter_type=supporter_type,
        tags=row.get("tags") or [],
        involvement_areas=row.get("involvementAreas") or [],
        skills=row.get("skills") or [],
        address=row.get("address"),
        lat=row.get("lat"),
        lon=row.get("lon"),
        donation_total=row.get("donationTotal"),
        referrer_email=referrer
    )


@app.post("/supporters", response_model=SupporterOut)
def create_supporter(supporter: SupporterIn):
    try:
        upsert_supporter(supporter)
        return get_supporter(supporter.email)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/supporters/{email}", response_model=SupporterOut)
def update_supporter(email: str, supporter: SupporterIn):
    if email != supporter.email:
        raise HTTPException(status_code=400, detail="Email in path and body must match")
    try:
        upsert_supporter(supporter)
        replace_relationships(email, supporter)
        return get_supporter(email)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/supporters/{email}")
def delete_supporter(email: str):
    try:
        run_write("MATCH (p:Person {email: $email}) DETACH DELETE p", {"email": email})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "deleted", "email": email}
