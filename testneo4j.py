import os
from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

if not NEO4J_URI or not NEO4J_PASSWORD:
    raise SystemExit("Missing NEO4J_URI or NEO4J_PASSWORD in .env")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD),
)

cypher_query = """
MATCH (n)
RETURN COUNT(n) AS count
LIMIT $limit
"""

try:
    with driver.session(database=NEO4J_DATABASE) as session:
        if hasattr(session, "execute_read"):
            results = session.execute_read(
                lambda tx: tx.run(cypher_query, limit=10).data()
            )
        else:
            results = session.read_transaction(
                lambda tx: tx.run(cypher_query, limit=10).data()
            )
        for record in results:
            print(record["count"])
finally:
    driver.close()
