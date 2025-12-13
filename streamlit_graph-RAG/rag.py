import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD



#: Neo4J Utility
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def run_query(query: str, params: dict = {}):
    driver = get_driver()
    with driver.session() as session:
        res = session.run(query, params)
        return [dict(record) for record in res]



#: Cypher Templates
Q_LEAST_DELAYED_FLIGHTS_ON_ROUTE = """
MATCH (j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(o:Airport {station_code: $origin})
MATCH (f)-[:ARRIVES_AT]->(d:Airport {station_code: $destination})
RETURN
  f.flight_number AS flight,
  avg(j.arrival_delay_minutes) AS avg_delay,
  count(j) AS samples
ORDER BY avg_delay ASC
LIMIT $limit
""" # Flights from ... to ... with the least delays

Q_LEAST_CROWDED_FLIGHTS_ON_ROUTE = """
MATCH (p:Passenger)-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(o:Airport {station_code: $origin})
MATCH (f)-[:ARRIVES_AT]->(d:Airport {station_code: $destination})
RETURN
  f.flight_number AS flight,
  count(DISTINCT p) AS passengers
ORDER BY passengers ASC
LIMIT $limit
""" # Flights from ... to ... with the least passengers

Q_FLIGHT_HAS_CLASS = """
MATCH (f:Flight {flight_number: $flight_number})
OPTIONAL MATCH (j:Journey)-[:ON]->(f)
WHERE j.passenger_class = $class
RETURN
  count(j) > 0 AS offers_class
""" # Does flight ... offer class ...?

Q_FLIGHT_AIRCRAFT_TYPE = """
MATCH (f:Flight {flight_number: $flight_number})
RETURN
  f.flight_number AS flight,
  f.fleet_type_description AS aircraft
""" # Which aircraft will I be flying on?

Q_BEST_FOOD_ROUTES = """
MATCH (j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(o:Airport)
MATCH (f)-[:ARRIVES_AT]->(d:Airport)
RETURN
  o.station_code AS origin,
  d.station_code AS destination,
  avg(j.food_satisfaction_score) AS avg_food,
  count(j) AS samples
ORDER BY avg_food DESC
LIMIT $limit
""" # Routes with the best food

Q_FREQUENT_FLYERS_ON_ROUTE = """
MATCH (p:Passenger)-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(o:Airport {station_code: $origin})
MATCH (f)-[:ARRIVES_AT]->(d:Airport {station_code: $destination})
WHERE p.loyalty_program_level IS NOT NULL
  AND p.loyalty_program_level <> 'None'
RETURN
  count(p) > 0 AS frequent_flyers_use_route
""" # Do frequent flyers use this route?

Q_MOST_POPULAR_AIRPORTS = """
MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(a:Airport)
RETURN
  a.station_code AS airport,
  count(j) AS journeys
ORDER BY journeys DESC
LIMIT $limit
""" # Most populat airports

Q_ROUTE_DISTANCE = """
MATCH (j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(o:Airport {station_code: $origin})
MATCH (f)-[:ARRIVES_AT]->(d:Airport {station_code: $destination})
RETURN
  avg(j.actual_flown_miles) AS avg_distance_miles,
  count(j) AS samples
""" # Flight distance from ... to ...

Q_HAS_DIRECT_FLIGHT = """
MATCH (j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(o:Airport {station_code: $origin})
MATCH (f)-[:ARRIVES_AT]->(d:Airport {station_code: $destination})
WHERE j.number_of_legs = 1
RETURN
  count(j) > 0 AS direct_flight_available
""" # Can I go from ... to ... directly?

Q_DOMINANT_GENERATION_AT_AIRPORT = """
MATCH (p:Passenger)-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(a:Airport {station_code: $code})
RETURN
  p.generation AS generation,
  count(p) AS passengers
ORDER BY passengers DESC
LIMIT 1
""" # Dominating passenger generation at airport ...

TEMPLATES = {
    "least_delayed_flights": Q_LEAST_DELAYED_FLIGHTS_ON_ROUTE,
    "least_crowded_flights": Q_LEAST_CROWDED_FLIGHTS_ON_ROUTE,
    "best_food_routes": Q_BEST_FOOD_ROUTES,
    "popular_airports": Q_MOST_POPULAR_AIRPORTS,
    "flight_has_class": Q_FLIGHT_HAS_CLASS,
    "route_distance": Q_ROUTE_DISTANCE,
    "dominant_generation_airport": Q_DOMINANT_GENERATION_AT_AIRPORT,
    "direct_flight": Q_HAS_DIRECT_FLIGHT,
    "flight_aircraft": Q_FLIGHT_AIRCRAFT_TYPE,
    "frequent_flyers_route": Q_FREQUENT_FLYERS_ON_ROUTE
}



#: Features Vector Embeddings
EMBEDDING_MODELS = {
    'miniLM': 'all-MiniLM-L6-v2',
    'mpnet': 'all-mpnet-base-v2'
}

def row_to_feature_text(row: dict) -> str:
    return ', '.join([f'{k}: {v}' for k, v in row.items()])

def embed_and_rank(user_query: str, rows: list, model_name: str):
    if not rows: return rows

    model = SentenceTransformer(EMBEDDING_MODELS[model_name])

    row_texts = [row_to_feature_text(r) for r in rows]
    row_embeddings = model.encode(row_texts)
    query_embedding = model.encode(user_query)

    similarities = np.dot(row_embeddings, query_embedding) / (np.linalg.norm(row_embeddings, axis=1) * np.linalg.norm(query_embedding))
    ranked = sorted(zip(rows, similarities), key=lambda x: x[1], reverse=True)

    return [r[0] for r in ranked]

def retrieve(intent: str, entities: dict, user_query: str, use_embeddings: bool = False, model_name: str = 'miniLM'):
    results = run_query(TEMPLATES[intent], entities)

    return embed_and_rank(
        user_query=user_query,
        rows=results,
        model_name=model_name
    ) if use_embeddings else results
