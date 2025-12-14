from config import CONFIG
import numpy as np
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import faiss



#: Neo4J Utility
NEO4J_DRIVER = GraphDatabase.driver(CONFIG.get('NEO4J_URI'), auth=(CONFIG.get('NEO4J_USERNAME'), CONFIG.get('NEO4J_PASSWORD')))

def run_query(query: str, params: dict = {}):
    with NEO4J_DRIVER.session() as session:
        res = session.run(query, params)
        return [dict(record) for record in res]



#: Cypher Templates
Q_MOST_DELAYED_FLIGHTS_ON_ROUTE = '''
MATCH (j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(o:Airport {station_code: $origin})
MATCH (f)-[:ARRIVES_AT]->(d:Airport {station_code: $destination})
RETURN
    f.flight_number AS flight,
    avg(j.arrival_delay_minutes) AS avg_delay,
    count(j) AS samples
ORDER BY avg_delay DESC
LIMIT $limit
''' # Flights from airport ... to airport ... with the most delays

Q_LEAST_CROWDED_FLIGHTS_ON_ROUTE = '''
MATCH (p:Passenger)-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(o:Airport {station_code: $origin})
MATCH (f)-[:ARRIVES_AT]->(d:Airport {station_code: $destination})
RETURN
    f.flight_number AS flight,
    count(DISTINCT p) AS passengers
ORDER BY passengers ASC
LIMIT $limit
''' # Flights from airport ... to airport ... with the least passengers

Q_FLIGHT_CLASSES_OFFERED = '''
MATCH (f:Flight {flight_number: $flight_number})
MATCH (j:Journey)-[:ON]->(f)
RETURN DISTINCT j.passenger_class AS offered_class
''' # Classes offered in flight ...

Q_FLIGHT_AIRCRAFT_TYPE = '''
MATCH (f:Flight {flight_number: $flight_number})
RETURN
    f.flight_number AS flight,
    f.fleet_type_description AS aircraft
''' # Which aircraft will I be flying on?

Q_WORST_FOOD_ROUTES = '''
MATCH (j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(o:Airport)
MATCH (f)-[:ARRIVES_AT]->(d:Airport)
RETURN
    o.station_code AS origin,
    d.station_code AS destination,
    avg(j.food_satisfaction_score) AS avg_food,
    count(j) AS samples
ORDER BY avg_food ASC
LIMIT $limit
''' # Routes with the worst food

Q_FREQUENT_FLYERS_ON_ROUTE = '''
MATCH (p:Passenger)-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(o:Airport {station_code: $origin})
MATCH (f)-[:ARRIVES_AT]->(d:Airport {station_code: $destination})
WHERE p.loyalty_program_level IS NOT NULL
    AND p.loyalty_program_level <> 'None'
RETURN
    count(p) > 0 AS frequent_flyers_use_route
''' # Do frequent flyers use this route?

Q_MOST_POPULAR_AIRPORTS = '''
MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(a:Airport)
RETURN
    a.station_code AS airport,
    count(j) AS journeys
ORDER BY journeys DESC
LIMIT $limit
''' # Most popular airports

Q_ROUTE_DISTANCE = '''
MATCH (j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(o:Airport {station_code: $origin})
MATCH (f)-[:ARRIVES_AT]->(d:Airport {station_code: $destination})
RETURN
    avg(j.actual_flown_miles) AS avg_distance_miles,
    count(j) AS samples
''' # Flight distance from airport ... to airport ...

Q_HAS_DIRECT_FLIGHT = '''
MATCH (j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(o:Airport {station_code: $origin})
MATCH (f)-[:ARRIVES_AT]->(d:Airport {station_code: $destination})
WHERE j.number_of_legs = 1
RETURN
    count(j) > 0 AS direct_flight_available
''' # Can I go from airport ... to airport ... directly?

Q_DOMINANT_GENERATION_AT_AIRPORT = '''
MATCH (p:Passenger)-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(a:Airport {station_code: $code})
RETURN
    p.generation AS generation,
    count(p) AS passengers
ORDER BY passengers DESC
LIMIT 1
''' # Dominating passenger generation at airport ...

TEMPLATES = {
    'most_delayed_flights': Q_MOST_DELAYED_FLIGHTS_ON_ROUTE,
    'least_crowded_flights': Q_LEAST_CROWDED_FLIGHTS_ON_ROUTE,
    'worst_food_routes': Q_WORST_FOOD_ROUTES,
    'popular_airports': Q_MOST_POPULAR_AIRPORTS,
    'flight_classes_offered': Q_FLIGHT_CLASSES_OFFERED,
    'route_distance': Q_ROUTE_DISTANCE,
    'dominant_generation_airport': Q_DOMINANT_GENERATION_AT_AIRPORT,
    'direct_flight': Q_HAS_DIRECT_FLIGHT,
    'flight_aircraft': Q_FLIGHT_AIRCRAFT_TYPE,
    'frequent_flyers_route': Q_FREQUENT_FLYERS_ON_ROUTE
}



#: Features Vector Embeddings
EMBEDDING_MODELS = {
    'minilm': 'all-MiniLM-L6-v2',
    'mpnet': 'all-mpnet-base-v2'
}

def row_to_feature_text(row: dict) -> str:
    return ', '.join([f'{key}: {value}' for key, value in row.items()])

def embed_and_rank(query: str, rows: list, model: str = 'minilm'):
    if not rows: return None

    model = SentenceTransformer(EMBEDDING_MODELS.get('minilm'))

    row_embeddings = model.encode([row_to_feature_text(r) for r in rows], convert_to_numpy=True)
    query_embedding = model.encode([query], convert_to_numpy=True)

    index = faiss.IndexFlatIP(row_embeddings.shape[1])
    index.add(row_embeddings)

    return [rows[i] for i in index.search(query_embedding, len(rows))[1][0]]

def retrieve(intent: str, entities: dict, query: str, embedder: str = None):
    if intent not in TEMPLATES: return [{'message': f"No specific query template for intent '{intent}'. Please rephrase your question."}]
    
    try: results = run_query(TEMPLATES[intent], entities)
    except Exception as e: return [{'error': f"Query failed: {str(e)}"}]

    if results: return embed_and_rank(query=query, rows=results, model=embedder) if embedder in EMBEDDING_MODELS else results
    else: return [{'message': "No matching data found in the knowledge graph."}]
