from config import CONFIG
from embedding import EMBEDDING_MODELS
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase



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
WITH f, o, d, 
     avg(j.arrival_delay_minutes) AS avg_delay,
     count(j) AS samples,
     collect({id: j.feedback_ID, class: j.passenger_class, food: j.food_satisfaction_score, 
              delay: j.arrival_delay_minutes, miles: j.actual_flown_miles, aircraft: f.fleet_type_description})[0..3] AS example_journeys
RETURN
    f.flight_number AS flight,
    o.station_code AS origin,
    d.station_code AS destination,
    avg_delay,
    samples,
    example_journeys
ORDER BY avg_delay DESC
LIMIT $limit
''' # Flights from airport ... to airport ... with the most delays

Q_LEAST_CROWDED_FLIGHTS_ON_ROUTE = '''
MATCH (p:Passenger)-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(o:Airport {station_code: $origin})
MATCH (f)-[:ARRIVES_AT]->(d:Airport {station_code: $destination})
WITH f, o, d, count(DISTINCT p) AS passengers,
     collect({id: j.feedback_ID, class: j.passenger_class, food: j.food_satisfaction_score,
              delay: j.arrival_delay_minutes, miles: j.actual_flown_miles, aircraft: f.fleet_type_description})[0..3] AS example_journeys
RETURN
    f.flight_number AS flight,
    o.station_code AS origin,
    d.station_code AS destination,
    passengers,
    example_journeys
ORDER BY passengers ASC
LIMIT $limit
''' # Flights from airport ... to airport ... with the least passengers

Q_FLIGHT_CLASSES_OFFERED = '''
MATCH (f:Flight {flight_number: $flight_number})
MATCH (j:Journey)-[:ON]->(f)
MATCH (f)-[:DEPARTS_FROM]->(o:Airport)
MATCH (f)-[:ARRIVES_AT]->(d:Airport)
WITH f, o, d, j.passenger_class AS class,
     collect({id: j.feedback_ID, food: j.food_satisfaction_score,
              delay: j.arrival_delay_minutes, miles: j.actual_flown_miles})[0..2] AS example_journeys
RETURN
    f.flight_number AS flight,
    o.station_code AS origin,
    d.station_code AS destination,
    f.fleet_type_description AS aircraft,
    class AS offered_class,
    example_journeys
''' # Classes offered in flight ...

Q_FLIGHT_AIRCRAFT_TYPE = '''
MATCH (f:Flight {flight_number: $flight_number})
MATCH (f)-[:DEPARTS_FROM]->(o:Airport)
MATCH (f)-[:ARRIVES_AT]->(d:Airport)
OPTIONAL MATCH (j:Journey)-[:ON]->(f)
WITH f, o, d, collect({id: j.feedback_ID, class: j.passenger_class, food: j.food_satisfaction_score,
                        delay: j.arrival_delay_minutes, miles: j.actual_flown_miles})[0..3] AS example_journeys
RETURN
    f.flight_number AS flight,
    o.station_code AS origin,
    d.station_code AS destination,
    f.fleet_type_description AS aircraft,
    example_journeys
''' # Which aircraft will I be flying on?

Q_WORST_FOOD_ROUTES = '''
MATCH (j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(o:Airport)
MATCH (f)-[:ARRIVES_AT]->(d:Airport)
WITH o, d, f, avg(j.food_satisfaction_score) AS avg_food, count(j) AS samples,
     collect({id: j.feedback_ID, class: j.passenger_class, food: j.food_satisfaction_score,
              delay: j.arrival_delay_minutes, miles: j.actual_flown_miles, 
              flight: f.flight_number, aircraft: f.fleet_type_description})[0..3] AS example_journeys
RETURN
    o.station_code AS origin,
    d.station_code AS destination,
    avg_food,
    samples,
    example_journeys
ORDER BY avg_food ASC
LIMIT $limit
''' # Routes with the worst food

Q_FREQUENT_FLYERS_ON_ROUTE = '''
MATCH (p:Passenger)-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(o:Airport {station_code: $origin})
MATCH (f)-[:ARRIVES_AT]->(d:Airport {station_code: $destination})
WHERE p.loyalty_program_level IS NOT NULL
    AND p.loyalty_program_level <> 'None'
WITH o, d, count(p) AS frequent_flyers, count(p) > 0 AS frequent_flyers_use_route,
     collect({id: j.feedback_ID, class: j.passenger_class, food: j.food_satisfaction_score,
              delay: j.arrival_delay_minutes, miles: j.actual_flown_miles,
              flight: f.flight_number, aircraft: f.fleet_type_description,
              loyalty_level: p.loyalty_program_level})[0..3] AS example_journeys
RETURN
    o.station_code AS origin,
    d.station_code AS destination,
    frequent_flyers_use_route,
    frequent_flyers,
    example_journeys
''' # Do frequent flyers use this route?

Q_MOST_POPULAR_AIRPORTS = '''
MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(a:Airport)
WITH a, count(j) AS journeys,
     collect({id: j.feedback_ID, class: j.passenger_class, food: j.food_satisfaction_score,
              delay: j.arrival_delay_minutes, miles: j.actual_flown_miles,
              flight: f.flight_number, aircraft: f.fleet_type_description})[0..3] AS example_journeys
RETURN
    a.station_code AS airport,
    journeys,
    example_journeys
ORDER BY journeys DESC
LIMIT $limit
''' # Most popular airports

Q_ROUTE_DISTANCE = '''
MATCH (j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(o:Airport {station_code: $origin})
MATCH (f)-[:ARRIVES_AT]->(d:Airport {station_code: $destination})
WITH o, d, avg(j.actual_flown_miles) AS avg_distance_miles, count(j) AS samples,
     collect({id: j.feedback_ID, class: j.passenger_class, food: j.food_satisfaction_score,
              delay: j.arrival_delay_minutes, miles: j.actual_flown_miles,
              flight: f.flight_number, aircraft: f.fleet_type_description})[0..3] AS example_journeys
RETURN
    o.station_code AS origin,
    d.station_code AS destination,
    avg_distance_miles,
    samples,
    example_journeys
''' # Flight distance from airport ... to airport ...

Q_HAS_DIRECT_FLIGHT = '''
MATCH (j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(o:Airport {station_code: $origin})
MATCH (f)-[:ARRIVES_AT]->(d:Airport {station_code: $destination})
WHERE j.number_of_legs = 1
WITH o, d, count(j) AS direct_flights, count(j) > 0 AS direct_flight_available,
     collect({id: j.feedback_ID, class: j.passenger_class, food: j.food_satisfaction_score,
              delay: j.arrival_delay_minutes, miles: j.actual_flown_miles,
              flight: f.flight_number, aircraft: f.fleet_type_description})[0..3] AS example_journeys
RETURN
    o.station_code AS origin,
    d.station_code AS destination,
    direct_flight_available,
    direct_flights,
    example_journeys
''' # Can I go from airport ... to airport ... directly?

Q_DOMINANT_GENERATION_AT_AIRPORT = '''
MATCH (p:Passenger)-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(a:Airport {station_code: $code})
WITH a, p.generation AS generation, count(p) AS passengers,
     collect({id: j.feedback_ID, class: j.passenger_class, food: j.food_satisfaction_score,
              delay: j.arrival_delay_minutes, miles: j.actual_flown_miles,
              flight: f.flight_number, aircraft: f.fleet_type_description})[0..3] AS example_journeys
RETURN
    a.station_code AS airport,
    generation,
    passengers,
    example_journeys
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



def retrieve(intent: str, entities: dict, query: str, embedder: str = 'minilm', top_k: int = 5):
    result = {
        'baseline': [],
        'embedding': []
    }
    
    if intent in TEMPLATES:
        try:
            baseline_results = run_query(TEMPLATES[intent], entities)
            result['baseline'] = baseline_results if baseline_results else [{'message': "No matching data found in the knowledge graph."}]
        except Exception as e:
            result['baseline'] = [{'error': f"Query failed: {str(e)}"}]
    else:
        result['baseline'] = [{'message': f"No specific query template for intent '{intent}'. Please rephrase your question."}]
    
    if embedder and embedder in EMBEDDING_MODELS:
        try:
            model_info = EMBEDDING_MODELS[embedder]
            model = SentenceTransformer(model_info['name'])
            
            query_embedding = model.encode(query).tolist()
            similarity_query = f'''
            MATCH (j:Journey)-[:ON]->(f:Flight)
            MATCH (f)-[:DEPARTS_FROM]->(o:Airport)
            MATCH (f)-[:ARRIVES_AT]->(d:Airport)
            WHERE j.{embedder} IS NOT NULL
            WITH j, f, o, d,
                vector.similarity.cosine(j.{embedder}, $query_embedding) AS score
            WHERE score > 0.5
            RETURN 
                j.feedback_ID as id,
                j.passenger_class as class,
                j.food_satisfaction_score as food,
                j.arrival_delay_minutes as delay,
                j.actual_flown_miles as miles,
                f.flight_number as flight,
                f.fleet_type_description as aircraft,
                o.station_code as origin,
                d.station_code as destination,
                score
            ORDER BY score DESC
            LIMIT $top_k
            '''
            
            embedding_results = run_query(similarity_query, {
                'query_embedding': query_embedding,
                'top_k': top_k
            })
            
            result['embedding'] = embedding_results if embedding_results else [{'message': "No similar journeys found."}]
            
        except Exception as e:
            result['embedding'] = [{'error': f"Embedding search failed: {str(e)}"}]
    else:
        result['embedding'] = [{'message': f"Invalid or no embedder specified. Available: {list(EMBEDDING_MODELS.keys())}"}]
    
    return result
