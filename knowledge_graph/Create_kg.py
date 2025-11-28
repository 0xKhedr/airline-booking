import pandas as pd
from neo4j import GraphDatabase



#: Data Loading
df = pd.read_csv('./Airline_surveys_sample.csv')



#: Nodes
passengers = df[[
    'record_locator',
    'loyalty_program_level',
    'generation'
]].drop_duplicates(['record_locator'])

journeys = df[[
    'feedback_ID',
    'food_satisfaction_score',
    'arrival_delay_minutes',
    'actual_flown_miles',
    'number_of_legs',
    'passenger_class'
]].drop_duplicates(['feedback_ID'])

flights = df[[
    'flight_number',
    'fleet_type_description'
]].drop_duplicates(['flight_number', 'fleet_type_description'])

airports = pd.concat([
    df[['origin_station_code']].rename(columns={'origin_station_code': 'station_code'}),
    df[['destination_station_code']].rename(columns={'destination_station_code': 'station_code'})
]).drop_duplicates()



#: Relations
took = df[[ # Passenger TOOK Journey
    'record_locator',
    'feedback_ID'
]].drop_duplicates()

on = df[[ # Journey ON Flight
    'feedback_ID',
    'flight_number',
    'fleet_type_description'
]].drop_duplicates()

departs_from = df[[ # Flight DEPARTS_FROM Airport
    'flight_number',
    'fleet_type_description',
    'origin_station_code'
]].drop_duplicates().rename(columns={'origin_station_code': 'station_code'})

arrives_at = df[[ # Flight ARRIVES_AT Airport
    'flight_number',
    'fleet_type_description',
    'destination_station_code'
]].drop_duplicates().rename(columns={'destination_station_code': 'station_code'})



#: Neo4j Build Knowledge Graph
batch_size = 1000

def build_kg(tx):
    #: Nuke
    tx.run("""
        MATCH (n)
        DETACH DELETE n
    """)

    #: Passengers
    passenger_records = passengers.to_dict('records')
    for i in range(0, len(passenger_records), batch_size):
        batch = passenger_records[i:i + batch_size]
        tx.run("""
            UNWIND $passengers AS passenger
            MERGE (p:Passenger {record_locator: passenger.record_locator})
            SET p.loyalty_program_level = passenger.loyalty_program_level,
                p.generation = passenger.generation
        """, passengers=batch)

    #: Journeys
    journey_records = journeys.to_dict('records')
    for i in range(0, len(journey_records), batch_size):
        batch = journey_records[i:i + batch_size]
        tx.run("""
            UNWIND $journeys AS journey
            MERGE (j:Journey {feedback_ID: journey.feedback_ID})
            SET j.food_satisfaction_score = journey.food_satisfaction_score,
                j.arrival_delay_minutes = journey.arrival_delay_minutes,
                j.actual_flown_miles = journey.actual_flown_miles,
                j.number_of_legs = journey.number_of_legs,
                j.passenger_class = journey.passenger_class
        """, journeys=batch)

    #: Flights
    flight_records = flights.to_dict('records')
    for i in range(0, len(flight_records), batch_size):
        batch = flight_records[i:i + batch_size]
        tx.run("""
            UNWIND $flights AS flight
            MERGE (f:Flight {flight_number: flight.flight_number, fleet_type_description: flight.fleet_type_description})
        """, flights=batch)

    #: Airports
    airport_records = airports.to_dict('records')
    for i in range(0, len(airport_records), batch_size):
        batch = airport_records[i:i + batch_size]
        tx.run("""
            UNWIND $airports AS airport
            MERGE (a:Airport {station_code: airport.station_code})
        """, airports=batch)

    #: TOOK
    took_records = took.to_dict('records')
    for i in range(0, len(took_records), batch_size):
        batch = took_records[i:i + batch_size]
        tx.run("""
            UNWIND $took AS relation
            MATCH (p:Passenger {record_locator: relation.record_locator})
            MATCH (j:Journey {feedback_ID: relation.feedback_ID})
            MERGE (p)-[:TOOK]->(j)
        """, took=batch)

    #: ON
    on_records = on.to_dict('records')
    for i in range(0, len(on_records), batch_size):
        batch = on_records[i:i + batch_size]
        tx.run("""
            UNWIND $on AS relation
            MATCH (j:Journey {feedback_ID: relation.feedback_ID})
            MATCH (f:Flight {flight_number: relation.flight_number, fleet_type_description: relation.fleet_type_description})
            MERGE (j)-[:ON]->(f)
        """, on=batch)

    #: DEPARTS_FROM
    departs_from_records = departs_from.to_dict('records')
    for i in range(0, len(departs_from_records), batch_size):
        batch = departs_from_records[i:i + batch_size]
        tx.run("""
            UNWIND $departs_from AS relation
            MATCH (f:Flight {flight_number: relation.flight_number, fleet_type_description: relation.fleet_type_description})
            MATCH (a:Airport {station_code: relation.station_code})
            MERGE (f)-[:DEPARTS_FROM]->(a)
        """, departs_from=batch)

    #: ARRIVES_AT
    arrives_at_records = arrives_at.to_dict('records')
    for i in range(0, len(arrives_at_records), batch_size):
        batch = arrives_at_records[i:i + batch_size]
        tx.run("""
            UNWIND $arrives_at AS relation
            MATCH (f:Flight {flight_number: relation.flight_number, fleet_type_description: relation.fleet_type_description})
            MATCH (a:Airport {station_code: relation.station_code})
            MERGE (f)-[:ARRIVES_AT]->(a)
        """, arrives_at=batch)



#: Neo4j Write
def load_config(path):
    config = {}
    with open(path) as file:
        for line in file:
            line = line.strip()
            if line and '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()

    return config

config = load_config('config.txt')

with GraphDatabase.driver(config['URI'], auth=(config['USERNAME'], config['PASSWORD'])) as driver:
    with driver.session() as session:
        session.execute_write(build_kg) 
