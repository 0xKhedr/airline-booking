from config import CONFIG
import pandas as pd
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase



EMBEDDING_MODELS = {
    'minilm': {
        'name': 'all-MiniLM-L6-v2',
        'dim': 384,
    },
    'mpnet': {
        'name': 'all-mpnet-base-v2',
        'dim': 768,
    }
}



if __name__ == '__main__':
    NEO4J_DRIVER = GraphDatabase.driver(CONFIG.get('NEO4J_URI'), auth=(CONFIG.get('NEO4J_USERNAME'), CONFIG.get('NEO4J_PASSWORD')))

    with NEO4J_DRIVER as driver:
        with driver.session() as session:
            result = session.run('''
            MATCH (j:Journey)-[:ON]->(f:Flight)
            MATCH (f)-[:DEPARTS_FROM]->(o:Airport)
            MATCH (f)-[:ARRIVES_AT]->(d:Airport)
            RETURN 
                j.feedback_ID as id, 
                j.passenger_class as class, 
                j.food_satisfaction_score as food, 
                j.arrival_delay_minutes as delay,
                j.actual_flown_miles as miles,
                f.flight_number as flight,
                f.fleet_type_description as aircraft,
                o.station_code as origin,
                d.station_code as destination
            ''')

            df = pd.DataFrame([r.data() for r in result])

        df['description'] = df.apply(lambda row: (
            f"A {row['class']} class journey on flight {row['flight']} ({row['aircraft']}) "
            f"from {row['origin']} to {row['destination']}. "
            f"The flight covered {row['miles']} miles. "
            f"Passenger food rating: {row['food']}/5. "
            f"Arrival delay: {row['delay']} minutes."
        ), axis=1)

        for name, model in EMBEDDING_MODELS.items():
            embedder = SentenceTransformer(model['name'])
            embeddings = embedder.encode(df['description'].tolist())
            data = [{'id': row['id'], 'embedding': embeddings[i].tolist()} for i, row in df.iterrows()]
            
            batch_size = 1000
            with driver.session() as session:
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    session.run(f'''
                    UNWIND $updates AS row
                    MATCH (j:Journey {{feedback_ID: row.id}})
                    SET j.{name} = row.embedding
                    ''', updates=batch)
                    
                session.run(f'''
                CREATE VECTOR INDEX semantic_index_{name} IF NOT EXISTS
                FOR (j:Journey)
                ON (j.{name})
                OPTIONS {{indexConfig: {{
                `vector.dimensions`: {model['dim']},
                `vector.similarity_function`: 'cosine'
                }}}}
                ''')
