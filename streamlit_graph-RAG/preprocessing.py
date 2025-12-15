from rag import TEMPLATES
from llm import chat, INTENT_QUERY, ENTITIES_QUERY
import json
import re



def classify_intent(query: str, model: str) -> str:
    response = chat(INTENT_QUERY.format(query=query), model).strip()
    valid_intents = set(list(TEMPLATES.keys()))
    
    for match in re.findall(r'[a-z_]+', response.lower()):
        if match in valid_intents: return match
    else: return response.lower()

def extract_entities(intent: str, query: str, model: str) -> dict:
    response = chat(ENTITIES_QUERY.format(intent=intent.strip(), query=query), model)
    
    entities = {}
    if json_match := re.search(r'[\{\[].*[\}\]]', response, re.DOTALL):
        try: entities = json.loads(json_match.group(0))
        except json.JSONDecodeError: pass
    else:
        try: entities = json.loads(response)
        except json.JSONDecodeError as e: return {'error': f"Failed to parse JSON: {str(e)}", 'raw_response': response}
    
    if entities.get('flight_number'):
        try: entities['flight_number'] = int(entities['flight_number'])
        except (ValueError, TypeError): pass
    
    return entities



if __name__ == '__main__':
    query = "Top 5 least crowded flights from DEX to DFX"
    intent = classify_intent(query, 'meta')
    entities = extract_entities(intent, query, 'meta')

    print(intent)
    print(entities)
