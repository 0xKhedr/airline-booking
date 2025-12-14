from llm import chat, INTENT_QUERY, ENTITIES_QUERY
import json
import re



def classify_intent(query: str, model: str) -> str:
    return chat(INTENT_QUERY.format(query=query), model).strip()

def extract_entities(intent: str, query: str, model: str) -> dict:
    response = chat(ENTITIES_QUERY.format(intent=intent.strip(), query=query), model)
    
    if json_match := re.search(r'[\{\[].*[\}\]]', response, re.DOTALL):
        try: return json.loads(json_match.group(0))
        except json.JSONDecodeError: pass
    else:
        try: return json.loads(response)
        except json.JSONDecodeError as e: return {'error': f"Failed to parse JSON: {str(e)}", 'raw_response': response}



if __name__ == '__main__':
    query = "Top 5 least crowded flights from DEX to DFX"
    intent = classify_intent(query, 'google')
    entities = extract_entities(intent, query, 'google')

    print(intent)
    print(entities)
