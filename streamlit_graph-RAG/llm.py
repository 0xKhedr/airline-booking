from config import CONFIG
from huggingface_hub import InferenceClient



LLM_MODELS = {
    'deepseek': 'deepseek-ai/DeepSeek-V3.2',
    'qwen': 'Qwen/Qwen3-4B-Instruct-2507',
    'openai': 'openai/gpt-oss-20b',
    'google': 'google/gemma-2-2b-it',
    'meta': 'meta-llama/Llama-3.2-1B-Instruct'
}

def chat(prompt: str, model: str = 'deepseek', max_tokens: int = 1024, temperature: float = 0.0) -> str:
    response = InferenceClient(token=CONFIG.get('HUGGINGFACE_TOKEN'), model=LLM_MODELS.get(model)).chat_completion(
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )

    return response.choices[0].message['content']



INTENT_QUERY = '''
Classify the user's question into exactly one of these intents:
1. most_delayed_flights
2. least_crowded_flights  
3. worst_food_routes
4. popular_airports
5. flight_classes_offered
6. route_distance
7. dominant_generation_airport
8. direct_flight
9. flight_aircraft
10. frequent_flyers_route

Output only the intent name, nothing else.

Examples:
Input: "Which flights from JFK to LAX have the most delays?" → most_delayed_flights
Input: "What classes are available on flight 2400?" → flight_classes_offered
Input: "Show me the busiest airports" → popular_airports

Now classify: {query}
'''

ENTITIES_QUERY = '''
Extract parameters for this intent: {intent}

Extract these parameters from the question:
- origin: 3-letter airport code or null
- destination: 3-letter airport code or null  
- flight_number: integer flight number or null (extract only the numeric part)
- code: airport code for dominant_generation_airport or null
- limit: integer (default 5)

Output only valid JSON in this exact format:
{{"origin": "CODE_OR_NULL", "destination": "CODE_OR_NULL", "flight_number": NUMBER_OR_NULL, "code": "CODE_OR_NULL", "limit": NUMBER}}

Examples:
Intent: most_delayed_flights, Question: "Top 10 delayed flights from JFK to LAX" → {{"origin": "JFK", "destination": "LAX", "flight_number": null, "code": null, "limit": 10}}
Intent: flight_classes_offered, Question: "Classes on flight UA123" → {{"origin": null, "destination": null, "flight_number": 123, "code": null, "limit": 5}}
Intent: popular_airports, Question: "Busiest airports" → {{"origin": null, "destination": null, "flight_number": null, "code": null, "limit": 5}}
Intent: dominant_generation_airport, Question: "Main generation at LAX" → {{"origin": null, "destination": null, "flight_number": null, "code": "LAX", "limit": 5}}

Now extract from: {query}
'''

ANSWER_QUERY = '''
You are an Airline Graph-RAG Assistant.

CRITICAL: The BASELINE CONTEXT below contains results that were ALREADY FILTERED by a specialized database query to answer the user's question. If the baseline context contains valid data (not errors), those results ARE the answer.

CONTEXT TYPES:
1. BASELINE CONTEXT: Pre-filtered query results with statistics, rankings, and example_journeys that directly answer the question
2. EMBEDDING CONTEXT: Additional similar journey examples from semantic search (may contain errors if not configured)

ANSWERING RULES:
1. IGNORE any error messages or "Invalid embedder" messages in EMBEDDING CONTEXT
2. If BASELINE CONTEXT has valid data (dictionaries with flight/route/statistics), answer the question using that data
3. The baseline results are already ranked/filtered - trust the ordering and values provided
4. Use EMBEDDING CONTEXT only as supplementary examples if it contains valid journey data
5. For ranking questions (least/most/top), the baseline results are already in the correct order

RESPOND WITH "The KG does not contain the answer." ONLY IF:
- BASELINE CONTEXT says "No matching data" or "No baseline results"
- BASELINE CONTEXT is empty or contains only error messages

DO NOT say "no data available" if baseline context has dictionaries with flight numbers, airports, or statistics.

BASELINE CONTEXT (Pre-filtered answer to the question):
{baseline_context}

EMBEDDING CONTEXT (Supplementary examples):
{embedding_context}

QUESTION:
{query}

INSTRUCTIONS:
- Answer directly using the baseline results
- Reference specific flight numbers, routes, and metrics from the data
- Add embedding examples only if they provide additional useful context
- Be concise and conversational
- Trust that the data is already filtered for the question's intent
'''



if __name__ == '__main__':
    prompt = "Hello"
    response = chat(prompt, model='openai')

    print(response)
