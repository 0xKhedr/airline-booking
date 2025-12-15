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

You MUST answer using ONLY the information in the CONTEXT sections below.

CONTEXT STRUCTURE:
1. BASELINE CONTEXT: Statistical aggregations and query-specific results (e.g., averages, counts, rankings)
2. EMBEDDING CONTEXT: Semantically similar individual journey examples from vector search

IMPORTANT RULES:
- Prioritize BASELINE CONTEXT for direct answers to the question's intent
- Use EMBEDDING CONTEXT to provide concrete examples, additional context, or when baseline is insufficient
- Both contexts were already filtered by the query - assume they satisfy the question's constraints
- Combine insights from both contexts for a comprehensive answer
- If baseline provides statistics, use embedding examples to illustrate them
- Returned rows are factual evidence

You MUST respond exactly with:
"The KG does not contain the answer."
ONLY IF:
- Both contexts are empty or contain only errors/messages

Do NOT invent facts.
Do NOT contradict the provided data.

BASELINE CONTEXT (Query-specific results with statistics and example_journeys):
{baseline_context}

EMBEDDING CONTEXT (Semantically similar journeys with similarity scores):
{embedding_context}

QUESTION:
{query}

TASK:
Provide a comprehensive, conversational answer that:
1. Directly answers the question using baseline statistics/results
2. Enriches the answer with embedding examples when available
3. References specific flight numbers, routes, and metrics
4. For ranking questions, use the provided numeric values
5. Maintains consistency between both contexts
'''



if __name__ == '__main__':
    prompt = "Hello"
    response = chat(prompt, model='openai')

    print(response)
