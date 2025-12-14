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
Input: "What classes are available on flight AA123?" → flight_classes_offered
Input: "Show me the busiest airports" → popular_airports

Now classify: {query}
'''

ENTITIES_QUERY = '''
Extract parameters for this intent: {intent}

Extract these parameters from the question:
- origin: 3-letter airport code or null
- destination: 3-letter airport code or null  
- flight_number: alphanumeric flight number or null
- code: airport code for dominant_generation_airport or null
- limit: integer (default 10)

Output only valid JSON in this exact format:
{{"origin": "CODE_OR_NULL", "destination": "CODE_OR_NULL", "flight_number": "NUMBER_OR_NULL", "code": "CODE_OR_NULL", "limit": NUMBER}}

Examples:
Intent: most_delayed_flights, Question: "Top 5 delayed flights from JFK to LAX" → {{"origin": "JFK", "destination": "LAX", "flight_number": null, "code": null, "limit": 5}}
Intent: flight_classes_offered, Question: "Classes on flight UA123" → {{"origin": null, "destination": null, "flight_number": "UA123", "code": null, "limit": 10}}
Intent: popular_airports, Question: "Busiest airports" → {{"origin": null, "destination": null, "flight_number": null, "code": null, "limit": 10}}
Intent: dominant_generation_airport, Question: "Main generation at LAX" → {{"origin": null, "destination": null, "flight_number": null, "code": "LAX", "limit": 10}}

Now extract from: {query}
'''

ANSWER_QUERY = '''
You are an Airline Graph-RAG Assistant.

You MUST answer using ONLY the information in the CONTEXT.

IMPORTANT RULES:
- The CONTEXT may contain results that were already filtered by the KG query.
- If the CONTEXT lists flights or values, you MUST assume they satisfy the constraints of the QUESTION
  (e.g., origin, destination, class, date), even if those constraints are not repeated in each row.
- Returned rows themselves are factual evidence.

You MUST answer the question IF:
- The CONTEXT contains one or more rows relevant to the QUESTION.

You MUST respond exactly with:
"The KG does not contain the answer."
ONLY IF:
- The CONTEXT is empty, OR
- The CONTEXT contains no rows relevant to the QUESTION.

Do NOT invent facts.
Do NOT require additional attributes (such as airport codes) if the question is already constrained by the query.

FILTERED, RELEVANT CONTEXT, NOT THE WHOLE DATA:
{context}

QUESTION:
{query}

TASK:
Answer concisely.
For ranking or comparison questions (e.g., least crowded),
use the provided numeric values to determine the answer.
Refer to flight numbers when available.
'''



if __name__ == '__main__':
    prompt = "Hello"
    response = chat(prompt, model='openai')

    print(response)
