from huggingface_hub import InferenceClient
from config import HUGGINGFACE_TOKEN
from typing import Optional



LLM_MODELS = {
    "gemma": "google/gemma-2-2b-it",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "phi": "microsoft/phi-2"
}

def call_llm(prompt: str, model: Optional[str] = 'gemma', max_tokens: int = 256, temperature: float = 0.2) -> str:
    resp = InferenceClient(token=HUGGINGFACE_TOKEN, model=LLM_MODELS[model]).chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )

    try:
        return resp.choices[0].message["content"]
    except Exception:
        return str(resp)

INTENT_PROMPT = """
You are an intent classification assistant specialized in airline queries.
Choose ONLY ONE of the following intents and return exactly that token (no extra text):

[intents] = [
  least_delayed_flights,
  least_crowded_flights,
  best_food_routes,
  popular_airports,
  flight_has_class,
  route_distance,
  dominant_generation_airport,
  direct_flight,
  flight_aircraft,
  frequent_flyers_route,
  generic_question
]

User question:
---
{question}
---

Return exactly one token from the list above.
"""

def build_answer_prompt(question: str, context: str) -> str:
    return f"""
You are an Airline Insights Assistant. Use ONLY the facts provided in the CONTEXT section below.
If the answer is not present in the context, respond exactly: "The KG does not contain the answer."

CONTEXT:
{context}

QUESTION:
{question}

TASK:
Answer the user's question concisely, refer to flight numbers or airports from the context when relevant, and do not invent facts.
"""
