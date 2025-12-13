from llm import call_llm, INTENT_PROMPT
import re
import spacy


# Load spaCy small model (requires: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm") from e

ALLOWED_INTENTS = {
    "search_flight",
    "route_delay_analysis",
    "airline_insights",
    "passenger_feedback",
    "compare_routes",
    "recommend_route",
    "airport_info",
    "generic_question",
}

def classify_intent(question: str) -> str:
    """
    Use LLM to classify intent. Returns one token from ALLOWED_INTENTS,
    fallback to 'generic_question' if result invalid.
    """
    prompt = INTENT_PROMPT.format(question=question)
    try:
        raw = call_llm(prompt, max_tokens=32)
        intent = raw.strip().splitlines()[0].strip()
    except Exception:
        # If LLM call fails, fallback to simple heuristic
        intent = _heuristic_intent(question)
    if intent not in ALLOWED_INTENTS:
        intent = _heuristic_intent(question)
    return intent

def _heuristic_intent(text: str) -> str:
    t = text.lower()
    if "from" in t and "to" in t:
        return "search_flight"
    if "delay" in t or "late" in t:
        return "route_delay_analysis"
    if "airline" in t or "company" in t:
        return "airline_insights"
    if "flight" in t and any(ch.isdigit() for ch in t):
        return "passenger_feedback"
    if "compare" in t:
        return "compare_routes"
    if "recommend" in t or "suggest" in t:
        return "recommend_route"
    if len(t.split()) < 4:
        return "generic_question"
    return "generic_question"

def extract_entities(text: str) -> dict:
    """
    Returns:
      {
        "iata_codes": [str],
        "flight_numbers": [str],
        "dates": [str],
        "spacy": [(text,label), ...]
      }
    """
    entities = {
        "iata_codes": re.findall(r"\b([A-Z]{3})\b", text),
        "flight_numbers": re.findall(r"\b([A-Z]{2}\d{2,4})\b", text),
        "dates": re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", text),
        "spacy": []
    }
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ("ORG", "GPE", "PERSON", "LOC", "DATE"):
            entities["spacy"].append((ent.text, ent.label_))
    return entities
