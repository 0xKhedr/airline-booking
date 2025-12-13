from preprocessing import classify_intent, extract_entities
from rag import retrieve
from llm import build_answer_prompt, call_llm, LLM_MODELS
import streamlit as st



st.set_page_config(page_title="Airline Graph-RAG Assistant", layout="centered")
st.title("Airline Graph-RAG Assistant")



llm_model_key = st.selectbox(
    'LLM Model',
    options=list(LLM_MODELS.keys()),
    index=0
)

embedding_mode = st.selectbox(
    'Embedding Model',
    options=['off', 'miniLM', 'mpnet'],
    index=0
)

question = st.text_input("Ask a question about flights, routes, delays, passengers...")



if st.button("Ask") and question.strip():
    intent = classify_intent(question)
    st.write("Detected intent: ", intent)

    entities = extract_entities(question)
    st.write("Extracted entities: ", entities)

    
    use_embeddings = embedding_mode != 'off'
    embedding_model = embedding_mode if use_embeddings else None

    results = retrieve(
        intent=intent,
        entities=entities,
        user_query=question,
        use_embeddings=use_embeddings,
        model_name=embedding_model
    )

    st.subheader("Retrieved KG Results")
    st.write(results or "No results found.")

    
    context_items = results[:5] if isinstance(results, list) else []
    context_text = '\n'.join([str(x) for x in context_items])

    prompt = build_answer_prompt(question, context_text)
    answer = call_llm(
        prompt,
        model_key=llm_model_key
    )

    st.subheader("LLM Answer (grounded in KG)")
    st.write(answer)
