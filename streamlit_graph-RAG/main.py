from preprocessing import classify_intent, extract_entities
from rag import retrieve, EMBEDDING_MODELS
from llm import chat, LLM_MODELS, ANSWER_QUERY
import streamlit as st



st.set_page_config(page_title="Airline Graph-RAG Assistant", layout="centered")
st.title("Airline Graph-RAG Assistant")



llm_model_key = st.selectbox('LLM Model', options=list(LLM_MODELS.keys()), index=0)
embedding_mode = st.selectbox('Embedding Model', options=['off'] + list(EMBEDDING_MODELS.keys()), index=0)
question = st.text_input("Ask a question about flights, routes, delays...")



if st.button("Ask") and question.strip():
    intent = classify_intent(question, model=llm_model_key)
    st.write("Detected intent: ", intent)

    entities = extract_entities(intent, question, model=llm_model_key)
    st.write("Extracted entities: ", entities)

    
    embedding_model = embedding_mode if embedding_mode != 'off' else None
    results = retrieve(intent=intent, entities=entities, query=question, embedder=embedding_model)

    st.subheader("Retrieved KG Results")
    st.write(results or "No results found.")

    
    context_items = results[:5] if isinstance(results, list) else []
    context_text = '\n'.join([str(x) for x in context_items])


    answer = chat(ANSWER_QUERY.format(context=context_text, query=question), model=llm_model_key, temperature=0.5)


    st.subheader("LLM Answer")
    st.write(answer)
