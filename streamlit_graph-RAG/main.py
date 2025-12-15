from preprocessing import classify_intent, extract_entities
from rag import retrieve, EMBEDDING_MODELS
from llm import chat, LLM_MODELS, ANSWER_QUERY
import streamlit as st



st.set_page_config(page_title="Airline Graph-RAG Assistant", layout="wide")
st.title("Airline Graph-RAG Assistant")

question = st.text_input("Ask a question about flights, routes, delays...")

col1, col2 = st.columns(2)
with col1:
    llm_model_key = st.selectbox('LLM Model', options=list(LLM_MODELS.keys()), index=0)
with col2:
    embedding_mode = st.selectbox('Embedding Model', options=['off'] + list(EMBEDDING_MODELS.keys()), index=0)



if st.button("Ask") and question.strip():
    intent = classify_intent(question, model=llm_model_key)
    st.write("Detected intent: ", intent)

    entities = extract_entities(intent, question, model=llm_model_key)
    st.write("Extracted entities: ", entities)

    
    embedding_model = embedding_mode if embedding_mode != 'off' else None
    results = retrieve(intent=intent, entities=entities, query=question, embedder=embedding_model, top_k=entities.get('limit', 5))

    # Display results side by side
    col1, col2 = st.columns(2)
    
    # Display Baseline Results
    with col1:
        st.subheader("Baseline Query Results")
        baseline_results = results.get('baseline', [])
        if baseline_results:
            if isinstance(baseline_results, list) and len(baseline_results) > 0:
                if 'error' in baseline_results[0] or 'message' in baseline_results[0]:
                    st.warning(baseline_results[0].get('error') or baseline_results[0].get('message'))
                else:
                    st.json(baseline_results)
            else:
                st.info("No baseline results available.")
        else:
            st.info("No baseline results available.")

    # Display Embedding Results
    with col2:
        st.subheader("Embedding Similarity Results")
        embedding_results = results.get('embedding', [])
        if embedding_results:
            if isinstance(embedding_results, list) and len(embedding_results) > 0:
                if 'error' in embedding_results[0] or 'message' in embedding_results[0]:
                    st.warning(embedding_results[0].get('error') or embedding_results[0].get('message'))
                else:
                    st.json(embedding_results)
            else:
                st.info("No embedding results available.")
        else:
            st.info("No embedding results available.")

    # Prepare contexts for LLM
    baseline_context = ""
    if baseline_results and isinstance(baseline_results, list):
        valid_baseline = [x for x in baseline_results if not ('error' in x or 'message' in x)]
        if valid_baseline:
            baseline_context = '\n'.join([str(x) for x in valid_baseline])
        else:
            baseline_context = str(baseline_results[0]) if baseline_results else "No baseline results."
    else:
        baseline_context = "No baseline results."
    
    embedding_context = ""
    if embedding_results and isinstance(embedding_results, list):
        valid_embedding = [x for x in embedding_results if not ('error' in x or 'message' in x)]
        if valid_embedding:
            embedding_context = '\n'.join([str(x) for x in valid_embedding[:5]])  # Top 5 similar journeys
        else:
            embedding_context = str(embedding_results[0]) if embedding_results else "No embedding results."
    else:
        embedding_context = "No embedding results."

    answer = chat(ANSWER_QUERY.format(
        baseline_context=baseline_context,
        embedding_context=embedding_context,
        query=question
    ), model=llm_model_key, temperature=0.5)

    st.subheader("LLM Answer")
    st.write(answer)
