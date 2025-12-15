import streamlit as st

st.set_page_config(page_title="Example Queries", layout="wide")
st.title("📋 Example Queries")

st.markdown("""
Here are some example questions you can ask the Airline Graph-RAG Assistant. 
Try these to explore different aspects of flight data!
""")

st.markdown("---")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("✈️ Flight Delays & Performance")
    st.code("Which flights from LAX to IAX have the most delays?")
    st.code("Show me the top 5 most delayed flights from DEX to EWX")
    
    st.subheader("👥 Passenger Information")
    st.code("What are the least crowded flights from ORX to IAX?")
    st.code("Do frequent flyers use the route from EWX to IAX?")
    st.code("What is the dominant passenger generation at LAX?")
    
    st.subheader("🍽️ Food & Service")
    st.code("Which routes have the worst food ratings?")
    st.code("Show me the top 10 routes with the worst food")

with col2:
    st.subheader("🛫 Flight Details")
    st.code("What classes are available on flight 2411?")
    st.code("Which aircraft will I be flying on flight 924?")
    st.code("Can I go from DEX to LAX directly?")
    
    st.subheader("📍 Airports & Routes")
    st.code("What are the most popular airports?")
    st.code("Show me the top 3 busiest airports")
    st.code("What is the flight distance from DFX to ORX?")

st.markdown("---")

st.info("💡 **Tip**: You can modify these examples by changing airport codes, flight numbers, or limits to suit your needs!")

# Optional: Add a section showing valid airport codes
with st.expander("📍 Sample Airport Codes in Dataset"):
    st.markdown("""
    Some airport codes available in the dataset:
    - **LAX, IAX, EWX, DEX, DFX, ORX, SFX, PHX**
    - **HNX, AUX, BOX, LHX, FRX, PIX, CHX, SAX**
    - **SAX, ANX, MCX, PDX, GSX, RSX, LIX, BJX**
    - **And many more...**
    
    Sample flight numbers: **924, 2411, 659, 2614, 5372, 1938, 2093**
    """)
