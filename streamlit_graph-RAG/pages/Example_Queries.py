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
    st.code("Which flights from JFK to LAX have the most delays?")
    st.code("Show me the top 5 most delayed flights from ORD to DFW")
    
    st.subheader("👥 Passenger Information")
    st.code("What are the least crowded flights from SFO to SEA?")
    st.code("Do frequent flyers use the route from ATL to MIA?")
    st.code("What is the dominant passenger generation at LAX?")
    
    st.subheader("🍽️ Food & Service")
    st.code("Which routes have the worst food ratings?")
    st.code("Show me the top 10 routes with the worst food")

with col2:
    st.subheader("🛫 Flight Details")
    st.code("What classes are available on flight 2400?")
    st.code("Which aircraft will I be flying on flight 1532?")
    st.code("Can I go from BOS to DEN directly?")
    
    st.subheader("📍 Airports & Routes")
    st.code("What are the most popular airports?")
    st.code("Show me the top 3 busiest airports")
    st.code("What is the flight distance from DCA to PHX?")

st.markdown("---")

st.info("💡 **Tip**: You can modify these examples by changing airport codes, flight numbers, or limits to suit your needs!")

# Optional: Add a section showing valid airport codes
with st.expander("📍 Common Airport Codes"):
    st.markdown("""
    - **JFK** - John F. Kennedy International (New York)
    - **LAX** - Los Angeles International
    - **ORD** - Chicago O'Hare International
    - **DFW** - Dallas/Fort Worth International
    - **ATL** - Hartsfield-Jackson Atlanta International
    - **SFO** - San Francisco International
    - **SEA** - Seattle-Tacoma International
    - **MIA** - Miami International
    - **BOS** - Boston Logan International
    - **DEN** - Denver International
    - **DCA** - Ronald Reagan Washington National
    - **PHX** - Phoenix Sky Harbor International
    """)
