import streamlit as st
import json
from llm.ollama_llm import query_ollama
from llm.rag_pipeline import retrieve_context
from logger import get_logger

logger = get_logger(__name__)

# Function to load latest sensor data (same as main.py)
def get_latest_sensor_data(path="data/farm_data_log.json", num_entries=3):
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return data[-num_entries:] if data else []
    except FileNotFoundError:
        logger.error(f"Sensor data file {path} not found.")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        return []

# Streamlit app configuration
st.set_page_config(
    page_title="AgriEdge: A Smart Farm Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        transition: background 0.2s;
    }
    .stButton>button:hover,
    .stButton>button:focus,
    .stButton>button:active {
        background-color: #388e3c !important; /* darker green */
        color: white !important;
        border-color: #388e3c !important;
        box-shadow: none !important;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        border: 1px solid #4CAF50;
    }
    .response-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #e8f5e9;
    }
    h1, h2, h3 { color: #2e7d32; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for query history
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Main app layout
st.title("🌾 AgriEdge: A Smart Farm Assistant")
st.markdown("Ask about your farm's conditions, and get tailored advice based on real-time sensor data and agricultural knowledge.")

# Sidebar for sensor data and query history
with st.sidebar:
    st.header("Recent Sensor Data")
    sensor_data_entries = get_latest_sensor_data()
    if sensor_data_entries:
        latest_entry = sensor_data_entries[-1]
        st.markdown(f"**Latest Reading: {latest_entry['timestamp']}**")
        st.markdown("""
            **Soil**                      
                - Moisture: {}  
                - pH: {}  
                - Temperature: {}  
            **Water**  
                - pH: {}  
                - Turbidity: {}  
                - Temperature: {}  
            **Environment**  
                - Humidity: {}  
                - Temperature: {}  
                - Rainfall: {}
        """.format(
            latest_entry["soil"]["moisture"],
            latest_entry["soil"]["pH"],
            latest_entry["soil"]["temperature"],
            latest_entry["water"]["pH"],
            latest_entry["water"]["turbidity"],
            latest_entry["water"]["temperature"],
            latest_entry["environment"]["humidity"],
            latest_entry["environment"]["temperature"],
            latest_entry["environment"]["rainfall"]
        ))
    else:
        st.warning("No sensor data available. Check 'data/farm_data_log.json'.")

    st.header("Query History")
    if st.session_state.query_history:
        # Show the clear history button at the top of the section
        clear_history = st.button("Clear Query History", key="clear_history_sidebar")
        if clear_history:
            st.session_state.query_history.clear()
            st.rerun()
        for i, (query, response) in enumerate(st.session_state.query_history[-5:]):  # Show last 5 queries
            with st.expander(f"Query {i+1}: {query[:30]}..."):
                st.write(f"**Query**: {query}")
                st.write(f"**Response**: {response}")
    else:
        st.info("No queries yet.")

# Query input section
st.subheader("Ask a Question")
user_query = st.text_input(
    "Enter your farm-related question (e.g., 'What should I do about soil moisture?')",
    key="query_input"
)

col1, col2 = st.columns([0.5, 0.5])  # Reduce the gap by using smaller ratios
with col1:
    submit_button = st.button("Submit Query", use_container_width=True)
with col2:
    # Only show 'Clear Response' if there is at least one query in history
    clear_response_button = st.button("Clear Response", use_container_width=True, disabled=not st.session_state.query_history)

# Clear the last response if clear_response_button is pressed
if clear_response_button and st.session_state.query_history:
    st.session_state.query_history.pop()
    st.rerun()

# Process query and display response
if submit_button and user_query:
    logger.info("User query: %s", user_query)
    with st.spinner("Processing your query..."):
        try:
            # Prepare sensor data
            sensor_data_entries = get_latest_sensor_data()
            combined_sensor_data = {
                entry["timestamp"]: {
                    "soil": entry["soil"],
                    "water": entry["water"],
                    "environment": entry["environment"]
                }
                for entry in sensor_data_entries
            }
            # Retrieve context and query LLM
            rag_context = retrieve_context(user_query)
            response = query_ollama(user_query, combined_sensor_data, rag_context)
            logger.info("\n--- FARM ASSISTANT RESPONSE ---\n")
            
            # Display response
            st.markdown("<div class='response-box'>", unsafe_allow_html=True)
            st.markdown("### Response")
            st.markdown(response)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add to query history
            st.session_state.query_history.append((user_query, response))
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            st.error("Error: Could not process query. Please try again.")

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center; color: #666;'>Powered by AgriEdge | Built with Streamlit</p>
""", unsafe_allow_html=True)
