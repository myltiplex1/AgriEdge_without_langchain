import os
import requests
from logger import get_logger

logger = get_logger(__name__)

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Load prompt from external file
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompt.txt")

def load_prompt_template():
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt file {PROMPT_FILE} not found.")
        raise
    except Exception as e:
        logger.error(f"Error loading prompt file: {e}")
        raise

# Load prompt template
prompt_template = load_prompt_template()

def query_ollama(user_query: str, sensor_data: dict = None, context: str = "") -> str:
    logger.info("Querying LLM with user query and context.")
    sensor_status = ""
    if sensor_data:
        # Use only the latest 3 timestamps
        timestamps = sorted(sensor_data.keys())[-3:]  # Last 3 entries
        sensor_lines = ["Sensor Data (latest 3 readings):"]
        for ts in timestamps:
            values = sensor_data[ts]
            sensor_lines.append(f"\nTimestamp: {ts}")
            for category, readings in values.items():
                sensor_lines.append(f"{category.capitalize()}:")
                for metric, val in readings.items():
                    sensor_lines.append(f"  {metric}: {val}")
        
        # Summarize trends
        sensor_lines.append("\nTrends and Changes Summary:")
        try:
            if len(timestamps) == 3:
                t1, t2, t3 = timestamps  # Oldest to newest
                for category in ["soil", "water", "environment"]:
                    sensor_lines.append(f"\n{category.capitalize()} Trends:")
                    for metric in sensor_data[t3][category]:
                        try:
                            v1 = sensor_data[t1][category][metric]
                            v2 = sensor_data[t2][category][metric]
                            v3 = sensor_data[t3][category][metric]

                            def to_float(v):
                                return float(str(v).replace("°C", "").replace("NTU", "").replace("%", "").strip())

                            fv1, fv2, fv3 = to_float(v1), to_float(v2), to_float(v3)
                            trend = "increased" if fv3 > fv1 else "decreased" if fv3 < fv1 else "remained stable"
                            sensor_lines.append(f"  {metric}: {trend} (from {v1} → {v2} → {v3})")
                        except:
                            trend = "changed" if v1 != v3 else "unchanged"
                            sensor_lines.append(f"  {metric}: {trend} (from '{v1}' → '{v2}' → '{v3}')")
        except Exception as e:
            logger.warning(f"Trend analysis failed: {e}")
        
        sensor_status = "\n".join(sensor_lines)

    # Format prompt
    try:
        prompt = prompt_template.format(
            sensor_status=sensor_status,
            rag_context=context,
            user_query=user_query
        )
        # Query Ollama API
        payload = {
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info("Received response from LLM.")
        return result.get("response", "Error: No response from LLM.")
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return f"Error: Could not process query. Please try again."