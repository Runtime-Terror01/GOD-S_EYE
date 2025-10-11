import os
import time
from google import genai
from google.genai.errors import APIError
from typing import List, Optional

try:
    # It is recommended to manage the API key via environment variables
    # or a secure configuration system in a production environment.
    client = genai.Client(api_key="AIzaSyCLuGKM6LHq6rb8tcCo5CWgqxbG5yswqeY")
except Exception:
    pass

MODEL_NAME = "gemini-flash-latest"

# --- AI Prompting ---

SYSTEM_INSTRUCTION = (
    "You are a highly specialized Security Footage Analysis Agent. Your task is to analyze video footage "
    "and generate a two-part security report. Your response MUST be objective, factual, and strictly "
    "adhere to the requested format. Do not add any conversational text outside of the requested structure."
)

USER_QUERY = (
    "Analyze the provided video from start to finish and generate a security report in two distinct sections. "
    "**It is critical that you do not stop prematurely and analyze the entire duration of the video until the final second.**\n\n"
    "Section 1: A comprehensive, machine-readable event log suitable for a database. Use the following simplified format for each notable event or time segment:\n"
    "TIMESTAMP: [HH:MM:SS] - [HH:MM:SS]\n"
    "OBSERVATIONS: [Detailed, objective description of all events, subjects, and actions.]\n"
    "FLAGGED_ITEMS: [List any suspicious behavior, unauthorized access, weapons, or harmful objects. State 'None' if nothing is found.]\n"
    "--- (use this separator between entries) ---\n\n"
    "Section 2: An actionable, human-readable summary for a security operator. It MUST follow this exact three-part format:\n"
    "OVERALL RISK LEVEL: [None, Low, Medium, High, Critical]\n"
    "JUSTIFICATION: [A concise paragraph explaining the key events that led to the assigned risk level.]\n"
    "RECOMMENDED ACTIONS: [A bulleted list of concrete, actionable steps for a security team to take. Examples: 'Immediately escalate to law enforcement due to credible threat.', 'Review access logs for the time of the forced entry.', 'Archive for informational purposes; no immediate action required.']\n\n"
    "Begin your entire response with '### COMPREHENSIVE REPORT ###' and '### SUMMARY FOR USER ###' to delimit the sections."
)


def _parse_model_response(response_text: str) -> Optional[List[str]]:
    try:
        report_marker = "### COMPREHENSIVE REPORT ###"
        summary_marker = "### SUMMARY FOR USER ###"
        report_start_index = response_text.find(report_marker)
        summary_start_index = response_text.find(summary_marker)

        if report_start_index == -1 or summary_start_index == -1:
            return None

        report_content = response_text[report_start_index + len(report_marker):summary_start_index].strip()
        summary_content = response_text[summary_start_index + len(summary_marker):].strip()

        if not report_content or not summary_content:
            return None

        return [report_content, summary_content]
    except Exception:
        return None


def analyze_video(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found at path: {file_path}")

    uploaded_file = None
    try:
        # 1. Upload the file
        uploaded_file = client.files.upload(file=file_path)

        # 2. Wait for processing to complete
        while True:
            f_status = client.files.get(name=uploaded_file.name)
            if f_status.state == "ACTIVE":
                break
            elif f_status.state == "FAILED":
                raise Exception(f"File processing failed for file: {uploaded_file.name}")
            time.sleep(5)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"file_data": {"file_uri": uploaded_file.uri, "mime_type": uploaded_file.mime_type}},
                        {"text": USER_QUERY},
                    ],
                }
            ],
            config=genai.types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
        )

        parsed_data = _parse_model_response(response.text)
        if not parsed_data:
            raise Exception("Failed to parse the model's response. The output may be malformed.")
            
        return parsed_data

    except APIError as e:
        raise e
    except Exception as e:
        raise Exception(f"An unexpected error occurred during video analysis: {e}")

    finally:
        # 5. Ensure cleanup
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception as e:
                pass

print(analyze_video("test_video.mp4"))