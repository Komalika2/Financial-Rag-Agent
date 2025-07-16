import google.generativeai as genai
import os

# Ensure your GOOGLE_API_KEY is set as an environment variable
# For quick testing, you can temporarily uncomment and set it directly,
# but for security, environment variables are preferred.
# os.environ["GOOGLE_API_KEY"] = "YOUR_ACTUAL_GEMINI_API_KEY_HERE"

api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set it before running this script.")
    print("Example (Windows CMD): set GOOGLE_API_KEY=YOUR_API_KEY")
    print("Example (Linux/macOS Bash): export GOOGLE_API_KEY=YOUR_API_KEY")
    exit()

genai.configure(api_key=api_key)

print("Listing available Gemini models:")
try:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"  {m.name}")
except Exception as e:
    print(f"An error occurred while listing models: {e}")
    print("This could be due to an invalid API key, network issue, or region restriction.")