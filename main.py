import sys
import os

# Add the project root to the system path to allow module imports
# This helps Python find modules in 'Scripts/' when main.py is run directly.
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# --- Debug Info ---
print(f"Current working directory (os.getcwd()): {os.getcwd()}")
print(f"Current file absolute path: {os.path.abspath(__file__)}")
print(f"Calculated project root for sys.path: {project_root}")
print(f"sys.path after modification: {sys.path}")
print(f"--- End Debug Info ---")

import json
from Scripts.embed_and_store import RAGPipeline
from Scripts.query_agent import FinancialAgent
import google.generativeai as genai

# --- Configuration ---
# IMPORTANT: Replace with your actual Google Gemini API Key if not using environment variable
# It's recommended to set this as an environment variable (e.g., GOOGLE_API_KEY)
# For local testing, you can put it here directly, but NEVER commit your API key to public repositories!
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY_HERE") # Make sure your actual key is here or in env var

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

# Ensure outputs directory exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def run_challenge():
    """
    Main function to run the RAG Sprint Challenge.
    """
    print("\n--- Starting RAG Sprint Challenge ---")

    # 1. Initialize RAG Pipeline (Embeddings & Vector Store)
    print("\nInitializing RAG Pipeline (Embeddings & Vector Store)...")
    rag_pipeline = RAGPipeline()
    if not rag_pipeline.load_or_create_index():
        print("Failed to load or create FAISS index. Exiting.")
        return

    # 2. Initialize Financial Agent
    print("\nInitializing Financial Agent...")
    try:
        agent = FinancialAgent(google_api_key=GOOGLE_API_KEY, rag_pipeline=rag_pipeline)
    except ValueError as e:
        print(f"Agent initialization failed: {e}")
        print("Please ensure GOOGLE_API_KEY environment variable is set, or replace 'YOUR_GEMINI_API_KEY_HERE' in main.py.")
        return

    # 3. Define and Run Test Queries
    test_queries = [
        # Simple queries
        "What was NVIDIA's total revenue in fiscal year 2024?",
        "What percentage of Google's 2023 revenue came from advertising?",
        # Comparative queries (require agent decomposition)
        "How much did Microsoft's cloud revenue grow from 2022 to 2023?",
        "Which of the three companies had the highest gross margin in 2023?",
        # Complex multi-step queries
        "Compare the R&D spending as a percentage of revenue across all three companies in 2023",
        "How did each company's operating margin change from 2022 to 2024?",
        "What are the main AI risks mentioned by each company and how do they differ?"
    ]

    all_results = []
    for i, query in enumerate(test_queries):
        print(f"\n--- Running Test Query {i+1}/{len(test_queries)} ---")
        result = agent.answer_query(query)
        all_results.append(result)
        # Using json.dumps for pretty printing the dictionary output
        print(json.dumps(result, indent=2)) 

    # 4. Save All Results to JSON
    output_file_path = os.path.join(OUTPUTS_DIR, 'challenge_results.json')
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n🎉 All test query results saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

if __name__ == "__main__":
    run_challenge()