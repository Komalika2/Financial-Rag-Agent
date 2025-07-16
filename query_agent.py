import os
import json
import google.generativeai as genai
from Scripts.embed_and_store import RAGPipeline # Assuming you renamed rag_pipeline.py
import re

class FinancialAgent:
    def __init__(self, google_api_key: str, rag_pipeline: RAGPipeline):
        if not google_api_key:
            raise ValueError("Google API Key must be provided for the LLM.")
        genai.configure(api_key=google_api_key)
        self.llm = genai.GenerativeModel('gemini-1.5-pro-latest')
        self.rag_pipeline = rag_pipeline
        print("FinancialAgent initialized with Gemini Pro LLM.")

    def _call_llm(self, prompt: str, temperature: float = 0.1):
        """Helper to call the LLM with a given prompt."""
        try:
            response = self.llm.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=temperature))
            return response.text.strip()
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return f"LLM error: {e}"

    def _decompose_query(self, query: str) -> list[str]:
        """
        Uses the LLM to decompose a complex query into simpler, retrievable sub-queries.
        """
        decomposition_prompt = f"""
        You are an expert financial analyst. A user is asking a complex question about financial filings.
        Your task is to break down the following complex question into a list of concise, independent sub-queries.
        Each sub-query should be directly answerable by retrieving information from a financial 10-K filing.
        Focus on specific metrics, companies, and years.

        Example:
        Question: "Compare cloud revenue growth rates across all three companies from 2022 to 2023"
        Sub-queries:
        - Microsoft cloud revenue 2022
        - Microsoft cloud revenue 2023
        - Google cloud revenue 2022
        - Google cloud revenue 2023
        - NVIDIA data center revenue 2022
        - NVIDIA data center revenue 2023 (NVIDIA's equivalent to cloud revenue is typically Data Center)

        Example:
        Question: "Which company had the highest operating margin in 2023?"
        Sub-queries:
        - Microsoft operating margin 2023
        - Google operating margin 2023
        - NVIDIA operating margin 2023

        Example:
        Question: "How much did NVIDIA's data center revenue grow from 2022 to 2023?"
        Sub-queries:
        - NVIDIA data center revenue 2022
        - NVIDIA data center revenue 2023

        Your response should be ONLY a comma-separated list of sub-queries. Do not include any other text or formatting.

        Question: "{query}"
        Sub-queries:
        """
        response = self._call_llm(decomposition_prompt, temperature=0.0)
        # Attempt to parse comma-separated list, clean up extra text
        sub_queries = [q.strip() for q in response.split(',') if q.strip()]
        # Remove any leading/trailing hyphens if present from bullet point formatting
        sub_queries = [re.sub(r'^- ', '', q).strip() for q in sub_queries]
        return sub_queries

    def _synthesize_answer(self, original_query: str, retrieved_data: list[dict], sub_queries: list[str]) -> tuple[str, str, list[dict]]:
        """
        Uses the LLM to synthesize a coherent answer from retrieved data.
        Returns the answer string, reasoning string, and formatted sources.
        """
        if not retrieved_data:
            return "Could not find relevant information to answer the query.", "No relevant data retrieved.", []

        # Prepare context for the LLM
        context_str = ""
        sources_for_output = []
        for i, data in enumerate(retrieved_data):
            context_str += f"--- Source {i+1} ({data['company']} {data['year']} - Page {data['page']}) ---\n"
            context_str += data['text'] + "\n\n"
            sources_for_output.append({
                "company": data['company'],
                "year": data['year'],
                "excerpt": data['text'][:200] + "...", # Take first 200 chars as excerpt
                "page": data['page'],
                "source_doc": data['source_doc']
            })
        
        # Determine reasoning based on sub-queries
        reasoning_str = ""
        if len(sub_queries) > 1:
            reasoning_str = "Performed query decomposition, retrieved data for multiple sub-queries, and synthesized the results."
        else:
            reasoning_str = "Retrieved relevant information and synthesized the answer."

        synthesis_prompt = f"""
        You are a financial analysis AI. Based on the following information retrieved from financial filings,
        answer the original user query comprehensively and concisely.
        If the query involves calculations (e.g., growth rates, comparisons), perform them.
        Cite the company and year from which the information was sourced where appropriate.

        Original Query: "{original_query}"
        Sub-queries performed: {', '.join(sub_queries)}

        Retrieved Context (multiple sources may be provided):
        {context_str}

        Please provide a direct answer, followed by your reasoning.
        Answer:
        """
        
        raw_answer_response = self._call_llm(synthesis_prompt)
        
        # Try to parse answer and reasoning
        answer_parts = raw_answer_response.split("Reasoning:", 1)
        answer_text = answer_parts[0].strip()
        final_reasoning = reasoning_str + "\n" + (answer_parts[1].strip() if len(answer_parts) > 1 else "No specific reasoning provided by LLM, relying on agent's default reasoning.")

        return answer_text, final_reasoning, sources_for_output

    def answer_query(self, query: str) -> dict:
        """
        Main method to answer a user query, potentially involving decomposition and multi-step retrieval.
        """
        print(f"\n--- Answering Query: '{query}' ---")
        
        sub_queries = [query] # Assume simple query initially

        # Simple heuristic to determine if decomposition is likely needed
        # This can be made smarter with LLM-based classification in advanced versions
        if any(keyword in query.lower() for keyword in ["compare", "highest", "lowest", "growth from", "each company", "across all"]):
            print("Complex query detected. Attempting query decomposition...")
            decomposed_queries = self._decompose_query(query)
            if decomposed_queries:
                sub_queries = decomposed_queries
                print(f"Decomposed into sub-queries: {sub_queries}")
            else:
                print("Query decomposition failed. Proceeding with original query.")

        all_retrieved_chunks = []
        unique_source_docs = set()

        for sub_q in sub_queries:
            # Attempt to extract company and year from sub-query for more targeted retrieval
            company_match = re.search(r'\b(microsoft|msft|google|googl|nvidia|nvda)\b', sub_q, re.IGNORECASE)
            year_match = re.search(r'\b(2022|2023|2024)\b', sub_q)
            
            target_company = company_match.group(1).upper() if company_match else None
            target_year = year_match.group(1) if year_match else None

            print(f"Retrieving for sub-query: '{sub_q}' (Target: {target_company or 'Any'} {target_year or 'Any'})")
            # You might want to pass company/year to retrieve_chunks for filtered search if your vector store supports it
            # For now, we rely on top-k and LLM synthesis to filter
            chunks = self.rag_pipeline.retrieve_chunks(sub_q, k=5) # Retrieve top 5 chunks for each sub-query
            all_retrieved_chunks.extend(chunks)
            for chunk in chunks:
                unique_source_docs.add((chunk['company'], chunk['year'], chunk['source_doc'], chunk['page']))

        # Remove duplicate chunks if multiple sub-queries retrieved the same chunk
        # A simple way: convert to a tuple of relevant fields and back to list
        unique_retrieved_chunks_data = {}
        for chunk in all_retrieved_chunks:
            # Use chunk_id or a hash of text+source as key to ensure uniqueness
            key = (chunk.get('chunk_id') or chunk['text'] + chunk['source_doc'])
            if key not in unique_retrieved_chunks_data:
                unique_retrieved_chunks_data[key] = chunk
        
        final_retrieved_chunks = list(unique_retrieved_chunks_data.values())

        print(f"Total unique chunks retrieved for all sub-queries: {len(final_retrieved_chunks)}")

        final_answer, reasoning, sources = self._synthesize_answer(query, final_retrieved_chunks, sub_queries)

        output = {
            "query": query,
            "answer": final_answer,
            "reasoning": reasoning,
            "sub_queries": sub_queries,
            "sources": sources
        }
        
        return output