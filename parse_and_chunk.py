import os
import json
from bs4 import BeautifulSoup
import tiktoken
import re

# --- Configuration ---
# DATA_DIR is expected to be 'financial-rag-agent/data' relative to the script's location
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
CHUNKS_DIR = os.path.join(os.path.dirname(__file__), '../chunks')

ENCODING_MODEL = "cl100k_base" # Used by OpenAI, good general-purpose token counting
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# Ensure chunks directory exists
os.makedirs(CHUNKS_DIR, exist_ok=True)

# --- Helper Functions ---

def get_tokens(text):
    """Returns the tokens for a given text using tiktoken."""
    encoding = tiktoken.get_encoding(ENCODING_MODEL)
    return encoding.encode(text)

def clean_text(text):
    """Basic text cleaning to remove excessive whitespace and standardize."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def extract_text_from_html(file_path):
    """
    Parses an HTML 10-K filing to extract relevant text,
    focusing on key sections (Item 1A, Item 7, Item 8).
    Handles common variations in SEC filing HTML.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

    soup = BeautifulSoup(html_content, 'html.parser')

    extracted_sections = {}
    current_section_title = None
    current_section_content = []

    # UPDATED: More flexible regex patterns to match headings even if embedded in larger text blocks.
    # Removed ^ and $ anchors from patterns, as re.search will find them anywhere.
    # Added \s* to handle varying whitespace around parts, including zero spaces (e.g., 7.MANAGEMENT)
    item_patterns = {
        "Item 1A.": re.compile(r'ITEM\s*1A\.\s*RISK\s*FACTORS', re.IGNORECASE),
        "Item 7.": re.compile(r'ITEM\s*7\.\s*MANAGEMENT[’\']S\s*DISCUSSION\s*AND\s*ANALYSIS(?:.*FINANCIAL\s*CONDITION\s*AND\s*RESULTS\s*OF\s*OPERATIONS|)', re.IGNORECASE),
        # The (?:...|) at the end makes the long full title optional, allowing it to match shorter "MD&A" forms too if they exist and are titled similarly.
        "Item 8.": re.compile(r'ITEM\s*8\.\s*FINANCIAL\s*STATEMENTS\s*AND\s*SUPPLEMENTARY\s*DATA', re.IGNORECASE)
    }
    
    section_friendly_names = {
        "Item 1A.": "Risk Factors",
        "Item 7.": "Management's Discussion and Analysis",
        "Item 8.": "Financial Statements and Supplementary Data"
    }

    item_order = ["Item 1A.", "Item 7.", "Item 8."]
    
    capturing_content = False
    
    for element in soup.find_all(['div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'b', 'strong']):
        text_content = clean_text(element.get_text(separator=' ', strip=True))
        
        if not text_content:
            continue

        found_new_item = False
        for item_key, pattern in item_patterns.items():
            # Check if the element's text contains a section pattern AND is within a reasonable length for a title
            # This allows for some leading/trailing text like "Table of Contents"
            if pattern.search(text_content) and len(text_content.split()) < 150: # Increased threshold
                if capturing_content and current_section_title:
                    extracted_sections[current_section_title] = "\n".join(current_section_content).strip()
                    print(f"  Finished extracting: {current_section_title}")
                
                current_section_title = section_friendly_names[item_key]
                current_section_content = [text_content] # Keep the heading itself
                capturing_content = True
                found_new_item = True
                print(f"  Started extracting: {current_section_title}")
                break
        
        if not found_new_item and capturing_content:
            current_section_content.append(text_content)
        
        # Specific stop condition for Item 8, looking for the start of subsequent items
        # Added a check for length heuristic to ensure it's not matching a large block of text
        if current_section_title == section_friendly_names["Item 8."]:
            # This regex is also updated to be more flexible, matching common subsequent items/sections
            if re.search(r'item\s*(9\.|10\.|11\.|12\.|13\.|14\.|15\.|exhibits|signatures|part\s*(iii|iv)|signatu)', text_content, re.IGNORECASE) and len(text_content.split()) < 150:
                print(f"  Detected end marker '{text_content[:50]}...' after Item 8. Stopping extraction.")
                break

    if current_section_title and current_section_content:
        extracted_sections[current_section_title] = "\n".join(current_section_content).strip()
        print(f"  Finished extracting: {current_section_title} (end of file or explicit stop)")

    full_document_text = []
    for item_key in item_order:
        friendly_name = section_friendly_names[item_key]
        if friendly_name in extracted_sections and extracted_sections[friendly_name]:
            full_document_text.append(f"--- {friendly_name} ---\n{extracted_sections[friendly_name]}")
        else:
            print(f"  Warning: '{friendly_name}' section not found or empty for {file_path}")
    
    return "\n\n".join(full_document_text)


def split_text_into_chunks(text, chunk_size_tokens, chunk_overlap_tokens):
    """
    Splits a large text into smaller, overlapping chunks based on token count.
    Uses a recursive character splitting strategy to maintain semantic units.
    """
    chunks = []
    
    # Prioritized separators
    # Empty string "" separator is handled by converting to tokens as a last resort
    separators = ["\n\n", "\n", " "] 
    
    def _recursive_split(current_text, separator_index):
        tokens = get_tokens(current_text)

        # Base case 1: if text is small enough or we've run out of separators
        if len(tokens) <= chunk_size_tokens or separator_index >= len(separators):
            # If still too long after all character separators, do a hard split by tokens.
            if len(tokens) > chunk_size_tokens:
                chunked_parts = []
                start_token_idx = 0
                while start_token_idx < len(tokens):
                    end_token_idx = min(start_token_idx + chunk_size_tokens, len(tokens))
                    chunked_parts.append(tiktoken.get_encoding(ENCODING_MODEL).decode(tokens[start_token_idx:end_token_idx]))
                    start_token_idx += (chunk_size_tokens - chunk_overlap_tokens)
                return chunked_parts
            else:
                return [current_text]

        current_separator = separators[separator_index]
        
        parts = current_text.split(current_separator)
        
        temp_chunks = []
        current_chunk_parts = []
        current_chunk_token_count = 0

        for i, part in enumerate(parts):
            # Recursively split the part if it's too large for the current separator level
            sub_parts_from_recursion = _recursive_split(part, separator_index + 1)
            
            for sub_part in sub_parts_from_recursion:
                sub_part_tokens = get_tokens(sub_part)
                
                # Estimate tokens for separator if we were to join parts (only add if not the very first part)
                separator_tokens_len = len(get_tokens(current_separator)) if current_separator and current_chunk_parts else 0

                # Check if adding this sub_part would exceed the chunk size
                if current_chunk_token_count + len(sub_part_tokens) + separator_tokens_len > chunk_size_tokens:
                    # Current chunk is full, save it
                    if current_chunk_parts:
                        temp_chunks.append(current_separator.join(current_chunk_parts).strip())
                    
                    # Start new chunk with overlap
                    current_chunk_parts = []
                    current_chunk_token_count = 0

                    if chunk_overlap_tokens > 0 and len(temp_chunks) > 0:
                        last_chunk_text = temp_chunks[-1]
                        last_chunk_tokens = get_tokens(last_chunk_text)
                        
                        overlap_start_index = max(0, len(last_chunk_tokens) - chunk_overlap_tokens)
                        overlap_tokens = last_chunk_tokens[overlap_start_index:]
                        overlap_text = tiktoken.get_encoding(ENCODING_MODEL).decode(overlap_tokens)
                        
                        current_chunk_parts.append(overlap_text)
                        current_chunk_token_count += len(get_tokens(overlap_text))
                
                current_chunk_parts.append(sub_part)
                current_chunk_token_count += len(sub_part_tokens)
        
        # Add any remaining part of the current chunk
        if current_chunk_parts:
            temp_chunks.append(current_separator.join(current_chunk_parts).strip())
            
        return temp_chunks

    return _recursive_split(text, 0)


# --- Main Processing Logic ---

def process_filings():
    """
    Orchestrates extraction and chunking for all 10-K filings
    and saves processed chunk metadata to a JSON file.
    """
    all_processed_chunks_metadata = []

    print(f"--- Diagnosing Data Directory ---")
    print(f"Script is looking for data in: {os.path.abspath(DATA_DIR)}")
    
    # Get list of all .htm files in the data directory
    all_files_in_data_dir = []
    try:
        all_files_in_data_dir = os.listdir(DATA_DIR)
    except FileNotFoundError:
        print(f"\n❌ ERROR: Data directory not found at {os.path.abspath(DATA_DIR)}.")
        print("Please create this directory and place your 10-K HTML files inside it.")
        return

    html_files = [f for f in all_files_in_data_dir if f.endswith('.htm')]
    
    if not html_files:
        print(f"\n❌ ERROR: No .htm files found in {os.path.abspath(DATA_DIR)}.")
        print("Please ensure your 10-K HTML files are placed directly inside this directory.")
        print(f"Files found in {os.path.abspath(DATA_DIR)}: {all_files_in_data_dir}")
        return

    print(f"🔍 Found {len(html_files)} HTML files to process in {os.path.abspath(DATA_DIR)}")
    print(f"List of .htm files found: {html_files}")
    
    # Sort files for consistent processing order
    html_files.sort()

    for file_name in html_files:
        file_path = os.path.join(DATA_DIR, file_name)
        
        print(f"\n➡️ Processing: {file_name} | Size: {(os.path.getsize(file_path) / (1024 * 1024)):.2f} MB")
        
        extracted_text = extract_text_from_html(file_path)
        
        if not extracted_text:
            print(f"  No content extracted from {file_name}. Skipping chunking.")
            continue

        output_txt_path = os.path.join(CHUNKS_DIR, file_name.replace('.htm', '.txt'))
        try:
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            print(f"  ✅ Saved extracted text → {output_txt_path}")
        except Exception as e:
            print(f"  ❌ Error saving extracted text for {file_name}: {e}")
        
        chunks = split_text_into_chunks(extracted_text, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)
        
        # Extract company code and year from file name
        # Assumes format like GOOGL_2022.htm. Fallback for unexpected names.
        parts = file_name.replace('.htm', '').split('_')
        company_code = parts[0] if len(parts) > 0 else "UNKNOWN"
        year = parts[1] if len(parts) > 1 else "UNKNOWN"

        for i, chunk_text in enumerate(chunks):
            all_processed_chunks_metadata.append({
                "text": chunk_text,
                "company": company_code,
                "year": year,
                "source_doc": file_name,
                "chunk_id": f"{company_code}_{year}_chunk_{i:04d}",
                "page": i + 1
            })
        print(f"  📊 Generated {len(chunks)} chunks from {file_name}")

    output_json_path = os.path.join(CHUNKS_DIR, 'processed_chunks_metadata.json')
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_processed_chunks_metadata, f, indent=4)
        print(f"\n🎉 Successfully processed all files!")
        print(f"All processed chunk metadata saved to: {output_json_path}")
        print(f"Total chunks generated across all files: {len(all_processed_chunks_metadata)}")
    except Exception as e:
        print(f"\n❌ Error saving final JSON metadata: {e}")

if __name__ == "__main__":
    process_filings()