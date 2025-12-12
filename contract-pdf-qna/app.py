# Set the OpenAI API Keys, embedding model,
import os
import asyncio
from dotenv import load_dotenv
from flask import Flask, request, jsonify, make_response
from pymongo import MongoClient
from datetime import datetime
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Milvus
from langchain_community.memory.motorhead_memory import MotorheadMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.agents import Tool, initialize_agent, AgentType
from time import time
from bson.objectid import ObjectId
import uuid
from flask_cors import CORS
from oauth2client import client
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
# Add imports for transcript processing
import json
import re
from typing import List, Dict
from pathlib import Path

# GCP Storage imports using fsspec (unified filesystem interface)
try:
    import fsspec
    import gcsfs
    import certifi
    import os
    import ssl
    
    # Configure SSL certificates for macOS compatibility
    # CRITICAL: Set these BEFORE creating any filesystem objects
    # gcsfs uses aiohttp which requires SSL certificates via env vars
    cert_path = certifi.where()
    
    # Always set these (don't check if already set - ensure they're correct)
    os.environ['SSL_CERT_FILE'] = cert_path
    os.environ['REQUESTS_CA_BUNDLE'] = cert_path
    os.environ['AIOHTTP_CA_BUNDLE'] = cert_path
    
    # Create SSL context with certifi certificates
    ssl_context = ssl.create_default_context(cafile=cert_path)
    
    print(f"✓ SSL certificates configured: {cert_path}")
    
    GCP_STORAGE_AVAILABLE = True
except ImportError:
    print("Warning: fsspec or gcsfs not installed. GCP Storage features disabled.")
    print("Install with: pip install fsspec gcsfs")
    GCP_STORAGE_AVAILABLE = False
    fsspec = None
    gcsfs = None
    certifi = None
    ssl_context = None

# Safe import of monitoring_module - handle missing dependencies gracefully
try:
    from monitoring_module import q_monitor, tracer, llm_trace_to_jaeger
except ImportError as e:
    print(f"Warning: Could not import monitoring_module: {e}")
    print("Monitoring features will be disabled. The app will continue to run.")
    # Create dummy functions to prevent errors
    def q_monitor(*args, **kwargs):
        pass
    class DummyTracer:
        def start_span(self, *args, **kwargs):
            return DummySpan()
    class DummySpan:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __getattr__(self, name):
            return self
    tracer = DummyTracer()
    def llm_trace_to_jaeger(*args, **kwargs):
        pass

from token_module import token_calculator, CallbackHandler
import threading

# Using new LangChain memory API - InMemoryChatMessageHistory
# Note: This is only used to store previous Q&A for standalone prompt, not used in chains
memory1 = InMemoryChatMessageHistory()
handler = CallbackHandler()
load_dotenv()
app = Flask(__name__)

JWT_AUDIENCE = os.getenv("JWT_AUDIENCE")
JWKS_URL = os.getenv("JWKS_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_HOST = os.getenv("MILVUS_HOST")
MONGO_URI = (
    os.getenv("MONGO_URI")
)

CORS(app, resources={r"/*": {"origins": "*"}})

mongo_client = MongoClient(MONGO_URI, unicode_decode_error_handler='ignore')
db = mongo_client["FrontDoorDB"]

model_name = "text-embedding-ada-002"
embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

# Initialize GCP Storage using fsspec (with Application Default Credentials)
# Supports multiple authentication methods via environment variables
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "ahs-demo-transcripts")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "generative-ai-390411")
GCP_SERVICE_ACCOUNT_PATH = os.getenv("GCP_SERVICE_ACCOUNT_PATH", None)  # Optional: path to service account JSON
gcs_fs = None  # fsspec filesystem instance for GCS

# Cache for transcript metadata to avoid re-reading files
transcript_metadata_cache = {}

if GCP_STORAGE_AVAILABLE:
    try:
        # fsspec with gcsfs uses Application Default Credentials automatically
        # Method 1: Use GOOGLE_APPLICATION_CREDENTIALS environment variable (if set)
        # Method 2: Use Application Default Credentials (gcloud auth application-default login)
        # Method 3: Use explicit service account path from env variable
        
        if GCP_SERVICE_ACCOUNT_PATH and os.path.exists(GCP_SERVICE_ACCOUNT_PATH):
            # Use explicit service account file from environment variable
            gcs_fs = fsspec.filesystem('gcs', token=GCP_SERVICE_ACCOUNT_PATH, project=GCP_PROJECT_ID)
            print(f"✓ GCP Storage initialized using fsspec with service account from: {GCP_SERVICE_ACCOUNT_PATH}")
        else:
            # Use Application Default Credentials (ADC)
            # This will use GOOGLE_APPLICATION_CREDENTIALS if set, otherwise ADC
            try:
                # Get Application Default Credentials explicitly and pass to fsspec
                # gcsfs needs explicit credentials to work properly with ADC
                import certifi
                from google.auth import default as google_auth_default
                
                cert_path = certifi.where()
                
                # Get ADC credentials explicitly
                credentials, _ = google_auth_default()
                
                # Create filesystem with explicit ADC credentials
                gcs_fs = fsspec.filesystem('gcs', token=credentials, project=GCP_PROJECT_ID)
                print(f"✓ GCP Storage filesystem created using fsspec")
                print(f"  Bucket: {GCP_BUCKET_NAME}")
                print(f"  Project: {GCP_PROJECT_ID}")
                print(f"  SSL Certificates: {cert_path}")
                print(f"  Using Application Default Credentials")
                
                # Optional: Test connection (but don't fail if it fails - might be SSL/certificate issues)
                try:
                    bucket_path = f"gs://{GCP_BUCKET_NAME}/"
                    # Try to list files to verify connection
                    test_files = gcs_fs.ls(bucket_path, detail=False)
                    print(f"  ✓ Connection test successful - Found {len(test_files)} files in bucket")
                except Exception as test_error:
                    # Warning but don't fail - filesystem object is created, might work on actual use
                    error_msg = str(test_error)
                    if "SSL" in error_msg or "certificate" in error_msg.lower():
                        print(f"  ⚠ SSL certificate issue detected (common on macOS)")
                        print(f"    The filesystem is created and may work despite this warning")
                        print(f"    If you encounter SSL errors, try:")
                        print(f"      /Applications/Python\\ 3.13/Install\\ Certificates.command")
                    else:
                        print(f"  ⚠ Connection test failed: {test_error}")
                        print(f"    The filesystem is created and may work on actual use")
                        print(f"    If issues persist, try:")
                        print(f"      1. Run: gcloud auth application-default login")
                        print(f"      2. Set GOOGLE_APPLICATION_CREDENTIALS env variable")
                        print(f"      3. Set GCP_SERVICE_ACCOUNT_PATH env variable to service account JSON")
            except Exception as e:
                print(f"✗ GCP Storage filesystem creation failed: {e}")
                print(f"  Options:")
                print(f"    1. Run: gcloud auth application-default login")
                print(f"    2. Set GOOGLE_APPLICATION_CREDENTIALS env variable")
                print(f"    3. Set GCP_SERVICE_ACCOUNT_PATH env variable to service account JSON")
                print(f"    4. Install SSL certificates: /Applications/Python\\ 3.13/Install\\ Certificates.command")
                gcs_fs = None
    except Exception as e:
        print(f"✗ GCP Storage initialization failed: {e}")
        print("  GCP Storage features will be disabled.")
        print("  Make sure fsspec and gcsfs are installed: pip install fsspec gcsfs")
        gcs_fs = None


class AgentAction:
    def __init__(self, tool: str, tool_input: str, log: str = None):
        self.tool = tool
        self.tool_input = tool_input
        self.log = log


# Creating a prompt template
prompt_template = """

You are assisting a customer care executive. Your role is to review the contract’s contextual information given in the context below.

{context}

Answer the given user inquiry based on context above as truthfully as possible, providing in-depth explanations together with answers to the inquiries.
You may rephrase the final response to make it concise and sound more human-like, but do not go out of context and do not lose important details and meaning.

You'll be asked about repairs, coverage policy and service questions about home appliances, home fixtures, home care, repairs/replacement and cleaning, and also about the renewal, cancellation or refund policies in the contract, whether a certain service is covered under the contract and similar context.

The contract context given will have information about contractual details, terms and conditions, renewals, cancellation, refund and service request policies, the coverage limits, limitation and exclusion policies. You will need to use and infer from all the information available in context to analyze and then respond with the final answer.

If the question is about a square feet limit, make sure to compare the numerical values properly. 
For example,
Question: "will my 800 square feet guest house be covered?"
The answer to this question will be No, as the square feet limit for guest houses is 750 and 800 is greater than 750.

If the context you have received says that some breakdown is not covered due to "Misuse or Accidental Acts", then say it is not covered.
For example,
Question: "A rat chewed the wires of my ceiling fan. Is the repair covered?"
The answer to this question will be No, as contract clearly mentions that damage due to pests like rat will not be covered.

If the inquiry is unrelated to home repair and service, answer with "I don't have the information to answer this question.". For example, questions like "Tell me about space.", "Write a poem for me.", "Where can I buy a refrigerator?", "Hi! How are you?", etc. are out of context.

Question: {question} Why?
Answer: """

# PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["context", "question"]
# )
# chain_type_kwargs = {"prompt": PROMPT}

# conversational retrieval QA
PROMPT = PromptTemplate.from_template(prompt_template)

sys_msg = """

You are assisting an AHS customer care executive with home insurance related inquiries from AHS customers. 

You are given a tool named Knowledge Base, always use this tool to answer the questions. 

The inquiry asked might be subject to some exclusions and limitations which need to be checked for first before answering the rest of the inquiry. 
You have to break down these complex inquiries into multiple subqueries and then use the knowledge base tool multiple times to return the overall answer from the subqueries for the customer's inquiry. 
Make sure to answer to all the subqueries before you return the final answer.

Following are some examples of complex inquiries and how they can be broken down into sub queries.

Example 1:
“My dryer is not drying the clothes properly. It could be because of lint blockage. Will you come to fix it?”.
1. Is the dryer covered by the plan? If yes, Is repair for link blockage in the dryer covered by the plan?

Example 2: 
“I got my refrigerator fixed last week. But there is another issue with it now. What if that problem was caused by the last repair?”
1. Is the refrigerator covered in the plan?
2. If yes, Can another issue with the refrigerator be fixed in a week’s time from the last repair?

Example 3: 
“I purchased a plan from AHS just 5 days ago, and now I want to repair the microwave because it is creating too much noise. Can I get this repair done?” 
1. Is the microwave covered by the plan? If yes, is the repair for noise from the microwave covered?
2. Can I file a service request within 5 days of getting the plan?

Example 4:
“I use my personal washing machine for my daycare business too at my home. The drain pump doesn't seem to be working. Is it covered?”
1. Is washing machine and it's drain pump covered by the plan?
2. Is the breakdown of washing machine due to commercial use covered?

Example 5: 
“My water heater is leaking for some reason. I need to get it fixed. That water leak seeped into the air conditioning system, so that is not working too. So I need to get that fixed too.“
1. Is the water heater covered by the plan?
2. Is the air conditioning system covered by the plan?
3. Is secondary damage to the air conditioning system due to the water heater covered?

Some questions might be simpler and so might not need breaking down. Find response to those questions as it is. Following are examples of such inquiries.

Example 6:
“My microwave is not working. Is it covered?”

Example 7:
“My toilet seat is broken. Will you repair it?”

Do not answer any questions for which information is not provided by the knowledge base tool. 

If the inquiry is unrelated to home repair and service, answer with "I don't have the information to answer this question." For example, questions like "Tell me about space.", "Write a poem for me.", "Where can I buy a refrigerator?", "Hi! How are you?", etc. are out of context.

"""


def input_prompt(entered_query, qa, llm):
    # Retriever chain as Tool for agent
    knowledge_base_tool = Tool(
        name="Knowledge Base",
        func=qa.run,
        description=(
            "Useful for answering questions related to insurance coverage of home appliances, home fixtures, their repairs/replacement, service requests, about the renewal, cancellation or refund policies, whether a certain service is covered under the contract, permit limit, code violation limit, modification limit, limitations and exclusions."
        ),
    )

    tools = [knowledge_base_tool]

    current_time = time()

    MOTORHEAD_API_KEY = os.getenv("MOTORHEAD_API_KEY")
    MOTORHEAD_CLIENT_ID = os.getenv("MOTORHEAD_CLIENT_ID")
    MOTORHEAD_SESSION_ID = str(current_time)
    MOTORHEAD_MEMORY_KEY = "chat_history"

    # Long Term chat memory
    memory = MotorheadMemory(
        api_key=MOTORHEAD_API_KEY,
        client_id=MOTORHEAD_CLIENT_ID,
        session_id=MOTORHEAD_SESSION_ID,
        memory_key=MOTORHEAD_MEMORY_KEY,
        return_messages=True,
        input_key="input",
        output_key="output",
    )

    #
    async def memory_initialize():
        await memory.init()

    asyncio.run(memory_initialize())

    # Initializing agent
    agent = initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
        memory=memory,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    new_prompt = agent.agent.create_prompt(system_message=sys_msg, tools=tools)

    agent.agent.llm_chain.prompt = new_prompt

    response = agent({"input": entered_query},callbacks=[handler])
    return response


# Function to get relevant documents
def relevant_docs(entered_query, retriever):
    relevant_document = "Referred Documents: " + str(
        retriever.get_relevant_documents(entered_query)
    )
    return relevant_document


# ==================== TRANSCRIPT PROCESSING HELPER FUNCTIONS ====================

def extract_transcript_metadata(transcript_content: str, file_name: str) -> Dict:
    """
    Extract contractType, planType, and state from transcript file content
    Uses hybrid approach: JSON parsing -> Regex patterns -> LLM (if needed)
    """
    metadata = {
        "contractType": None,
        "planType": None,
        "state": None
    }
    
    try:
        # Method 1: Try parsing as JSON first (fastest)
        try:
            transcript_data = json.loads(transcript_content)
            if isinstance(transcript_data, dict):
                # Check common metadata field locations
                metadata_fields = transcript_data.get("metadata", {})
                if not metadata_fields:
                    metadata_fields = transcript_data
                
                # Extract contractType (case-insensitive keys)
                metadata["contractType"] = (
                    metadata_fields.get("contractType") or 
                    metadata_fields.get("contract_type") or
                    metadata_fields.get("contractType") or
                    metadata_fields.get("type")
                )
                
                # Extract planType
                metadata["planType"] = (
                    metadata_fields.get("planType") or
                    metadata_fields.get("plan_type") or
                    metadata_fields.get("selectedPlan") or
                    metadata_fields.get("selected_plan") or
                    metadata_fields.get("plan")
                )
                
                # Extract state
                metadata["state"] = (
                    metadata_fields.get("state") or
                    metadata_fields.get("selectedState") or
                    metadata_fields.get("selected_state") or
                    metadata_fields.get("stateCode")
                )
                
                # If all found, return early
                if all([metadata["contractType"], metadata["planType"], metadata["state"]]):
                    return metadata
        except json.JSONDecodeError:
            pass  # Not JSON, continue to text parsing
        
        # Method 2: Regex-based text parsing (fast, no LLM needed)
        content_upper = transcript_content.upper()
        
        # Extract contract type
        # Look for RE or Real Estate mentions
        if re.search(r'\bRE\b', content_upper) or "REAL ESTATE" in content_upper:
            metadata["contractType"] = "RE"
        elif re.search(r'\bDTC\b', content_upper) or "DIRECT TO CONSUMER" in content_upper or "DIRECT-TO-CONSUMER" in content_upper:
            metadata["contractType"] = "DTC"
        
        # Extract plan type using regex patterns
        plan_patterns = {
            "ShieldComplete": [
                r"SHIELD\s*COMPLETE",
                r"SHIELDCOMPLETE",
                r"COMPLETE\s*PLAN"
            ],
            "ShieldEssential": [
                r"SHIELD\s*ESSENTIAL",
                r"SHIELDESSENTIAL",
                r"ESSENTIAL\s*PLAN"
            ],
            "ShieldPlus": [
                r"SHIELD\s*PLUS",
                r"SHIELDPLUS",
                r"PLUS\s*PLAN"
            ],
            "ShieldSilver": [
                r"SHIELD\s*SILVER",
                r"SHIELDSILVER",
                r"SILVER\s*PLAN"
            ],
            "ShieldGold": [
                r"SHIELD\s*GOLD",
                r"SHIELDGOLD",
                r"GOLD\s*PLAN"
            ],
            "ShieldPlatinum": [
                r"SHIELD\s*PLATINUM",
                r"SHIELDPLATINUM",
                r"PLATINUM\s*PLAN"
            ]
        }
        
        for plan, patterns in plan_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_upper):
                    metadata["planType"] = plan
                    break
            if metadata["planType"]:
                break
        
        # Extract state codes (two-letter US state codes)
        # First try full state name matching (more accurate)
        state_names = {
            "CA": ["CALIFORNIA", "CALIF"],
            "NY": ["NEW YORK"],
            "TX": ["TEXAS"],
            "FL": ["FLORIDA"],
            "IL": ["ILLINOIS"],
            "PA": ["PENNSYLVANIA"],
            "OH": ["OHIO"],
            "GA": ["GEORGIA"],
            "NC": ["NORTH CAROLINA"],
            "MI": ["MICHIGAN"],
            "NJ": ["NEW JERSEY"],
            "VA": ["VIRGINIA"],
            "WA": ["WASHINGTON"],
            "AZ": ["ARIZONA"],
            "MA": ["MASSACHUSETTS"],
            "TN": ["TENNESSEE"],
            "IN": ["INDIANA"],
            "MO": ["MISSOURI"],
            "MD": ["MARYLAND"],
            "WI": ["WISCONSIN"]
        }
        
        for state_code, names in state_names.items():
            if any(name in content_upper for name in names):
                metadata["state"] = state_code
                break
        
        # If not found by name, try state code matching with context
        if not metadata["state"]:
            # Common state codes (prioritize common ones)
            common_state_codes = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI", 
                                 "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"]
            other_state_codes = ["AL", "AK", "AR", "CO", "CT", "DE", "HI", "ID", "IA", "KS",
                                "KY", "LA", "ME", "MN", "MS", "MT", "NE", "NV", "NH", "NM",
                                "ND", "OK", "OR", "RI", "SC", "SD", "UT", "VT", "WV", "WY", "DC"]
            
            all_state_codes = common_state_codes + other_state_codes
            
            for state_code in all_state_codes:
                # Pattern: state code with word boundaries, but check context
                pattern = r'\b' + state_code + r'\b'
                matches = list(re.finditer(pattern, content_upper))
                
                for match in matches:
                    # Check surrounding context (avoid false positives like "IN" in "calling")
                    start = max(0, match.start() - 15)
                    end = min(len(content_upper), match.end() + 15)
                    context = content_upper[start:end]
                    
                    # Positive indicators
                    positive_keywords = ["STATE", "PLAN", "CONTRACT", "COVERAGE", "POLICY", 
                                       "CALIFORNIA", "TEXAS", "FLORIDA", "NEW YORK", "ILLINOIS"]
                    # Negative indicators (words that might contain state codes)
                    negative_keywords = ["CALLING", "INFORMATION", "INSPECTION", "INSTALLATION"]
                    
                    # Check if context has positive keywords and not negative ones
                    has_positive = any(keyword in context for keyword in positive_keywords)
                    has_negative = any(keyword in context for keyword in negative_keywords)
                    
                    if has_positive or (not has_negative and len(context.strip()) < 30):
                        metadata["state"] = state_code
                        break
                
                if metadata["state"]:
                    break
    
    except Exception as e:
        print(f"Error extracting metadata from transcript {file_name}: {e}")
    
    return metadata


def list_transcript_files_gcp(limit: int = None, offset: int = 0, search: str = None) -> tuple:
    """
    List transcript files from GCP bucket using fsspec with pagination and search support
    
    IMPORTANT: This function searches through ALL files in the GCS bucket (all 147 files).
    It first lists all files from GCS, then applies search filter, then pagination.
    
    Returns tuple: (paginated_transcripts, total_count)
    - Lists ALL files from GCS bucket first (searches through complete file list)
    - If search is provided, filters ALL files by file name (case-insensitive partial match)
    - Then applies pagination to the filtered results
    - If limit is None, returns all transcripts (for backward compatibility)
    - If limit is set, only reads file contents for the paginated subset (much faster)
    """
    all_file_info = []  # Store basic file info without reading content
    try:
        # Ensure SSL certificates are set (in case function is called before app initialization)
        if GCP_STORAGE_AVAILABLE and certifi:
            cert_path = certifi.where()
            if 'SSL_CERT_FILE' not in os.environ:
                os.environ['SSL_CERT_FILE'] = cert_path
            if 'REQUESTS_CA_BUNDLE' not in os.environ:
                os.environ['REQUESTS_CA_BUNDLE'] = cert_path
            if 'AIOHTTP_CA_BUNDLE' not in os.environ:
                os.environ['AIOHTTP_CA_BUNDLE'] = cert_path
        
        if not gcs_fs:
            print(f"ERROR list_transcript_files_gcp: gcs_fs is None!")
            return ([], 0) if limit else []
        
        print(f"DEBUG list_transcript_files_gcp: Starting, gcs_fs type={type(gcs_fs)}, limit={limit}, offset={offset}, search={search}")
        bucket_path = f"gs://{GCP_BUCKET_NAME}/"
        
        # List files - try both root and transcripts/ prefix
        prefixes = ["transcripts/", ""]
        seen_files = set()  # Track files we've already processed
        
        for prefix in prefixes:
            try:
                # List files with details
                full_path = bucket_path + prefix if prefix else bucket_path
                print(f"DEBUG: Attempting to list files from: {full_path}")
                files = gcs_fs.ls(full_path, detail=True)
                print(f"DEBUG: Found {len(files)} items in {full_path}")
                
                for file_info in files:
                    # file_info can be a dict (detail=True) or string (detail=False)
                    # Handle both cases
                    if isinstance(file_info, str):
                        # If it's a string, it's just the path
                        file_path = file_info
                        file_size = 0
                        time_created = None
                    else:
                        # It's a dict with details
                        file_path = file_info.get('name', '')
                        file_size = file_info.get('size', 0)
                        time_created = file_info.get('timeCreated', None)
                    
                    # Skip directories (they end with /)
                    if file_path.endswith('/'):
                        continue
                    
                    # Only include JSON and TXT files
                    if not (file_path.endswith('.json') or file_path.endswith('.txt')):
                        continue
                    
                    # Extract filename
                    file_name = file_path.split("/")[-1]
                    
                    # Skip if already added
                    if file_name in seen_files:
                        continue
                    seen_files.add(file_name)
                    
                    # Convert timeCreated to ISO format if available
                    upload_date = None
                    if time_created:
                        if isinstance(time_created, str):
                            upload_date = time_created
                        else:
                            # If it's a datetime object, convert to ISO
                            upload_date = time_created.isoformat() if hasattr(time_created, 'isoformat') else str(time_created)
                    
                    # Store basic file info without reading content yet
                    all_file_info.append({
                        "fileName": file_name,
                        "filePath": file_path,
                        "uploadDate": upload_date,
                        "fileSize": file_size if file_size else 0,
                        "timeCreated": time_created
                    })
            except Exception as e:
                # Log the error for debugging
                error_msg = str(e)
                import traceback
                error_trace = traceback.format_exc()
                print(f"ERROR listing files with prefix '{prefix}': {error_msg}")
                print(f"Full traceback:\n{error_trace}")
                if "SSL" in error_msg or "certificate" in error_msg.lower():
                    print(f"  SSL certificate issue detected!")
                    print(f"  SSL_CERT_FILE={os.environ.get('SSL_CERT_FILE', 'NOT SET')}")
                    print(f"  REQUESTS_CA_BUNDLE={os.environ.get('REQUESTS_CA_BUNDLE', 'NOT SET')}")
                    print(f"  AIOHTTP_CA_BUNDLE={os.environ.get('AIOHTTP_CA_BUNDLE', 'NOT SET')}")
                # Continue to next prefix
                continue
        
        # Sort by upload date (newest first)
        all_file_info.sort(key=lambda x: x.get("uploadDate", "") or "", reverse=True)
        
        # Log total files found from GCS before any filtering
        total_files_from_gcs = len(all_file_info)
        print(f"DEBUG: Listed ALL {total_files_from_gcs} files from GCS bucket")
        
        # Store sample file names before filtering (for debugging)
        sample_file_names = []
        if total_files_from_gcs > 0:
            sample_file_names = [f.get("fileName", "") for f in all_file_info[:10]]
            print(f"DEBUG: Sample file names from GCS (first 10): {sample_file_names}")
        
        # Apply search filter if provided (case-insensitive partial match on file name)
        # This searches through ALL files from GCS (all 147 files)
        if search and search.strip():
            search_term = search.strip().lower()
            print(f"DEBUG: Searching through ALL {total_files_from_gcs} files from GCS for: '{search_term}'")
            print(f"DEBUG: Search will match any file name containing '{search_term}' (case-insensitive)")
            
            # Filter: search through all files from GCS
            matching_files = []
            checked_count = 0
            for file_info in all_file_info:
                file_name = file_info.get("fileName", "")
                file_name_lower = file_name.lower()
                
                # Debug: log first few comparisons
                if checked_count < 5:
                    matches = search_term in file_name_lower
                    print(f"DEBUG: Checking file '{file_name}' (lowercase: '{file_name_lower}') - contains '{search_term}'? {matches}")
                    checked_count += 1
                
                if search_term in file_name_lower:
                    matching_files.append(file_info)
            
            all_file_info = matching_files
            print(f"DEBUG: Search complete - Found {len(all_file_info)} matching files out of {total_files_from_gcs} total files in GCS")
            
            # If no matches, show sample file names to help debug
            if len(all_file_info) == 0 and total_files_from_gcs > 0:
                print(f"DEBUG: No matches found for '{search_term}'")
                print(f"DEBUG: Sample file names available in GCS: {sample_file_names}")
                print(f"DEBUG: Tip: Check if any file names contain '{search_term}'. Try calling without search parameter to see all file names.")
        else:
            print(f"DEBUG: No search term provided - returning all {total_files_from_gcs} files from GCS")
        
        total_count = len(all_file_info)
        print(f"DEBUG: Final count after search: {total_count} files, limit={limit}, offset={offset}")
        
        # If limit is None, return all (backward compatibility - read all files)
        if limit is None:
            print("DEBUG: limit is None - returning all transcripts")
            transcripts = []
            for file_info in all_file_info:
                # Extract contract metadata from file content
                transcript_metadata = {
                    "contractType": None,
                    "planType": None,
                    "state": None
                }
                
                # Check cache first
                cache_key = f"{file_info['filePath']}_{file_info['timeCreated']}"
                if cache_key in transcript_metadata_cache:
                    transcript_metadata = transcript_metadata_cache[cache_key]
                else:
                    # Read file content to extract metadata (limit size for performance)
                    try:
                        file_size = file_info.get('fileSize', 0)
                        if file_size and file_size < 50000:  # Only read files < 50KB for metadata extraction
                            with gcs_fs.open(file_info['filePath'], 'r') as f:
                                content = f.read()
                            transcript_metadata = extract_transcript_metadata(content, file_info['fileName'])
                            # Cache the result
                            transcript_metadata_cache[cache_key] = transcript_metadata
                        elif file_size:
                            print(f"Skipping metadata extraction for large file: {file_info['fileName']} ({file_size} bytes)")
                    except Exception as e:
                        print(f"Error reading transcript {file_info['fileName']} for metadata extraction: {e}")
                
                transcripts.append({
                    "fileName": file_info['fileName'],
                    "filePath": file_info['filePath'],
                    "uploadDate": file_info['uploadDate'],
                    "fileSize": file_info['fileSize'],
                    "metadata": {},
                    "contractType": transcript_metadata.get("contractType"),
                    "planType": transcript_metadata.get("planType"),
                    "state": transcript_metadata.get("state")
                })
            return transcripts
        
        # Apply pagination BEFORE reading file contents (optimization)
        print(f"DEBUG: Applying pagination - slicing all_file_info[{offset}:{offset + limit}]")
        paginated_file_info = all_file_info[offset:offset + limit]
        print(f"DEBUG: Paginated to {len(paginated_file_info)} files out of {total_count} total")
        
        # Now read file contents only for the paginated subset
        transcripts = []
        for file_info in paginated_file_info:
            # Extract contract metadata from file content
            transcript_metadata = {
                "contractType": None,
                "planType": None,
                "state": None
            }
            
            # Check cache first
            cache_key = f"{file_info['filePath']}_{file_info['timeCreated']}"
            if cache_key in transcript_metadata_cache:
                transcript_metadata = transcript_metadata_cache[cache_key]
            else:
                # Read file content to extract metadata (limit size for performance)
                try:
                    file_size = file_info.get('fileSize', 0)
                    if file_size and file_size < 50000:  # Only read files < 50KB for metadata extraction
                        with gcs_fs.open(file_info['filePath'], 'r') as f:
                            content = f.read()
                        transcript_metadata = extract_transcript_metadata(content, file_info['fileName'])
                        # Cache the result
                        transcript_metadata_cache[cache_key] = transcript_metadata
                    elif file_size:
                        print(f"Skipping metadata extraction for large file: {file_info['fileName']} ({file_size} bytes)")
                except Exception as e:
                    print(f"Error reading transcript {file_info['fileName']} for metadata extraction: {e}")
            
            transcripts.append({
                "fileName": file_info['fileName'],
                "filePath": file_info['filePath'],
                "uploadDate": file_info['uploadDate'],
                "fileSize": file_info['fileSize'],
                "metadata": {},
                "contractType": transcript_metadata.get("contractType"),
                "planType": transcript_metadata.get("planType"),
                "state": transcript_metadata.get("state")
            })
        
        print(f"DEBUG: Returning {len(transcripts)} transcripts with total_count={total_count}")
        return (transcripts, total_count)
        
    except Exception as e:
        print(f"Error listing transcript files from GCP: {e}")
        import traceback
        traceback.print_exc()
        return ([], 0) if limit else []


def read_transcript_file_gcp(file_name: str) -> tuple:
    """
    Read transcript file content from GCP bucket using fsspec
    Returns: (content, file_metadata_dict)
    """
    try:
        if not gcs_fs:
            raise Exception("GCP Storage not available")
        
        bucket_path = f"gs://{GCP_BUCKET_NAME}/"
        
        # Try different possible paths
        possible_paths = [
            f"{bucket_path}transcripts/{file_name}",
            f"{bucket_path}{file_name}",
        ]
        
        file_path = None
        for path in possible_paths:
            if gcs_fs.exists(path):
                file_path = path
                break
        
        if not file_path:
            raise FileNotFoundError(f"Transcript file not found: {file_name}")
        
        # Read file content using fsspec
        with gcs_fs.open(file_path, 'r') as f:
            content = f.read()
        
        # Get file metadata
        file_info = gcs_fs.info(file_path)
        time_created = file_info.get('timeCreated', None)
        
        # Convert timeCreated to ISO format if available
        upload_date = None
        if time_created:
            if isinstance(time_created, str):
                upload_date = time_created
            else:
                # If it's a datetime object, convert to ISO
                upload_date = time_created.isoformat() if hasattr(time_created, 'isoformat') else str(time_created)
        
        file_metadata = {
            "fileName": file_name,
            "fileSize": file_info.get('size', 0),
            "uploadDate": upload_date,
            "metadata": {}  # fsspec doesn't provide custom metadata
        }
        
        return content, file_metadata
    
    except Exception as e:
        print(f"Error reading transcript file from GCP: {e}")
        raise


def extract_atomic_questions(transcript_content: str, llm) -> List[Dict]:
    """
    Extract atomic questions from transcript content using LLM
    """
    extraction_prompt = ChatPromptTemplate.from_template(
        """
        You are an expert at analyzing customer service transcripts and extracting atomic questions.
        
        Analyze the following transcript and extract all atomic questions that customers asked.
        An atomic question is a single, specific question that can be answered independently.
        
        Transcript:
        {transcript}
        
        Return ONLY a JSON array of questions in this format:
        [
            {{
                "question": "Is the refrigerator covered?",
                "context": "Customer mentioned refrigerator issue",
                "questionType": "coverage"
            }},
            {{
                "question": "What is the repair limit?",
                "context": "Customer asked about repair costs",
                "questionType": "limit"
            }}
        ]
        
        Extract all questions. Return only valid JSON, no additional text.
        """
    )
    
    extraction_chain = LLMChain(llm=llm, prompt=extraction_prompt, verbose=True)
    
    try:
        result = extraction_chain.run({"transcript": transcript_content})
        # Clean the result - remove markdown code blocks if present
        result = re.sub(r'```json\n?', '', result)
        result = re.sub(r'```\n?', '', result)
        result = result.strip()
        
        questions = json.loads(result)
        # Add question IDs
        for idx, q in enumerate(questions):
            q["questionId"] = f"q{idx + 1}"
        return questions
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from LLM: {e}")
        print(f"LLM Response: {result[:500]}")
        return []
    except Exception as e:
        print(f"Error extracting questions: {e}")
        return []


def process_single_transcript_question(question: str, contract_type: str, selected_plan: str, 
                                     selected_state: str, gpt_model: str, vector_db: Milvus, 
                                     llm, llm2, retriever, handler) -> Dict:
    """
    Process a single question from transcript and return answer with chunks
    Reuses logic from /start endpoint but without conversation context
    """
    try:
        q_start_time = time()
        standalone_result = question  # No conversation context for transcript questions
        
        if gpt_model == "Search":
            prompt_template = """
            You are assisting a customer care executive. Your role is to review the contract's contextual information given in the context below.

            {context}

            Answer the given user inquiry based on context above as truthfully as possible, providing in-depth explanations together with answers to the inquiries.
            You may rephrase the final response to make it concise and sound more human-like, but do not go out of context and do not lose important details and meaning.

            Question: {question}
            Answer: """
            
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            chain_type_kwargs = {"prompt": PROMPT}
            qa = RetrievalQA.from_chain_type(
                llm=llm, 
                retriever=retriever, 
                verbose=True,
                chain_type_kwargs=chain_type_kwargs
            )
            
            answer = qa.run(standalone_result, callbacks=[handler])
            relevant_documents = relevant_docs(standalone_result, retriever=retriever)
            
        elif gpt_model == "Infer":
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, verbose=True)
            agent_response = input_prompt(standalone_result, qa, llm)
            answer = agent_response["output"]
            knowledge_base_thoughts = [
                item[0].tool_input for item in agent_response["intermediate_steps"] 
                if item[0].tool == 'Knowledge Base'
            ]
            relevant_documents = ""
            for action_input in knowledge_base_thoughts:
                relevant_documents += relevant_docs(action_input, retriever)
        else:
            return {
                "error": f"Invalid gpt_model: {gpt_model}",
                "answer": "",
                "relevantChunks": [],
                "confidence": 0.0,
                "latency": 0.0
            }
        
        q_latency = time() - q_start_time
        
        # Parse relevant documents to extract chunks
        chunks = []
        if relevant_documents:
            # Parse the relevant_documents string to extract chunks
            doc_parts = relevant_documents.split("Referred Documents: ")
            for part in doc_parts:
                if part.strip():
                    # Extract document content (limit length)
                    content = part.strip()[:500]
                    chunks.append({
                        "content": content,
                        "score": 0.85  # Default score, can be enhanced with actual similarity
                    })
        
        return {
            "answer": answer,
            "relevantChunks": chunks[:4],  # Limit to top 4 chunks
            "confidence": 0.90,  # Default confidence, can be calculated from LLM
            "latency": q_latency
        }
    except Exception as e:
        print(f"Error processing transcript question: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "answer": "Error processing question",
            "relevantChunks": [],
            "confidence": 0.0,
            "latency": 0.0
        }


# Feedback CRUD Operations


# Feedback CRUD Operations
# Create (Insert) operation
def insert_feedback(data, email_id):
    feedbacks_collection_user = f"feedbacks_{email_id}"
    feedbacks_collection = db[feedbacks_collection_user]
    result = feedbacks_collection.insert_one(data)
    print(f"Document inserted with ID: {result.inserted_id}")


# Read operation
def read_feedback(query, email_id):
    feedbacks_collection_user = f"feedbacks_{email_id}"
    feedbacks_collection = db[feedbacks_collection_user]
    search_query = {"entered_query": query}
    documents = (
        feedbacks_collection.find(search_query)
        if search_query
        else feedbacks_collection.find()
    )
    for document in documents:
        print(document)


# Update operation
def update_feedback(query, new_data, email_id):
    feedbacks_collection_user = f"feedbacks_{email_id}"
    feedbacks_collection = db[feedbacks_collection_user]
    search_query = {"entered_query": query}
    result = feedbacks_collection.update_one(search_query, {"$set": new_data})
    print(f"Modified {result.modified_count} document(s)")


# Delete operation
def delete_feedback(query, email_id):
    feedbacks_collection_user = f"feedbacks_{email_id}"
    feedbacks_collection = db[feedbacks_collection_user]
    search_query = {"entered_query": query}
    result = feedbacks_collection.delete_one(search_query)
    print(f"Deleted {result.deleted_count} document(s)")


# Questions and Answers CRUD Operations
# Create (Insert) operation
def insert_qna(data, email_id):
    qna_collection_today = f"chats_{email_id}"
    qna_collection = db[qna_collection_today]
    result = qna_collection.insert_one(data)
    print(f"Document inserted with ID: {result.inserted_id}")
    return result


# Anirudha Read operation
# def read_qna(query, email_id):
#     qna_collection_today = f"chats_{email_id}"
#     qna_collection = db[qna_collection_today]
#     search_query = {"entered_query": query}
#     documents = qna_collection.find(search_query) if search_query else qna_collection.find()
#     for document in documents:
#         print(document)

# def read_qna(email_id,mongo_qna_id=None):


def read_qna(email_id, conversation_id):
    qna_collection_user = f"chats_{email_id}"
    qna_collection = db[qna_collection_user]
    search_query = {"_id": ObjectId(conversation_id)}
    documents = qna_collection.find_one(search_query)
    return documents


# Update operation
def update_qna(query, new_data, email_id):
    qna_collection_today = f"chats_{email_id}"
    qna_collection = db[qna_collection_today]
    search_query = {"entered_query": query}
    result = qna_collection.update_one(search_query, {"$set": new_data})
    print(f"Modified {result.modified_count} document(s)")


# Delete operation
def delete_qna(query, email_id):
    qna_collection_today = f"chats_{email_id}"
    qna_collection = db[qna_collection_today]
    search_query = {"entered_query": query}
    result = qna_collection.delete_one(search_query)
    print(f"Deleted {result.deleted_count} document(s)")


def update_chat(new_data, conversation_id, email_id):
    qna_collection_user = f"chats_{email_id}"
    qna_collection = db[qna_collection_user]
    search_query = {"_id": ObjectId(conversation_id)}
    result = qna_collection.update_one(search_query, {"$push": {"chats": new_data}})
    print(f"Modified {result.modified_count} document(s)")


def token_process(authorization_header):
    parts = authorization_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        bearer_token = parts[1]
        try:
            token = client.verify_id_token(bearer_token, JWT_AUDIENCE)
            return (token), 200
        except Exception as e:
            if str(e).split(",")[0] == "Token used too late":
                return jsonify({"message": "Token has expired"}), 403
            else:
                return jsonify({"message": "Token is invalid"}), 403
    else:
        return jsonify({"message": "Token is missing"}), 401


@app.before_request
def before_request():
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }
    if request.method == "OPTIONS" or request.method == "options":
        return jsonify(headers), 200


@app.route("/feedback", methods=["POST"])
def feedback():
    with tracer.start_span('api/feedback'):
        authorization_header = request.headers.get("Authorization")

        # case 5: missing token
        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        user_feedback = request.get_json()

        # extract from query parameters
        conversation_id = request.args.get("conversation-id")
        chat_id = request.args.get("chat-id")

        # extract values from input
        reaction = user_feedback.get("reaction")
        response = user_feedback.get("response")
        user_email = token_data[0]["email"]

        query_time = datetime.utcnow()

        # output to be stored in mongodb collection
        feedback_json = {
            "query_time": query_time,
            "conversation_id": conversation_id,
            "chat_id": chat_id,
            "reaction": reaction,
            "response": response,
        }

        insert_feedback(feedback_json, user_email)
        return {}


@app.route("/start", methods=["POST"])
def start():
    try:
        with tracer.start_span('api/start') as parent0:
            with tracer.start_span('authorization', child_of=parent0):
                start_time = time()
                authorization_header = request.headers.get("Authorization")

                if authorization_header is None:
                    return jsonify({"message": "Token is missing"}), 401

                if authorization_header:
                    token_data = token_process(authorization_header)

                    if token_data[1] == 401 or token_data[1] == 403:
                        return (token_data[0].get_json()), token_data[1]
            
            with tracer.start_span('data-fetching', child_of=parent0):
                data = request.get_json()
                if not data:
                    return jsonify({"error": "Request body is missing or invalid"}), 400
                
                contract_type = data.get("contractType")
                selected_plan = data.get("selectedPlan")
                selected_state = data.get("selectedState")
                gpt_model = data.get("gptModel")
                entered_query = data.get("enteredQuery")
                
                # Validate required fields
                if not all([contract_type, selected_plan, selected_state, gpt_model, entered_query]):
                    return jsonify({"error": "Missing required fields: contractType, selectedPlan, selectedState, gptModel, enteredQuery"}), 400
                
                # user_email = "kartik.dabre@mindstix.com"
                user_email = token_data[0]["email"]
                conversation_id = request.args.get("conversation-id")

                collection_mapping = {
                    "RE": {
                        "ShieldEssential": f"{selected_state}_RE_ShieldEssential",
                        "ShieldPlus": f"{selected_state}_RE_ShieldPlus",
                        "default": f"{selected_state}_RE_ShieldComplete",
                    },
                    "DTC": {
                        "ShieldSilver": f"{selected_state}_DTC_ShieldSilver",
                        "ShieldGold": f"{selected_state}_DTC_ShieldGold",
                        "default": f"{selected_state}_DTC_ShieldPlatinum",
                    },
                }

                # Get the collection name based on contract_type and selected_plan
                selected_collection_name = collection_mapping.get(contract_type, {}).get(
                    selected_plan, collection_mapping.get(contract_type, {}).get("default")
                )
            with tracer.start_span('vector_db-initialization', child_of=parent0):
                # Selecting collection dynamically
                vector_db1: Milvus = Milvus(
                    embed,
                    collection_name=selected_collection_name,
                    connection_args={"host": MILVUS_HOST, "port": "19530"},
                )
            
            # Initialize variables to prevent undefined errors
            agent_resp = None
            relevant_documents = ""

            if gpt_model == "Search":
                with tracer.start_span('Search', child_of=parent0) as parent1:
                    with tracer.start_span('llm-retriever-initialization', child_of=parent1):
                        llm2 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
                        llm = ChatOpenAI(temperature=0.0, model="gpt-4o")
                        retriever = vector_db1.as_retriever(search_kwargs={"k": 4})
                    
                    with tracer.start_span('memory_update', child_of=parent1):
                        memory1.clear()
                        question1 = ""
                        answer1 = ""
                        if conversation_id is not None and conversation_id != "":
                            docs = read_qna(email_id=user_email, conversation_id=conversation_id)
                            if docs and "chats" in docs and len(docs["chats"]) > 0:
                                question1 = docs["chats"][-1]['entered_query']
                                answer1 = docs["chats"][-1]['response']
                                # Store in new memory API format (only if values exist)
                                if question1 and answer1:
                                    memory1.add_message(HumanMessage(content=question1))
                                    memory1.add_message(AIMessage(content=answer1))

                    with tracer.start_span('standalone-prompt-chain', child_of=parent1) as p:
                        standalone_prompt = ChatPromptTemplate.from_template(
                        """
                        Act as an expert in question rephrasing and create a standalone question in its own language by analyzing previous question, answer to the previous question and current question.
                        If the current question is not related to previous question and answer, then return the current question as standalone question. you have analyze if the component or appliance mentioned in the current question is related to the component or appliance mentioned in the previoius question and answer. based on that create the standalone question.
                        standalone question should always contain the appliance name, unless it is a service related question. questions related to modifications, code violation upgrades and permits are not bound to any appliance, so do not rephrase the question and do not relate this to any appliance related question.
                        previous question: """ + question1 + """
                        answer of previous question: """ + answer1 + """
                        current question: """ + entered_query + """
                        
                        examples:
                        1)  previous question:''
                            answer of previous question: ''
                            current question: is the Fridge covered?
                            standalone question: is the Fridge covered?
                        
                        2)  previous question: is the Air Conditioner system covered?
                            answer of previous question: yes, the air conditioner system is covered under the contract.
                            current question: is the compressor covered?
                            standalone question: is the compressor of the air conditioner system covered?
                        
                        In some of the cases, we will not need rephrasing, for example:
                        
                        3)  previous question: is the kitchen faucet covered?
                            answer of previous question: yes, the kitchen faucet is covered under the contract.
                            current question: is the garbage disposal covered?
                            standalone question: is the garbage disposal covered?
                        
                        4)  previous question: is the washer covered
                            answer of previous question: yes, washer is covered under the contract.
                            current question: there is damage to air conditioning unit because of leak but it is secondary, is it covered?
                            standalone question: there is damage to air conditioning unit because of leak but it is secondary, is it covered?
                        
                        """
                        )
                        start = int(time())
                        standalone_chain = LLMChain(llm=llm2, prompt=standalone_prompt, verbose=True)

                        standalone_result = standalone_chain.run({"input": entered_query},callbacks=[handler])
                        print(standalone_result)
                        res1, tok1 = handler.infi()
                        llm_trace_to_jaeger(res1, p.span_id, p.trace_id)
                        a = threading.Thread(target=token_calculator, args=(tok1,))
                        a.start()

                        print(f"time taken for standalone = {time() - start}")

                    with tracer.start_span('q_monitor', child_of=parent1) as parentq:
                        t = threading.Thread(target=q_monitor, args=(parentq,entered_query,))
                        t.start()
                        # q_monitor(parentq,entered_query)

                    with tracer.start_span('llm-RetrievalQA-chain', child_of=parent1) as q:
                        # Creating a prompt template
                        prompt_template = """

                        You are assisting a customer care executive. Your role is to review the contract’s contextual information given in the context below.

                        {context}

                        Answer the given user inquiry based on context above as truthfully as possible, providing in-depth explanations together with answers to the inquiries.
                        You may rephrase the final response to make it concise and sound more human-like, but do not go out of context and do not lose important details and meaning.

                        You'll be asked about repairs, coverage policy and service questions about home appliances, home fixtures, home care, repairs/replacement and cleaning, and also about the renewal, cancellation or refund policies in the contract, whether a certain service is covered under the contract and similar context.

                        The contract context given will have information about contractual details, terms and conditions, renewals, cancellation, refund and service request policies, the coverage limits, limitation and exclusion policies. You will need to use and infer from all the information available in context to analyze and then respond with the final answer.

                        If the question is about a square feet limit, make sure to compare the numerical values properly. 
                        For example,
                        Question: "will my 800 square feet guest house be covered?"
                        The answer to this question will be No, as the square feet limit for guest houses is 750 and 800 is greater than 750.

                        If the inquiry is unrelated to home repair and service, answer with "I don't have the information to answer this question.". For example, questions like "Tell me about space.", "Write a poem for me.", "Where can I buy a refrigerator?", "Hi! How are you?", etc. are out of context.

                        Always include the appliance name in the answer and provide in depth information.

                        Make the answer as short as possible with in depth information.

                        Question: """ + standalone_result + """
                        Answer: """

                        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context"])
                        chain_type_kwargs = {"prompt": PROMPT}
                        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, verbose=True,
                                                        chain_type_kwargs=chain_type_kwargs)

                        agent_resp = qa.run(standalone_result,callbacks=[handler])
                        res2, tok2 = handler.infi()
                        llm_trace_to_jaeger(res2, q.span_id, q.trace_id)
                        b = threading.Thread(target=token_calculator, args=(tok2,))
                        b.start()
                    
                    with tracer.start_span('relevant_documents', child_of=parent1):
                        relevant_documents = relevant_docs(entered_query, retriever=retriever)

            elif gpt_model == "Infer":
                with tracer.start_span('Infer', child_of=parent0) as parent1:
                    with tracer.start_span('llm-retriever-initialization', child_of=parent1):
                        llm3 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
                        llm = ChatOpenAI(temperature=0.0, model='gpt-4o')
                        llm2 = ChatOpenAI(temperature=0.0, model='gpt-4o')
                        retriever = vector_db1.as_retriever(search_kwargs={"k": 4})
                        
                    with tracer.start_span('memory_update', child_of=parent1):
                        memory1.clear()
                        question1 = ""
                        answer1 = ""
                        if conversation_id is not None and conversation_id != "":
                            docs = read_qna(email_id=user_email, conversation_id=conversation_id)
                            if docs and "chats" in docs and len(docs["chats"]) > 0:
                                question1 = docs["chats"][-1]['entered_query']
                                answer1 = docs["chats"][-1]['response']
                                # Store in new memory API format (only if values exist)
                                if question1 and answer1:
                                    memory1.add_message(HumanMessage(content=question1))
                                    memory1.add_message(AIMessage(content=answer1))

                    with tracer.start_span('standalone-prompt-chain', child_of=parent1) as p:
                        standalone_prompt = ChatPromptTemplate.from_template(
                            """
                            Identify if the current question is related to previous question and answer and Create a standalone question in its own language by analyzing previous question, answer to the previous question and current question.
                            If the current question is not related to previous question and answer, then return the current question as standalone question. If the previous question and answer is not available, then return current question as standalone question. you have analyze if the component or appliance mentioned in the current question is related to the component or appliance mentioned in the previoius question and answer. based on that create the standalone question.
                            standalone question should always contain the appliance name, unless it is a service related question. questions related to modifications, code violation upgrades and permits are not bound to any appliance, so do not rephrase the question and do not relate this to any appliance related question.
                            Always only return the output.
                            previous question: """ + question1 + """
                            answer of previous question: """ + answer1 + """
                            current question: """ + entered_query + """
                    
                            examples:
                            If there is no previous question or previous answer, then do not create the standalone question at all.
                            1)  previous question:''
                                answer of previous question: ''
                                current question: is the Fridge covered?
                                standalone question: is the Fridge covered?
                                
                            If there is secondary damage to the appliance being talked, create a standalone question in following way.
                            2)  previous question: my oven caught fire, is the oven covered?
                                answer of the previous question:Yes, your oven is covered by the plan. The plan covers all parts and components of installed ranges, ovens, and cooktops, including burner, display, self-clean, igniter, element, control panel and board, oven heating element, and temperature sensor. However, there are certain limitations and exclusions that apply, so it's important to review the specific terms and conditions of your plan for more details.
                                current question: this fire has damaged the exhaust fan located above it, is it covered?
                                standalone question: is the secondary damaged caused by the fire in the oven to the exhaust fan covered? 
                    
                            In some of the cases, current question wont need rephrasing, for example:
                            
                            3)  previous question: is the washer covered
                                answer of previous question: yes, washer is covered under the contract.
                                current question: there is damage to air conditioning unit because of leak but it is secondary damage, is it covered?
                                standalone question: there is damage to air conditioning unit because of leak but it is secondary damage, is it covered?
                            
                                    """
                        )
                        start = int(time())
                        standalone_chain = LLMChain(llm=llm3, prompt=standalone_prompt, verbose=True)

                        standalone_result = standalone_chain.run({"input": entered_query})
                        print(standalone_result)
                        res1, tok1 = handler.infi()
                        llm_trace_to_jaeger(res1, p.span_id, p.trace_id)
                        a = threading.Thread(target=token_calculator, args=(tok1,))
                        a.start()

                    with tracer.start_span('q_monitor', child_of=parent1) as parentq:
                        t = threading.Thread(target=q_monitor, args=(parentq,entered_query,))
                        t.start()

                    with tracer.start_span('llm-RetrievalQA-chain', child_of=parent1) as q:
                        qa = RetrievalQA.from_chain_type(llm=llm2, retriever=retriever, verbose=True)
                        agent_response = input_prompt(standalone_result, qa, llm)
                        agent_resp = agent_response["output"]
                        res2, tok2 = handler.infi()
                        llm_trace_to_jaeger(res2, q.span_id, q.trace_id)
                        b = threading.Thread(target=token_calculator, args=(tok2,))
                        b.start()
                    
                    with tracer.start_span('relevant_documents', child_of=parent1):
                        knowledge_base_thoughts = [item[0].tool_input for item in agent_response["intermediate_steps"] if
                                            item[0].tool == 'Knowledge Base']
                        relevant_documents = ""
                        for action_input in knowledge_base_thoughts:
                            relevant_documents += relevant_docs(action_input, retriever)
            else:
                return jsonify({"error": f"Invalid gpt_model: {gpt_model}. Must be 'Search' or 'Infer'"}), 400

            with tracer.start_span('output-formating', child_of=parent0):
                # Validate that we have a response
                if agent_resp is None:
                    return jsonify({"error": "Invalid gpt_model. Must be 'Search' or 'Infer'"}), 400
                
                ai_response = agent_resp

                word_count = len(relevant_documents.split())
                latency = time() - start_time

                query_time = datetime.now()

                chat = {
                    "chat_id": str(uuid.uuid4()),
                    "entered_query": entered_query,
                    "response": ai_response,
                    "relevant_docs": relevant_documents,
                    "gpt_model": gpt_model,
                    "chat_timestamp": query_time,
                    "latency": latency,
                    "word_count": word_count
                }

                if conversation_id is None or conversation_id == "":
                    qna_json = {
                        "conversation_name": entered_query,
                        "contract_type": contract_type,
                        "selected_plan": selected_plan,
                        "selected_state": selected_state,
                        "query_time": query_time,
                        "chats": [chat],
                    }

                    conversation_id = insert_qna(email_id=user_email, data=qna_json)
                    conversation_id = conversation_id.inserted_id

                else:
                    add_chat = update_chat(
                        new_data=chat, conversation_id=conversation_id, email_id=user_email
                    )

                output_json = {"aiResponse": ai_response, "conversationId": str(conversation_id), "chatId":chat.get("chat_id")}

        return make_response(jsonify(output_json), 200)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in /start endpoint: {str(e)}")
        print(f"Traceback: {error_trace}")
        return jsonify({"error": "An error occurred while processing your request", "details": str(e)}), 500


@app.route("/history", methods=["GET"])
def chat_history():
    with tracer.start_span('api/history'):
        authorization_header = request.headers.get("Authorization")

        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        conversation_id = request.args.get("conversation-id")
        user_email = token_data[0]["email"]

        docs = read_qna(email_id=user_email, conversation_id=conversation_id)

        feedback_collection_user = f"feedbacks_{user_email}"
        feedback_collection = db[feedback_collection_user]
        feedback_reaction = feedback_collection.find(
            {"conversation_id": str(conversation_id)}
        )
        feedback_dict = {}

        for doc in feedback_reaction:
            chat_id = str(doc["chat_id"])
            feedback_dict[chat_id] = doc["reaction"]

        chats = docs["chats"]
        for chat in chats:
            chat_id = chat.get("chat_id")
            if chat_id in feedback_dict:
                chat["reaction"] = feedback_dict[chat_id]

        output_json = {
            "conversationName": docs["conversation_name"],
            "contractType": docs["contract_type"],
            "selectedPlan": docs["selected_plan"],
            "selectedState": docs["selected_state"],
            "chats": chats,
            "gptModel": chats[0]["gpt_model"],
        }
        if docs:
            return make_response(jsonify(output_json), 200)
        else:
            return make_response(
                jsonify({"message": "No data found in the specified conversation"}), 404
            )


@app.route("/sidebar", methods=["GET"])
def sidebar_history():
    with tracer.start_span('api/sidebar'):
        authorization_header = request.headers.get("Authorization")

        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        user_email = token_data[0]["email"]

        qna_collection_user = f"chats_{user_email}"
        qna_collection = db[qna_collection_user]

        # projection means setting the key name to 1, i.e we want all ids and names from given collection
        result = qna_collection.find({}, {"_id": 1, "conversation_name": 1})

        output_json = [
                        {
                            "conversationId": str(doc["_id"]),
                            "conversationName": doc["conversation_name"],
                        }
                        for doc in result
                    ][::-1]

        if output_json:
            return make_response(jsonify(output_json), 200)
        else:
            return make_response(jsonify([]), 200)


@app.route("/delete", methods=["DELETE"])
def delete_conversation():
    with tracer.start_span('api/delete'):
        authorization_header = request.headers.get("Authorization")

        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        user_email = token_data[0]["email"]

        qna_collection_user = f"chats_{user_email}"
        qna_collection = db[qna_collection_user]
        conversation_id = request.args.get("conversation-id")

        qna_collection.delete_one({"_id": ObjectId(conversation_id)})
        return {}


@app.route("/edit-conversation-name", methods=["PATCH"])
def edit_name():
    with tracer.start_span('api/edit-conversation-name'):
        authorization_header = request.headers.get("Authorization")

        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        user_email = token_data[0]["email"]

        data = request.get_json()
        new_name = data.get("newName")

        conversation_id = request.args.get("conversation-id")

        try:
            qna_collection_user = f"chats_{user_email}"
            qna_collection = db[qna_collection_user]

            qna_collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {"$set": {"conversation_name": new_name}},
            )
            return jsonify({"message": "Conversation name updated successfully"})

        except Exception as e:
            return jsonify({"error": str(e)})


@app.route("/referred-clauses", methods=["GET"])
def referred_clauses():
    with tracer.start_span('api/referred-clauses'):
        authorization_header = request.headers.get("Authorization")

        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        user_email = token_data[0]["email"]
        conversation_id = request.args.get("conversation-id")
        chat_id = request.args.get("chat-id")

        try:
            docs = read_qna(email_id=user_email, conversation_id=conversation_id)

            chat_ans = docs["chats"]
            for chat_obj in chat_ans:
                if chat_obj.get("chat_id") == chat_id:
                    question = chat_obj.get("entered_query")
                    answer = chat_obj.get("response")

            referred_clauses_json = {
                "contractType": docs["contract_type"],
                "selectedState": docs["selected_state"],
                "selectedPlan": docs["selected_plan"],
                "question": question,
                "answer": answer,
                "referredClauses": chat_obj["relevant_docs"],
                "gpt_model": chat_obj.get("gpt_model"),
                "latency": chat_obj.get("latency", None),
                "word_count": chat_obj.get("word_count", None)
            }

            return referred_clauses_json

        except Exception as e:
            return jsonify({"error": str(e)}), 404


# ==================== TRANSCRIPT ENDPOINTS ====================

@app.route("/transcripts", methods=["GET"])
def list_transcripts():
    """List transcript files from GCP bucket with pagination and search (default: 10 records per page)
    
    IMPORTANT: When searching, this endpoint searches through ALL files in the GCS bucket (all 147 files).
    It lists all files from GCS first, then filters by search term, then applies pagination.
    
    Query Parameters:
    - limit (int, default: 10): Number of records per page
    - offset (int, default: 0): Number of records to skip
    - search (str, optional): Search term to filter transcripts by file name (case-insensitive partial match)
                             Searches through ALL files from GCS bucket
    - q (str, optional): Alias for 'search' parameter
    """
    try:
        with tracer.start_span('api/transcripts'):
            authorization_header = request.headers.get("Authorization")
            
            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401
            
            if authorization_header:
                token_data = token_process(authorization_header)
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]
            
            if not gcs_fs:
                return jsonify({"error": "GCP Storage not configured or unavailable"}), 500
            
            # Get query parameters - default limit is 10 for better performance
            limit_param = request.args.get("limit", "10")
            offset_param = request.args.get("offset", "0")
            search_param = request.args.get("search") or request.args.get("q")  # Support both 'search' and 'q' parameters
            print(f"DEBUG API: Raw params - limit_param='{limit_param}', offset_param='{offset_param}', search_param='{search_param}'")
            
            try:
                limit = int(limit_param) if limit_param else 10
            except (ValueError, TypeError):
                limit = 10
                print(f"DEBUG API: Invalid limit param, using default: 10")
            
            try:
                offset = int(offset_param) if offset_param else 0
            except (ValueError, TypeError):
                offset = 0
                print(f"DEBUG API: Invalid offset param, using default: 0")
            
            # Validate parameters
            if limit < 1:
                print(f"DEBUG API: limit < 1, setting to 10")
                limit = 10
            if offset < 0:
                print(f"DEBUG API: offset < 0, setting to 0")
                offset = 0
            
            # List transcript files from GCP with pagination and search (only reads content for paginated subset)
            print(f"DEBUG API: Calling list_transcript_files_gcp(limit={limit}, offset={offset}, search={search_param}), gcs_fs={gcs_fs is not None}")
            paginated_transcripts, total_count = list_transcript_files_gcp(limit=limit, offset=offset, search=search_param)
            print(f"DEBUG API: Found {len(paginated_transcripts)} transcripts (showing {offset} to {offset + len(paginated_transcripts)} of {total_count} total)")
            
            return jsonify({
                "transcripts": paginated_transcripts,
                "totalCount": total_count,
                "limit": limit,
                "offset": offset,
                "hasMore": (offset + limit) < total_count,
                "search": search_param if search_param else None
            }), 200
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in /transcripts endpoint: {str(e)}")
        print(f"Traceback: {error_trace}")
        return jsonify({"error": "An error occurred while fetching transcripts", "details": str(e)}), 500


@app.route("/transcripts/process", methods=["POST"])
def process_transcript():
    """Process transcript: fetch from GCP, extract questions, and get answers"""
    try:
        with tracer.start_span('api/transcripts/process') as parent0:
            start_time = time()
            
            # Authorization
            with tracer.start_span('authorization', child_of=parent0):
                authorization_header = request.headers.get("Authorization")
                
                if authorization_header is None:
                    return jsonify({"message": "Token is missing"}), 401
                
                if authorization_header:
                    token_data = token_process(authorization_header)
                    if token_data[1] == 401 or token_data[1] == 403:
                        return (token_data[0].get_json()), token_data[1]
            
            # Get request data
            with tracer.start_span('data-fetching', child_of=parent0):
                data = request.get_json()
                if not data:
                    return jsonify({"error": "Request body is missing or invalid"}), 400
                
                transcript_file_name = data.get("transcriptFileName")
                contract_type = data.get("contractType")
                selected_plan = data.get("selectedPlan")
                selected_state = data.get("selectedState")
                gpt_model = data.get("gptModel", "Search")
                extract_questions = data.get("extractQuestions", True)
                provided_questions = data.get("questions", [])
                
                # Validate required fields
                if not transcript_file_name:
                    return jsonify({"error": "transcriptFileName is required"}), 400
                
                if extract_questions and not all([contract_type, selected_plan, selected_state]):
                    return jsonify({
                        "error": "contractType, selectedPlan, selectedState are required when extractQuestions=true"
                    }), 400
            
            user_email = token_data[0]["email"]
            
            # Read transcript from GCP bucket
            with tracer.start_span('download-transcript', child_of=parent0):
                if not gcs_fs:
                    return jsonify({"error": "GCP Storage not configured or unavailable"}), 500
                
                try:
                    transcript_content, file_metadata = read_transcript_file_gcp(transcript_file_name)
                    
                    # Parse transcript (assuming JSON format)
                    try:
                        transcript_data = json.loads(transcript_content)
                        if isinstance(transcript_data, dict):
                            transcript_text = transcript_data.get("text", 
                                transcript_data.get("transcript", 
                                transcript_data.get("content", str(transcript_data))))
                        else:
                            transcript_text = transcript_content
                    except json.JSONDecodeError:
                        # If not JSON, treat as plain text
                        transcript_text = transcript_content
                    
                except FileNotFoundError as e:
                    return jsonify({"error": f"Transcript file not found: {transcript_file_name}"}), 404
                except Exception as e:
                    return jsonify({"error": f"Error reading transcript file: {str(e)}"}), 500
            
            # Extract questions
            questions = []
            if extract_questions:
                with tracer.start_span('extract-questions', child_of=parent0):
                    llm_extract = ChatOpenAI(temperature=0.0, model="gpt-4o")
                    questions = extract_atomic_questions(transcript_text, llm_extract)
                    
                    if not questions:
                        return jsonify({
                            "error": "No questions could be extracted from transcript",
                            "transcriptId": transcript_file_name.replace(".json", "").replace(".txt", "")
                        }), 400
            else:
                questions = provided_questions
                if not questions:
                    return jsonify({"error": "No questions provided"}), 400
            
            # Initialize vector DB and LLM
            with tracer.start_span('vector_db-initialization', child_of=parent0):
                collection_mapping = {
                    "RE": {
                        "ShieldEssential": f"{selected_state}_RE_ShieldEssential",
                        "ShieldPlus": f"{selected_state}_RE_ShieldPlus",
                        "default": f"{selected_state}_RE_ShieldComplete",
                    },
                    "DTC": {
                        "ShieldSilver": f"{selected_state}_DTC_ShieldSilver",
                        "ShieldGold": f"{selected_state}_DTC_ShieldGold",
                        "default": f"{selected_state}_DTC_ShieldPlatinum",
                    },
                }
                
                selected_collection_name = collection_mapping.get(contract_type, {}).get(
                    selected_plan, collection_mapping.get(contract_type, {}).get("default")
                )
                
                vector_db1 = Milvus(
                    embed,
                    collection_name=selected_collection_name,
                    connection_args={"host": MILVUS_HOST, "port": "19530"},
                )
                
                retriever = vector_db1.as_retriever(search_kwargs={"k": 4})
                
                if gpt_model == "Search":
                    llm2 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
                    llm = ChatOpenAI(temperature=0.0, model="gpt-4o")
                elif gpt_model == "Infer":
                    llm3 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
                    llm = ChatOpenAI(temperature=0.0, model='gpt-4o')
                    llm2 = ChatOpenAI(temperature=0.0, model='gpt-4o')
                else:
                    return jsonify({"error": f"Invalid gpt_model: {gpt_model}. Must be 'Search' or 'Infer'"}), 400
            
            # Process each question
            results = []
            total_latency = 0
            confidences = []
            
            with tracer.start_span('process-questions', child_of=parent0):
                for question_obj in questions:
                    question_text = question_obj.get("question", "")
                    question_id = question_obj.get("questionId", f"q{len(results) + 1}")
                    
                    result = process_single_transcript_question(
                        question_text, contract_type, selected_plan, 
                        selected_state, gpt_model, vector_db1, llm, llm2, 
                        retriever, handler
                    )
                    
                    result["questionId"] = question_id
                    result["question"] = question_text
                    result["context"] = question_obj.get("context", "")
                    result["questionType"] = question_obj.get("questionType", "general")
                    
                    if "error" not in result:
                        confidences.append(result.get("confidence", 0.0))
                        total_latency += result.get("latency", 0.0)
                    
                    results.append(result)
            
            # Calculate summary
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            response = {
                "transcriptId": transcript_file_name.replace(".json", "").replace(".txt", ""),
                "transcriptMetadata": {
                    "fileName": file_metadata["fileName"],
                    "uploadDate": file_metadata["uploadDate"],
                    "fileSize": file_metadata["fileSize"]
                },
                "questions": results,
                "summary": {
                    "totalQuestions": len(questions),
                    "processedQuestions": len([r for r in results if "error" not in r]),
                    "averageConfidence": round(avg_confidence, 2),
                    "totalLatency": round(total_latency, 2)
                }
            }
            
            return jsonify(response), 200
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in /transcripts/process endpoint: {str(e)}")
        print(f"Traceback: {error_trace}")
        return jsonify({
            "error": "An error occurred while processing transcript", 
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
