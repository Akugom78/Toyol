import os
import json
import datetime
import streamlit as st
from openai import OpenAI
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. CONFIGURATION
# ==========================================
API_KEY = st.secrets.get("DASHSCOPE_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

LLM_MODEL = "qwen-plus"          
EMBED_MODEL = "BAAI/bge-small-en-v1.5" 
CHUNKS_FILE = "./data/chunks.json"
DB_DIR = "./chroma_db"
MAX_HISTORY = 10  # Keep last 10 user-assistant exchanges (20 messages total)

if not API_KEY:
    st.error("❌ Missing DASHSCOPE_API_KEY. Please add it to .streamlit/secrets.toml or set it as an environment variable.")
    st.stop()

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ==========================================
# 2. DYNAMIC GREETING HELPER
# ==========================================
def get_dynamic_greeting():
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        time_greeting = "Good morning"
    elif 12 <= current_hour < 18:
        time_greeting = "Good afternoon"
    else:
        time_greeting = "Good evening"
        
    return f"{time_greeting}! I am your Senior ATC Training Officer and Compliance Analyst assistant.\n\nI can help you with:\n• **Q&A / Procedural Lookup**\n• **Document Drafting** (Manuals, UOIs, Memos, SOPs)\n• **Regulation Research**\n• **Document Discrepancy Analysis**\n\nHow can I help you today?"

# ==========================================
# 3. THE FULL SYSTEM PROMPT (Always remembered)
# ==========================================
SYSTEM_PROMPT = """You are a Senior Air Traffic Controller with extensive operational experience.
You also serve as an ATC Training Officer, Regulatory Document Author, and Compliance Analyst.

Your responsibilities include:
  • Training and evaluating ATC personnel
  • Drafting Manuals, Units of Instruction (UOIs), Memos, SOPs, and Notices in standardized ICAO/State format
  • Researching current and historical ICAO, State, and Unit regulations
  • Comparing documents to identify discrepancies, conflicts, and amendments

You are operating inside a Retrieval-Augmented Generation (RAG) system.
When document chunks are retrieved, they will be provided to you under the heading [RETRIEVED CONTEXT].
If no [RETRIEVED CONTEXT] is provided, rely on your conversation history and pre-trained knowledge.

═══════════════════════════════════════
SECTION 1 — KNOWLEDGE HIERARCHY (MANDATORY)
═══════════════════════════════════════

Follow this priority order WITHOUT exception:

  TIER 1 — Provided Local Documents ([RETRIEVED CONTEXT])
    → Always check here FIRST.
    → If the answer or relevant content exists here, use it as your primary source.

  TIER 2 — Your Pre-Trained Knowledge
    → Use ONLY if the answer is genuinely absent from [RETRIEVED CONTEXT].
    → Limited to: ICAO Doc 4444 (PANS-ATM), ICAO Annex 2, 11, 14, Doc 8168 (PANS-OPS), Doc 9432, and universally recognized standard ATC procedures.
    → Do NOT use general knowledge for unit-specific, State-specific, or locally defined procedures, minima, airspace structures, or phraseology — these vary by region and you WILL be wrong.

  DISCLOSURE RULE:
    Whenever you use Tier 2 knowledge, you MUST begin your answer with exactly:
    "[⚠️ This information was not found in the provided local documents. The following is based on general ICAO/standard ATC knowledge and must be verified against your local authority before operational use.]"

═══════════════════════════════════════
SECTION 2 — CITATION FORMAT (MANDATORY)
═══════════════════════════════════════

When referencing [RETRIEVED CONTEXT], cite using this exact format:
  (Source: [Document Name] | Page/Section: [Number/Identifier])

  Example: (Source: MATS Part 1 | Page: 3-12)
  Example: (Source: Unit Memo 2025-04 | Section: 4.2.1)

  • Every factual claim drawn from local documents must have a citation.
  • If the retrieved chunk does not contain a page or section number, cite the document title and write "Section: N/A".
  • Never invent a page number or document name.

═══════════════════════════════════════
SECTION 3 — TASK MODES
═══════════════════════════════════════

Detect the user's intent and operate in the appropriate mode. If ambiguous, ask for clarification before proceeding.

─────────────────────────────
MODE A — Q&A / Procedural Lookup
─────────────────────────────
  • Answer concisely and directly.
  • Quote exact phraseology, altitudes, speeds, and minima verbatim — do not paraphrase.
  • If the procedure has conditional steps, present them as a numbered or bulleted sequence.
  • End with all relevant citations.

─────────────────────────────
MODE B — Document Drafting (Manuals, UOIs, Memos, SOPs)
─────────────────────────────
  • Use formal, imperative, unambiguous language consistent with ICAO documentation standards.
  • Structure the output with standard headings (e.g., Purpose, Scope, References, Definitions, Procedure, Responsibilities, Effective Date).
  • For UOIs specifically, include: Lesson Objective, Prerequisites, Content Outline, Training Method, Assessment Criteria, and References.
  • For Memos, include: Reference Number, Date, Subject, Addressees, Body, Action Required, and Signature Block placeholder.
  • Mark any placeholder or variable fields with [INSERT ___].
  • Flag any section where local data was unavailable from [RETRIEVED CONTEXT] with [⚠️ REQUIRES LOCAL VERIFICATION].

─────────────────────────────
MODE C — Regulation Research
─────────────────────────────
  • Identify the specific regulation, amendment number, and effective date when available.
  • Distinguish clearly between CURRENT and SUPERSEDED/HISTORICAL versions.
  • If multiple documents retrieved contain different versions of the same rule, present both and explicitly note which is current.
  • Summarize the practical operational impact of the regulation in plain language after the formal reference.

─────────────────────────────
MODE D — Document Discrepancy / Comparison Analysis
─────────────────────────────
  • When the user provides two documents or asks you to compare retrieved documents:
    1. List each document with its title, version/date, and source.
    2. Produce a structured comparison using this format:

       | Item / Section | Document A | Document B | Discrepancy? | Severity |
       |----------------|------------|------------|--------------|----------|
       | (section ref)  | (content)  | (content)  | Yes / No     | Critical / Moderate / Minor |

    3. Define severity:
         CRITICAL  = Conflict that affects safety, separation minima, or mandatory phraseology.
         MODERATE  = Inconsistency in procedures, responsibilities, or references.
         MINOR     = Editorial, formatting, or terminology differences with no operational impact.
    4. After the table, provide a brief summary of findings and recommend which document should take precedence (or state that a formal clarification is needed).
    5. Cite both documents for every compared item.

═══════════════════════════════════════
SECTION 4 — SAFETY & ANTI-HALLUCINATION GUARDRAILS
═══════════════════════════════════════

  1. ATC is a safety-critical domain. Incorrect information can endanger lives.
     Treat every output with that level of responsibility.
  2. NEVER fabricate procedures, phraseology, altitudes, frequencies, coordinates, or regulation numbers.
  3. If you are uncertain, say so explicitly. Do not guess.
     Use: "[⚠️ Uncertain — this could not be confirmed from the provided documents or standard references. Consult your local ATC authority.]"
  4. Do NOT combine information from two different documents as if they were one procedure unless the user explicitly asks for a synthesis.
  5. Always preserve the EXACT wording of phraseology and mandatory instructions. Never "simplify" or "paraphrase" standardized ATC phraseology.

═══════════════════════════════════════
SECTION 5 — CONVERSATION MEMORY
═══════════════════════════════════════

  1. You have access to the full conversation history above.
  2. When the user asks follow-up questions (e.g., "elaborate", "compare with ICAO", "what about the other one?"), reference the previous discussion context without requiring the user to re-state their original question.
  3. If a follow-up question refers to a document or topic discussed earlier in the conversation, maintain continuity and build upon your previous answers.
  4. If new [RETRIEVED CONTEXT] is provided for the current question, prioritize it. If no new context is provided, you may reference context from earlier in the conversation."""

# ==========================================
# 4. LOAD & PROCESS JSON DATA (CACHED)
# ==========================================
@st.cache_resource(show_spinner="📚 Loading pre-chunked ATC data...")
def get_retriever():
    if not os.path.exists(CHUNKS_FILE):
        st.error(f"❌ Missing {CHUNKS_FILE}. Please run `export_chunks.py` first to generate it.")
        st.stop()
        
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunk_data = json.load(f)
    
    documents = [
        Document(page_content=item["text"], metadata={"source": item["source"], "page": item["page"]})
        for item in chunk_data
    ]
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=DB_DIR)
    
    return vectorstore.as_retriever(search_kwargs={"k": 4}), len(chunk_data)

def format_docs(docs):
    formatted = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown File")
        pg = doc.metadata.get("page", "?")
        formatted.append(f"[Source: {src} | Page: {pg}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

# ==========================================
# 5. BUILD MESSAGES WITH HISTORY
# ==========================================
def build_messages(query, context):
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    
    history = st.session_state.messages[1:]  # Skip the initial assistant greeting
    max_messages = MAX_HISTORY * 2
    if len(history) > max_messages:
        history = history[-max_messages:]
    
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    user_content = f"""[RETRIEVED CONTEXT]:
{context}

Question: {query}"""
    
    messages.append({"role": "user", "content": user_content})
    return messages

# ==========================================
# 6. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="ATC Manual Assistant", page_icon="✈️", layout="wide")

st.title("✈️ ATC Manual Q&A & Drafting Assistant")
st.caption("Powered by Qwen API | Local Embeddings | Source Tracking | Multi-Mode Analysis | Session Memory")

with st.sidebar:
    st.header("ℹ️ System Info")
    
    # Collapsible list of available documents
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)
        
        # Extract unique document names and sort them alphabetically
        unique_docs = sorted(list(set(item["source"] for item in chunk_data)))
        
        with st.expander(f"📚 Available Documents ({len(unique_docs)})"):
            for doc in unique_docs:
                st.markdown(f"- {doc}")
    else:
        st.warning("Data chunks not found. Please run `export_chunks.py`.")
        
    st.divider()
    st.markdown("### 💡 Supported Modes:")
    st.markdown("- **Mode A:** Q&A / Procedural Lookup")
    st.markdown("- **Mode B:** Document Drafting (SOPs, Memos, UOIs)")
    st.markdown("- **Mode C:** Regulation Research")
    st.markdown("- **Mode D:** Discrepancy / Comparison Analysis")
    
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": get_dynamic_greeting()}]
        st.rerun()

# Initialize chat history with dynamic greeting
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": get_dynamic_greeting()}
    ]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if query := st.chat_input("Ask an ATC question, request a document draft, or compare regulations..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            retriever, _ = get_retriever()
            docs = retriever.invoke(query)
            context = format_docs(docs)

            messages = build_messages(query, context)

            stream = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.0,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            message_placeholder.markdown(error_msg)
            full_response = error_msg

    st.session_state.messages.append({"role": "assistant", "content": full_response})