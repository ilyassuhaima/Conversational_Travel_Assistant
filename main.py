"""
Kavak Conversational Travel Assistant
======================================
Architecture: LangGraph stateful agent with three specialized tools:
  1. FlightSearchTool   – filters mock JSON flight listings
  2. RAGTool            – answers visa/policy questions via FAISS vector search
  3. General fallback   – handled by the LLM directly

LLM Backend : Groq (llama-3.3-70b-versatile) — free tier, no billing required
Embeddings  : HuggingFace all-MiniLM-L6-v2 — fully local, no API key needed
Vector Store: FAISS (in-memory, persisted to disk)

Chunking Strategy:
  - Chunk size  : 512 chars  → captures complete policy statements without splitting rules
  - Chunk overlap: 50 chars  → ~10% overlap prevents boundary blindness at chunk edges
  - Splitter    : RecursiveCharacterTextSplitter (respects paragraph/sentence boundaries)

Why these values?
  - 512 is large enough to hold a full visa rule (multi-condition sentences) but small
    enough to keep retrieved context focused. At 1024 we'd retrieve irrelevant context
    alongside the answer; at 128 we'd split rules mid-sentence.
  - 50-char overlap ensures that a query matching content straddling two chunk boundaries
    still retrieves the correct context. 10% is the standard heuristic for policy docs.

Why HuggingFace all-MiniLM-L6-v2 for embeddings?
  - Runs fully locally — zero API calls, zero cost, no key needed
  - 384-dimensional vectors; fast and accurate for small knowledge bases (<1000 chunks)
  - Top performer on MTEB benchmarks among lightweight models
  - Downloads once (~90MB) and caches locally via HuggingFace hub
"""

import json
import os
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_PATH = BASE_DIR / "faiss_index"

FLIGHTS_PATH = DATA_DIR / "flights.json"
KNOWLEDGE_BASE_PATH = DATA_DIR / "visa_rules.md"

# ──────────────────────────────────────────────
# LLM & Embeddings
# ──────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=1024,
    # GROQ_API_KEY is read automatically from .env
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    # all-MiniLM-L6-v2 rationale:
    #   - Fully local — no API key, no cost, works offline after first download
    #   - ~90MB model, downloads once and caches via HuggingFace hub
    #   - 384-dimensional vectors; excellent semantic similarity for policy/visa text
    #   - Top lightweight model on MTEB benchmarks; fast inference on CPU
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # cosine similarity requires L2-normalized vectors
)


# ──────────────────────────────────────────────
# Pydantic models for structured LLM output
# ──────────────────────────────────────────────
class FlightQuery(BaseModel):
    """Structured flight search parameters extracted from natural language."""

    origin: str = Field(description="Departure city, e.g. 'Dubai'")
    destination: str = Field(description="Arrival city, e.g. 'Tokyo'")
    month: str | None = Field(
        default=None, description="Preferred travel month, e.g. 'August'"
    )
    alliance: str | None = Field(
        default=None,
        description="Preferred airline alliance: 'Star Alliance', 'oneworld', 'SkyTeam', or None",
    )
    airline: str | None = Field(
        default=None, description="Specific airline preference if any"
    )
    avoid_overnight_layover: bool = Field(
        default=False,
        description="True if user wants to avoid overnight layovers",
    )
    refundable_only: bool = Field(
        default=False, description="True if user wants only refundable tickets"
    )
    max_price_usd: float | None = Field(
        default=None, description="Maximum price in USD if specified"
    )
    cabin: str | None = Field(
        default=None,
        description="Cabin class preference: 'Economy', 'Business', 'First'",
    )


# ──────────────────────────────────────────────
# Vector Store (RAG)
# ──────────────────────────────────────────────
def build_or_load_vectorstore() -> FAISS:
    """
    Build FAISS index from the knowledge base, or load from disk if already built.

    Chunking parameters:
      chunk_size=512    — fits one complete visa rule or policy paragraph
      chunk_overlap=50  — 10% overlap prevents boundary blindness
      separators        — tries paragraph → sentence → word breaks in order
    """
    if FAISS_INDEX_PATH.exists():
        return FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    text = KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.create_documents([text])

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(FAISS_INDEX_PATH))
    return vectorstore


vectorstore = build_or_load_vectorstore()
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},  # return top-3 most relevant chunks
)


# ──────────────────────────────────────────────
# Tool Definitions
# ──────────────────────────────────────────────
@tool
def search_flights(query: str) -> str:
    """
    Search available flights based on a natural language query.
    Use this tool when the user asks about flight availability, prices,
    routes, airlines, alliances, layovers, or travel options.
    The query should describe what the user is looking for.
    """
    flights: list[dict] = json.loads(FLIGHTS_PATH.read_text(encoding="utf-8"))

    # Use the LLM to extract structured parameters from the free-text query
    extraction_llm = llm.with_structured_output(FlightQuery)
    params: FlightQuery = extraction_llm.invoke(
        f"Extract flight search parameters from this query: {query}"
    )

    results = flights

    # Filter: origin (case-insensitive partial match)
    if params.origin:
        results = [
            f for f in results
            if params.origin.lower() in f["from"].lower()
        ]

    # Filter: destination
    if params.destination:
        results = [
            f for f in results
            if params.destination.lower() in f["to"].lower()
        ]

    # Filter: alliance
    if params.alliance:
        results = [
            f for f in results
            if params.alliance.lower() in f["alliance"].lower()
        ]

    # Filter: specific airline
    if params.airline:
        results = [
            f for f in results
            if params.airline.lower() in f["airline"].lower()
        ]

    # Filter: avoid overnight layovers
    if params.avoid_overnight_layover:
        results = [f for f in results if not f["overnight_layover"]]

    # Filter: refundable only
    if params.refundable_only:
        results = [f for f in results if f["refundable"]]

    # Filter: max price
    if params.max_price_usd:
        results = [f for f in results if f["price_usd"] <= params.max_price_usd]

    # Filter: cabin class
    if params.cabin:
        results = [
            f for f in results
            if f.get("cabin", "").lower() == params.cabin.lower()
        ]

    # Filter: month (match against departure_date)
    if params.month:
        month_map = {
            "january": "01", "february": "02", "march": "03",
            "april": "04", "may": "05", "june": "06",
            "july": "07", "august": "08", "september": "09",
            "october": "10", "november": "11", "december": "12",
        }
        month_num = month_map.get(params.month.lower())
        if month_num:
            results = [
                f for f in results
                if f["departure_date"].split("-")[1] == month_num
            ]

    if not results:
        return (
            "No flights found matching your criteria. "
            "Try relaxing some filters — for example, removing the alliance "
            "or overnight-layover restriction."
        )

    # Sort by price ascending
    results.sort(key=lambda x: x["price_usd"])

    # Format output
    lines = [f"Found {len(results)} flight(s) matching your criteria:\n"]
    for f in results:
        layover_str = (
            f"  Layovers  : {', '.join(f['layovers'])} "
            f"({'overnight ⚠️' if f['overnight_layover'] else 'no overnight'})"
            if f["layovers"]
            else "  Layovers  : Non-stop ✈️"
        )
        lines.append(
            f"── {f['airline']} ({f['alliance']}) [{f['flight_id']}] ──\n"
            f"  Route     : {f['from']} → {f['to']} | {f['cabin']}\n"
            f"  Dates     : {f['departure_date']} → {f['return_date']}\n"
            f"{layover_str}\n"
            f"  Duration  : {f['duration_hours']}h\n"
            f"  Price     : ${f['price_usd']:,} USD\n"
            f"  Refundable: {'✅ Yes' if f['refundable'] else '❌ No'}\n"
            f"  Seats left: {f['seats_available']}\n"
        )

    return "\n".join(lines)


@tool
def answer_travel_policy(question: str) -> str:
    """
    Answer questions about visa requirements, refund policies, baggage rules,
    transit visas, layover policies, or airline alliance benefits.
    Use this tool when the user asks policy or regulation questions.
    """
    docs = retriever.invoke(question)

    if not docs:
        return (
            "I couldn't find specific policy information for your question. "
            "Please check official airline or embassy websites for the most current rules."
        )

    # Combine retrieved chunks into context
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    # Use LLM to synthesize a focused answer from retrieved context
    synthesis_prompt = f"""You are a knowledgeable travel policy assistant.
Using ONLY the following retrieved policy excerpts, answer the user's question clearly and concisely.
If the context doesn't fully cover the question, say so and recommend checking official sources.

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

Provide a helpful, accurate answer based on the context above."""

    response = llm.invoke([HumanMessage(content=synthesis_prompt)])
    return response.content


# ──────────────────────────────────────────────
# LangGraph Agent
# ──────────────────────────────────────────────
TOOLS = [search_flights, answer_travel_policy]

llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """You are Aria, a friendly and knowledgeable travel assistant powered by Kavak Travel.
You help users plan international trips by searching for flights and answering travel policy questions.

Your capabilities:
1. 🔍 Search flights — use the `search_flights` tool for any flight-related queries
2. 📋 Policy & visa info — use the `answer_travel_policy` tool for visa rules, refund policies, baggage, and layover questions
3. 💬 General travel advice — answer conversationally when no tool is needed

Guidelines:
- Always be warm, helpful, and concise
- When presenting flight results, highlight the best value option
- Proactively mention relevant policies (e.g., if a flight has an overnight layover, note the user can ask about transit visas)
- If a query is ambiguous, ask one clarifying question
- Never make up flight data or policy rules — use the provided tools
- Format flight results cleanly; use emoji sparingly but effectively"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: AgentState) -> dict[str, Any]:
    """Core LLM reasoning node — decides whether to call a tool or respond directly."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Conditional edge: route to tools if tool_calls present, else end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def build_graph() -> StateGraph:
    """Assemble the LangGraph computation graph."""
    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")  # after tool execution, return to agent

    return graph.compile()


# ──────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────
_graph = build_graph()


def chat(user_message: str, history: list[dict] | None = None) -> str:
    """
    Send a message and get a response. Maintains conversation history.

    Args:
        user_message: The user's input text.
        history: List of previous messages as dicts with 'role' and 'content'.

    Returns:
        The assistant's response as a string.
    """
    messages = []
    if history:
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_message))

    result = _graph.invoke({"messages": messages})
    return result["messages"][-1].content


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────
def main():
    """Run the assistant interactively in the terminal."""
    print("=" * 60)
    print("  ✈️  Kavak Travel Assistant (Aria)")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    history: list[dict] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! Safe travels. ✈️")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Aria: Goodbye! Safe travels. ✈️")
            break

        response = chat(user_input, history)
        print(f"\nAria: {response}")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()