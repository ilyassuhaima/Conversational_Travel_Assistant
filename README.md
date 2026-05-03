# ✈️ Kavak Conversational Travel Assistant

A conversational AI travel assistant named **Aria** that helps users search flights and answer travel policy questions using natural language.

## Features

- **Flight Search**: Find flights by route, airline alliance, price, cabin class, and more
- **Travel Policies**: Answer questions about visas, refunds, baggage, and layover rules
- **Conversational Interface**: Chat naturally with the assistant
- **Web UI**: Streamlit-based chat interface with conversation history
- **CLI Mode**: Terminal-based interaction
- **Conversation Management**: Save, switch between, and continue previous conversations

## Technical Details

### Architecture

- **LLM**: OpenAI GPT-4o-mini for reasoning and response generation
- **Agent Framework**: LangGraph for stateful conversation management
- **Vector Store**: FAISS for semantic search over travel policy documents
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (local, no API required)
- **Frontend**: Streamlit for web interface

### Data Sources

- **Flights**: Mock JSON data with 12 sample flights across 4 airlines and 3 alliances
- **Policies**: Markdown knowledge base covering visa rules, refund policies, baggage allowances, and airline alliance benefits

### Key Components

1. **Flight Search Tool**: Parses natural language queries into structured parameters using GPT-4's structured output, then filters JSON flight data
2. **RAG Tool**: Retrieves relevant policy chunks from FAISS vector store, synthesizes answers using GPT-4o-mini
3. **Agent Logic**: LangGraph state machine that routes between tool calls and direct responses

### Setup Requirements

- **Python**: 3.8+
- **API Key**: OpenAI API key
- **Dependencies**: Listed in `requirements.txt`
- **Embeddings**: Downloads ~90MB model on first run (cached locally)

### File Structure

```
.
├── main.py              # Core agent and CLI
├── streamlit_app.py     # Web UI
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── data/
│   ├── flights.json     # Flight data
│   └── visa_rules.md    # Policy knowledge base
└── faiss_index/         # Auto-generated vector store (created on first run)
```

## Usage Examples

**Flight Search**:
- "Find round-trip flights from Dubai to Tokyo in August with Star Alliance airlines"
- "Show me the cheapest flights to Paris under $500"

**Policy Questions**:
- "Do UAE citizens need a visa for Japan?"
- "What's the refund policy for non-refundable tickets?"

**Web UI Features**:
- **Conversation History**: Access previous chats from the collapsible sidebar
- **New Chat**: Start fresh conversations with the "New Chat" button
- **Continue Conversations**: Click on any previous chat to resume where you left off
- **Auto-generated Titles**: Conversations are automatically titled based on your first message

## Development

The assistant uses structured LLM output for robust query parsing and two-stage RAG (retrieve + synthesize) to prevent hallucination in policy answers. All components are designed for easy extension with real APIs or additional data sources.

A production-quality conversational travel assistant built with **LangGraph**, **OpenAI**, **FAISS**, and **Streamlit**. The assistant, named **Aria**, helps users plan international trips by searching flights and answering visa/policy questions through natural language.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Conversational_Travel_Assistant.git
cd Conversational_Travel_Assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
```

### 5. Run the CLI

```bash
python main.py
```

### 6. Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

---

## System Overview

```
User Input
    │
    ▼
┌─────────────────────────────────────────┐
│           LangGraph Agent               │
│                                         │
│  ┌──────────┐    ┌──────────────────┐   │
│  │  Agent   │───▶│   Tool Router    │   │
│  │  Node    │    │ (Agent decides)  │   │
│  └──────────┘    └──────────────────┘   │
│        ▲                │               │
│        │       ┌────────┴────────┐      │
│        │       ▼                 ▼      │
│        │  ┌─────────┐   ┌─────────────┐ │
│        │  │ Flight  │   │  RAG Tool   │ │
│        │  │ Search  │   │  (FAISS+    | │
│        │  │  Tool   │   │   LLM)      │ │
│        │  └────┬────┘   └──────┬──────┘ │
│        │       └───────┬───────┘        │
│        └───────────────┘                │
└─────────────────────────────────────────┘
    │
    ▼
Assistant Response
```

### Flow Description

1. **User sends a message** → enters the LangGraph `agent` node
2. **Agent node** calls Claude with the conversation history + system prompt + tool definitions
3. **LLM decides** whether to call a tool or respond directly
4. **If tool needed** → routes to `ToolNode` which executes the appropriate tool
5. **Tool output** is appended to the message history and returned to the agent
6. **Agent responds** with a final synthesized answer
7. **Conditional edge** checks for remaining tool calls; loops if needed, ends otherwise

---

## Prompt Strategies

### System Prompt (Aria persona)
The system prompt establishes Aria as a friendly, knowledgeable travel assistant with clear behavioral guidelines:
- Always use tools for factual data (never hallucinate flight prices or visa rules)
- Format flight results cleanly with emoji for scannability
- Proactively surface relevant information (e.g., mention transit visa policy when overnight layover is found)

### Structured Output Extraction
Flight query parsing uses `llm.with_structured_output(FlightQuery)` — a Pydantic model that forces OpenaAI to extract normalized fields (origin, destination, alliance, overnight preference, etc.) from free-text queries. This is more robust than regex or keyword matching because it handles paraphrasing ("I want to avoid sleeping at the airport" → `avoid_overnight_layover: true`).

### RAG Synthesis Prompt
The RAG tool uses a two-step approach:
1. **Retrieve** top-3 chunks from FAISS (similarity search)
2. **Synthesize** — a separate llm call with a strict instruction to answer only from retrieved context, preventing hallucination

---

## RAG & Retrieval Details

### Embedding Model: `text-embedding-3-small`
- Chosen over `text-embedding-3-large`: our KB has ~20 chunks — larger model adds cost with no measurable retrieval gain at this scale
- Chosen over open-source (`sentence-transformers`): avoids an additional dependency, and OpenAI v3 embeddings outperform most open-source models on MTEB semantic similarity benchmarks
- 1536-dimensional output vectors; sufficient for cosine similarity over a small corpus

### Chunking Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `chunk_size` | 512 tokens | Fits one complete visa rule or policy paragraph. At 1024, retrieved context becomes noisy; at 128, multi-condition rules get split |
| `chunk_overlap` | 50 tokens | ~10% overlap — standard heuristic for policy documents. Prevents boundary blindness when a query maps to content straddling two chunk edges |
| `separators` | `["\n\n", "\n", ". ", " ", ""]` | Tries paragraph → sentence → word breaks in order, preserving semantic units |

### Vector Store: FAISS (in-memory + disk persistence)
- No server, no Docker, no API keys required
- `save_local()` / `load_local()` persists the index between runs
- At this scale (<1,000 documents), FAISS cosine similarity is indistinguishable in quality from managed services like Pinecone or Weaviate
- `search_type="similarity"`, `k=3` returns top-3 chunks — enough context for most policy questions without overwhelming the synthesis prompt

---

## Agent Logic (LangGraph)

### Why LangGraph over LangChain AgentExecutor?
- **Explicit state graph**: every node and edge is visible and debuggable
- **Typed state**: `AgentState` with annotated message list ensures correct message accumulation
- **Conditional routing**: `should_continue()` cleanly separates tool-calling from final response
- **Loops safely**: the `tools → agent` edge enables multi-tool calls in one turn without manual orchestration

### Graph Nodes
| Node | Role |
|------|------|
| `agent` | Calls OpenAI with full message history; decides whether to use tools |
| `tools` | LangGraph `ToolNode` — executes whichever tool the llm selected |

### Edges
| Edge | Condition |
|------|-----------|
| `START → agent` | Always |
| `agent → tools` | If llm's response contains `tool_calls` |
| `agent → END` | If llm responded directly (no tool calls) |
| `tools → agent` | Always (return tool output to agent for synthesis) |

---

## Sample Outputs

### Flight Search
**Input:** `Find me a round-trip to Tokyo in August with Star Alliance airlines only. I want to avoid overnight layovers.`

```
Found 2 flight(s) matching your criteria:

── Turkish Airlines (Star Alliance) [TK-001] ──
  Route     : Dubai → Tokyo | Economy
  Dates     : 2024-08-15 → 2024-08-30
  Layovers  : Istanbul (no overnight)
  Duration  : 14h
  Price     : $950 USD
  Refundable: ✅ Yes
  Seats left: 8

── ANA (Star Alliance) [NH-002] ──
  Route     : Dubai → Tokyo | Economy
  Dates     : 2024-08-18 → 2024-08-31
  Layovers  : Abu Dhabi, Narita (no overnight)
  Duration  : 16h
  Price     : $1,100 USD
  Refundable: ✅ Yes
  Seats left: 4
```

### Visa Query
**Input:** `Can a UAE passport holder travel to Japan without a visa?`

> Yes! UAE passport holders can enter Japan visa-free for up to 30 days for tourism. Your passport must be valid for at least 6 months beyond your stay, and you may be asked to show a return ticket at immigration. No advance visa application is needed.

### Policy Query
**Input:** `What happens if I cancel a refundable ticket 30 hours before departure?`

> Canceling a refundable ticket between 24 and 48 hours before departure incurs a 15% fee (deducted from your refund). Since 30 hours falls in this window, you'd receive 85% of the ticket price back, processed within 7–14 business days.

---
