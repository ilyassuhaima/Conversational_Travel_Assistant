# ✈️ Kavak Conversational Travel Assistant

A production-quality conversational travel assistant built with **LangGraph**, **Claude (Anthropic)**, **FAISS**, and **Streamlit**. The assistant — named **Aria** — helps users plan international trips by searching flights and answering visa/policy questions through natural language.

---

## 🗂️ Repository Structure

```
kavak_travel_assistant/
├── main.py              # Core agent logic (LangGraph + tools + CLI)
├── streamlit_app.py     # Streamlit chat UI
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── data/
    ├── flights.json     # Mock flight listings (12 flights, 4 airlines, 3 alliances)
    └── visa_rules.md    # Knowledge base: visa rules, refund, baggage, layover policies
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/kavak-travel-assistant.git
cd kavak-travel-assistant
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
GROQ_API_KEY=gsk_...
```

> **Note:** Only one API key needed — get it free (no credit card) at **console.groq.com**.
> Embeddings run fully locally via HuggingFace (`all-MiniLM-L6-v2`). The model downloads once (~90MB) and caches locally. No OpenAI or Anthropic key required.

### 5. Run the CLI

```bash
python main.py
```

### 6. Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

---

## 🏗️ System Overview

```
User Input
    │
    ▼
┌─────────────────────────────────────────┐
│           LangGraph Agent               │
│                                         │
│  ┌──────────┐    ┌──────────────────┐   │
│  │  Agent   │───▶│   Tool Router    │   │
│  │  Node    │    │ (Claude decides) │   │
│  └──────────┘    └──────────────────┘   │
│        ▲                │               │
│        │       ┌────────┴────────┐      │
│        │       ▼                 ▼      │
│        │  ┌─────────┐   ┌─────────────┐│
│        │  │ Flight  │   │  RAG Tool   ││
│        │  │ Search  │   │  (FAISS +   ││
│        │  │  Tool   │   │   Claude)   ││
│        │  └────┬────┘   └──────┬──────┘│
│        │       └───────┬───────┘       │
│        └───────────────┘               │
└─────────────────────────────────────────┘
    │
    ▼
Assistant Response
```

### Flow Description

1. **User sends a message** → enters the LangGraph `agent` node
2. **Agent node** calls Claude with the conversation history + system prompt + tool definitions
3. **Claude decides** whether to call a tool or respond directly
4. **If tool needed** → routes to `ToolNode` which executes the appropriate tool
5. **Tool output** is appended to the message history and returned to the agent
6. **Agent responds** with a final synthesized answer
7. **Conditional edge** checks for remaining tool calls; loops if needed, ends otherwise

---

## 🧠 Prompt Strategies

### System Prompt (Aria persona)
The system prompt establishes Aria as a friendly, knowledgeable travel assistant with clear behavioral guidelines:
- Always use tools for factual data (never hallucinate flight prices or visa rules)
- Format flight results cleanly with emoji for scannability
- Proactively surface relevant information (e.g., mention transit visa policy when overnight layover is found)

### Structured Output Extraction
Flight query parsing uses `llm.with_structured_output(FlightQuery)` — a Pydantic model that forces Claude to extract normalized fields (origin, destination, alliance, overnight preference, etc.) from free-text queries. This is more robust than regex or keyword matching because it handles paraphrasing ("I want to avoid sleeping at the airport" → `avoid_overnight_layover: true`).

### RAG Synthesis Prompt
The RAG tool uses a two-step approach:
1. **Retrieve** top-3 chunks from FAISS (similarity search)
2. **Synthesize** — a separate Claude call with a strict instruction to answer only from retrieved context, preventing hallucination

---

## 🔍 RAG & Retrieval Details

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

## 🤖 Agent Logic (LangGraph)

### Why LangGraph over LangChain AgentExecutor?
- **Explicit state graph**: every node and edge is visible and debuggable
- **Typed state**: `AgentState` with annotated message list ensures correct message accumulation
- **Conditional routing**: `should_continue()` cleanly separates tool-calling from final response
- **Loops safely**: the `tools → agent` edge enables multi-tool calls in one turn without manual orchestration

### Graph Nodes
| Node | Role |
|------|------|
| `agent` | Calls Claude with full message history; decides whether to use tools |
| `tools` | LangGraph `ToolNode` — executes whichever tool Claude selected |

### Edges
| Edge | Condition |
|------|-----------|
| `START → agent` | Always |
| `agent → tools` | If Claude's response contains `tool_calls` |
| `agent → END` | If Claude responded directly (no tool calls) |
| `tools → agent` | Always (return tool output to agent for synthesis) |

---

## 💡 Sample Outputs

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

## 🎁 Bonus Features

- **Structured LLM output parsing** with Pydantic for robust query normalization
- **Two-stage RAG**: retrieve → synthesize (prevents hallucination in policy answers)
- **Streamlit UI** with suggested prompts, sidebar info panel, and conversation reset
- **FAISS index persistence** (skips re-embedding on subsequent runs)
- **Graceful no-results handling** with actionable suggestions to relax filters

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq (`llama-3.3-70b-versatile`) — free, no billing |
| Agent Framework | LangGraph 0.2+ |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` — fully local, free |
| Vector Store | FAISS (faiss-cpu) |
| Structured Output | Pydantic v2 |
| Frontend | Streamlit |
| Data | JSON (flights) + Markdown (policies) |

---

## 📝 Notes

- The FAISS index is built automatically on first run and cached to `faiss_index/`
- All LLM calls use `temperature=0` for deterministic, factual responses
- The agent is stateless between sessions; conversation history is managed client-side