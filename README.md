# Agentic RAG Resume-JD Matcher

A command-line agent that scores how well a resume matches a job description — and can explain its reasoning through natural follow-up conversation — by combining Retrieval-Augmented Generation (RAG) with tool-calling agent behavior.

## What makes this "agentic," not just a RAG pipeline

A simple RAG script would always do the same fixed sequence: embed query → retrieve → generate. That's not an agent — it's a pipeline with no decisions to make.

This project is different because the model itself decides:
- **What to search for** — it constructs its own retrieval query based on the job description's content, so a backend JD and a frontend JD retrieve different resume sections.
- **When to search again** — on a follow-up question like *"why did I score low on testing?"*, the agent independently decides to call `search_resume` a second time with a new query, rather than just replaying its earlier answer. This was verified directly: asking that question caused the agent to re-retrieve and ground its answer in specific resume text (citing "Selenium" and "CRUD operations" by name) instead of reasoning from memory alone.
- **When *not* to score** — if given only a job title instead of real job description text, the agent recognizes the input is insufficient and asks for the actual JD, rather than inventing plausible-sounding requirements to score against. (This was a real bug caught during testing — see Design Decisions below.)

## Architecture
resume.txt --> ingest.py --> ChromaDB (persisted, chunked by section)
    ^
    |
User <--> agent.py (Gemini function-calling loop) <--> tools.py
    |
search_resume(query)
score_jd_match(jd_text, retrieved_context)


1. **`ingest.py`** — chunks `resume.txt` by section (Summary, Technical Skills, Projects, Education) and stores embeddings in a persistent ChromaDB collection. Run once, or whenever the resume changes.
2. **`tools.py`** — two functions exposed to the agent as tools:
   - `search_resume(query)` — semantic search over the resume chunks.
   - `score_jd_match(jd_text, retrieved_context)` — prompts Gemini to return a structured JSON match score, matching/missing skills, and a verdict, grounded only in the retrieved context (not the full resume).
3. **`agent.py`** — the multi-turn dispatch loop. Registers both tools with Gemini, maintains conversation history across turns, and executes whichever tool the model decides to call.

## Setup

```bash
git clone <your-repo-url>
cd Agentic_RAG_Resume_Matcher
python -m venv venv
./venv/Scripts/activate      # Windows
pip install -r requirements.txt
```

Create a `.env` file with: GEMINI_API_KEY=your_key_here

> Note: `resume.txt` in this repo uses a placeholder name for public demo purposes — swap in your own resume text to use this for real.

Build the resume index (run once, or after editing `resume.txt`):
```bash
python ingest.py
```

Run the agent:
```bash
python agent.py
```

At the `You:` prompt, either paste a job description directly, type `jd` to load one from `jd.txt`, or ask a follow-up question about a previous score.

## Example interaction
You: jd
[loaded jd.txt]
[agent] search_resume succeeded
[agent] score_jd_match succeeded
Assistant: Based on a comparison of your resume with the Junior Python
Backend Developer job description, your match score is 45%...

You: Why exactly did I score low on testing?
[agent] search_resume succeeded
Assistant: You scored low on the testing requirement because of a mismatch
between the specific testing tools requested... Your resume mentions using
Selenium for testing login and CRUD operations, while the job description
explicitly asks for PyTest...


## Design decisions & tradeoffs

- **Why Gemini instead of the Claude API:** Anthropic's API has no free tier; Gemini's free tier made iterative development feasible. This required adapting the same tool-calling concepts (structured schemas, dispatch on tool name, multi-turn state) to a different SDK.
- **Why `PersistentClient` instead of an in-memory Chroma client:** the resume only needs to be ingested once; every subsequent run of `agent.py` should reuse the same index rather than re-embedding from scratch.
- **Why `score_jd_match` requires the model to pass `retrieved_context` explicitly, rather than the code fetching it automatically:** this keeps the retrieval step a genuine agent decision rather than a hidden, hardcoded step — the model has to actually call `search_resume` first and choose what to hand off.
- **A real bug found during testing:** when given a bare job title instead of real JD text, the agent initially called `score_jd_match` anyway and fabricated a plausible-but-fictional set of job requirements to score against. Fixed with an explicit system instruction telling the model not to invent job requirements and to ask for the real JD text when input is insufficient. This is documented because it's a good example of an LLM agent failure mode that isn't obvious until you deliberately test edge cases, not just the happy path.
- **Error handling:** tool failures (missing ChromaDB index, malformed JSON from the model, API quota errors) are caught and surfaced as explicit `{"error": ...}` results rather than crashing or silently printing a wrong answer as if it succeeded.
- **Free-tier quota limits:** Gemini's free tier caps some models at 20 requests/day; this project uses `gemini-3.5-flash-lite` for a higher daily limit suited to iterative testing.

## Known limitations

- Conversation history resets each time `agent.py` is restarted (no persistence across sessions).
- Single resume, single user — not designed for multi-user use.
- Match scoring quality depends on Gemini's judgment; it is not a deterministic or auditable scoring system.

## Tech stack

Python, Google Gemini API (`google-genai`), ChromaDB (persistent vector store), python-dotenv.
