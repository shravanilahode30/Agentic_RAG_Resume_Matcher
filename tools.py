import os
import json
import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def search_resume(query: str, n_results: int = 2) -> str:
    """Retrieve the most relevant resume chunks for a given query."""
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    try:
        collection = chroma_client.get_collection(name="resume")
    except Exception:
        return "ERROR: Resume index not found. Run ingest.py first to build it from resume.txt."

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    documents = results["documents"][0]
    sections = [meta["section"] for meta in results["metadatas"][0]]

    combined = "\n\n".join(
        f"[{section}]\n{doc}" for section, doc in zip(sections, documents)
    )
    return combined


def score_jd_match(jd_text: str, retrieved_context: str) -> dict:
    """Score how well the retrieved resume context matches a job description."""
    prompt = f"""You are a resume-to-job-description matching engine. Compare the resume context and job description provided below, and return ONLY a JSON object. No explanation, no markdown, no extra text — just the raw JSON.

Fields to extract:
- match_score: integer from 0 to 100, representing overall skill match percentage
- matching_skills: array of strings — skills/technologies present in BOTH the resume context and job description (treat close synonyms as matches, e.g. "JS" and "JavaScript")
- missing_skills: array of strings — skills/technologies required in the job description but NOT found in the resume context
- verdict: one short sentence (max 15 words) summarizing the fit

Example 1:

Resume: \"\"\"Python developer with 2 years experience in Django, REST APIs, and PostgreSQL. Familiar with Git and basic AWS deployment.\"\"\"

Job description: \"\"\"Looking for a Python/Django developer with REST API experience. Docker and AWS knowledge preferred.\"\"\"

Output: {{"match_score": 75, "matching_skills": ["Python", "Django", "REST APIs", "AWS"], "missing_skills": ["Docker"], "verdict": "Strong core match with one missing skill: Docker."}}

Example 2:

Resume: \"\"\"Frontend developer skilled in React, JavaScript, and CSS.\"\"\"

Job description: \"\"\"Backend role requiring Python, Django, and PostgreSQL.\"\"\"

Output: {{"match_score": 10, "matching_skills": [], "missing_skills": ["Python", "Django", "PostgreSQL"], "verdict": "Poor match — candidate's skills are frontend-focused, role is backend."}}

Now analyze this actual resume context and job description:

Resume context: \"\"\"{retrieved_context}\"\"\"

Job description: \"\"\"{jd_text}\"\"\"

Output:"""
    
    interaction = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=prompt
    )

    raw_output = interaction.output_text.strip()

    if raw_output.startswith("```"):
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]

    start = raw_output.find("{")
    end = raw_output.rfind("}") + 1
    raw_output = raw_output[start:end]

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {"error": "Could not parse model output as JSON", "raw": raw_output}