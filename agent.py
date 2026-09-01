import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tools import search_resume, score_jd_match

load_dotenv()
client = genai.Client()

MAX_TOOL_CALLS_PER_TURN = 5  # guard against infinite loops (Day 18 lesson)


# --- Tool schemas ---
search_resume_function = {
    "name": "search_resume",
    "description": "Search the candidate's resume for content relevant to a specific query or skill. Use this to find resume context before scoring a job description, or to answer a follow-up question about why a particular skill matched or didn't match.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in the resume, e.g. 'React and frontend experience' or 'Django REST Framework'"
            }
        },
        "required": ["query"]
    }
}

score_jd_match_function = {
    "name": "score_jd_match",
    "description": "Score how well the candidate's resume matches a job description. Call search_resume first to retrieve relevant resume context, then call this with the job description text and that retrieved context.",
    "parameters": {
        "type": "object",
        "properties": {
            "jd_text": {
                "type": "string",
                "description": "The full text of the job description."
            },
            "retrieved_context": {
                "type": "string",
                "description": "Resume context previously retrieved via search_resume."
            }
        },
        "required": ["jd_text", "retrieved_context"]
    }
}

tools = types.Tool(function_declarations=[search_resume_function, score_jd_match_function])
config = types.GenerateContentConfig(
    tools=[tools],
    system_instruction=(
        "You are a resume-to-job-description matching assistant. "
        "Only call score_jd_match when the user has provided actual job description "
        "text with real requirements — not just a job title or a vague topic. "
        "If the user's message is only a title (e.g. 'Frontend Developer Intern') or "
        "otherwise too short or vague to be a real job description, do NOT invent or "
        "assume any job requirements. Instead, respond in plain text asking the user "
        "to paste the full job description text."
    )
)


def call_with_retry(contents, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=contents,
                config=config
            )
        except Exception as e:
            print(f"[agent] Attempt {attempt + 1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    print("[agent] All retries failed — Gemini API may be down.")
    return None


def dispatch_tool(tool_name: str, tool_args: dict) -> dict:
    if tool_name == "search_resume":
        result_text = search_resume(tool_args["query"])
        if result_text.startswith("ERROR:"):
            return {"error": result_text}
        return {"result": result_text}
    elif tool_name == "score_jd_match":
        return score_jd_match(tool_args["jd_text"], tool_args["retrieved_context"])
    else:
        return {"error": f"Unknown tool: {tool_name}"}


def run_agent(conversation: list):
    """Runs the tool-call loop for the latest user turn. Mutates and returns conversation."""
    for _ in range(MAX_TOOL_CALLS_PER_TURN):
        response = call_with_retry(conversation)
        if response is None:
            return conversation

        part = response.candidates[0].content.parts[0]

        if part.function_call:
            tool_name = part.function_call.name
            tool_args = dict(part.function_call.args)

            result = dispatch_tool(tool_name, tool_args)

            if "error" in result:
                print(f"[agent] {tool_name} FAILED: {result}")
            else:
                print(f"[agent] {tool_name} succeeded")

            # append model's function call turn, then our function response turn
            conversation.append(response.candidates[0].content)
            function_response_part = types.Part.from_function_response(
                name=tool_name,
                response=result
            )
            conversation.append(types.Content(role="user", parts=[function_response_part]))
            # loop again — model may call another tool or answer in text
        else:
            print(f"\nAssistant: {part.text}\n")
            conversation.append(response.candidates[0].content)
            return conversation

    print("[agent] Max tool calls reached for this turn — stopping to avoid a loop.")
    return conversation


if __name__ == "__main__":
    conversation = []
    print("Agentic RAG Resume-JD Matcher — paste a JD, type 'jd' to load jd.txt, or ask a follow-up. Ctrl+C to quit.\n")
    while True:
        user_input = input("You: ")

        if user_input.strip().lower() == "jd":
            with open("jd.txt", "r", encoding="utf-8") as f:
                user_input = f.read()
            print("[loaded jd.txt]")

        conversation.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))
        conversation = run_agent(conversation)