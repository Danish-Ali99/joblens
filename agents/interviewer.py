"""Interview Coach — predicts likely interview questions for this JD."""
from langchain_core.messages import SystemMessage, HumanMessage
from .base import get_llm, retrieve_resume_context

RETRIEVAL_QUERY = (
    "projects experience technical work deep accomplishments responsibilities "
    "challenges impact"
)


SYSTEM = """You are an interview coach who has seen thousands of interviews
for engineering and AI roles. Predict the questions a real interviewer would
ask THIS candidate for THIS role, based on the resume and JD.

Produce EXACTLY these markdown sections:

## Likely Behavioral Questions (3)
For each, give:
- The question (something specific to the candidate's resume + JD)
- A 1-2 sentence STAR-method hint on how to answer

## Likely Technical Questions (3-4)
For each, give:
- The question (drawn from technologies/topics in the JD the candidate claims
  on their resume — the interviewer will dig here)
- A 2-3 line "what a good answer covers" hint

## Likely Project Deep-Dive (2)
The interviewer will pick specific projects from the resume and grill. Predict
the top 2 follow-up questions per project, with answer hints.

## Curveball / Stress Questions (2)
Questions about gaps, weaknesses, or unusual choices the interviewer might
notice. With a hint on how to answer without sounding defensive.

Be specific. Don't give generic "Tell me about yourself" — name the actual
project, technology, or claim from THIS resume that the interviewer will hit."""


def interviewer_agent(state):
    resume_context = retrieve_resume_context(state, RETRIEVAL_QUERY, k=3)
    llm = get_llm(temperature=0.6)
    response = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=(
            f"Resume (top-k relevant sections via RAG):\n{resume_context}\n\n"
            f"Job Description:\n{state['job_description']}"
        )),
    ])
    return {"interview_questions": response.content}
