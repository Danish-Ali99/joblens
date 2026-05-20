"""Cover Letter Writer — drafts a tailored cover letter."""
from langchain_core.messages import SystemMessage, HumanMessage
from .base import get_llm, retrieve_resume_context

RETRIEVAL_QUERY = (
    "summary motivation interests career goals projects education leadership "
    "experience accomplishments"
)


SYSTEM = """You are a senior career coach who writes cover letters that get
interviews. You write naturally, specifically, and without buzzword soup.

Draft a complete, ready-to-send cover letter based on the candidate's resume
and the target job description.

Rules:
- Open with a specific hook — why THIS company / role, not generic.
- 2 short paragraphs in the body, each grounded in a specific resume fact
  that maps to a specific JD requirement.
- 1 closing paragraph: clear next step, calm confidence, no begging.
- Total length: 220-300 words. Cover letters longer than this don't get read.
- No "I am writing to express my interest in the position..." — that's an
  instant skip line.
- Use the candidate's real name from the resume.
- Address it to "Hiring Manager" unless a name appears in the JD.
- End with a placeholder for the candidate's signature.

Output ONLY the cover letter text, no preamble, no markdown headers, no
explanation. Just the letter as it would be pasted into an email."""


def cover_letter_agent(state):
    resume_context = retrieve_resume_context(state, RETRIEVAL_QUERY, k=6)
    llm = get_llm(temperature=0.7)
    response = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=(
            f"Resume (top-k relevant sections via RAG):\n{resume_context}\n\n"
            f"Job Description:\n{state['job_description']}"
        )),
    ])
    return {"cover_letter": response.content}
