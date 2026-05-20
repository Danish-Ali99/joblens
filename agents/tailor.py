"""Resume Tailor — suggests specific edits to the resume for this JD."""
from langchain_core.messages import SystemMessage, HumanMessage
from .base import get_llm, retrieve_resume_context

RETRIEVAL_QUERY = (
    "experience projects bullets accomplishments achievements impact summary "
    "internships work"
)


SYSTEM = """You are an expert resume editor specializing in tailoring resumes
to specific job descriptions. You write bullets that are tight, quantified, and
keyword-aligned.

Produce EXACTLY these markdown sections:

## Keywords to Add
A comma-separated list of JD-specific keywords missing from the resume that
should be naturally woven in. Be honest — don't suggest keywords the candidate
doesn't actually have experience with.

## Bullets to Rewrite
For 3-5 existing resume bullets, write a tailored rewrite in this format:

**Original:** <quote what's roughly on the resume now>
**Rewrite:** <a sharper, JD-aligned version>
**Why:** <one line on why this lands better for THIS JD>

## Bullets to Add
2-3 NEW bullets to add to the resume (based on projects/experience the
candidate already has), worded to align with the JD. Same format as above
(skip Original).

## Top-of-Resume Summary Rewrite
Rewrite the candidate's resume summary (2-3 sentences) to laser-target this
specific JD. Keep it honest — only claim what the candidate actually has.

Never invent experience the candidate doesn't have."""


def tailor_agent(state):
    resume_context = retrieve_resume_context(state, RETRIEVAL_QUERY, k=4)
    llm = get_llm(temperature=0.5)
    response = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=(
            f"Resume (top-k relevant sections via RAG):\n{resume_context}\n\n"
            f"Job Description:\n{state['job_description']}"
        )),
    ])
    return {"resume_tailoring": response.content}
