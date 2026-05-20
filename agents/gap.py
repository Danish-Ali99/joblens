"""Skills Gap Analyst — pinpoints missing skills and ranks by JD priority."""
from langchain_core.messages import SystemMessage, HumanMessage
from .base import get_llm, retrieve_resume_context

RETRIEVAL_QUERY = (
    "skills technologies tools certifications education stack frameworks "
    "languages programming"
)


SYSTEM = """You are a skills-gap analyst. Compare the candidate's resume against
the job description and identify what's MISSING.

Produce EXACTLY these markdown sections:

## Hard Gaps (Must-Have Skills Missing)
The skills/tools/experiences explicitly required by the JD that don't appear
in the resume. For each, note WHY it matters for this role. Rank by importance.

## Soft Gaps (Nice-to-Haves Missing)
Skills mentioned as "preferred" or "bonus" in the JD that aren't on the resume.

## Adjacent Skills the Candidate Has
Things on the resume that are CLOSE to what the JD wants but not exact —
the candidate could position these as a partial match in an interview.

## Closing the Gap in 30 Days
3-5 specific, concrete things the candidate could do in the next 30 days
to close the hard gaps (e.g., "Build a project using X", "Take Y course",
"Contribute to Z OSS repo"). Make them actionable, not vague.

Be specific about exact technologies and concepts. No generic advice."""


def gap_agent(state):
    resume_context = retrieve_resume_context(state, RETRIEVAL_QUERY, k=3)
    llm = get_llm(temperature=0.5)
    response = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=(
            f"Resume (top-k relevant sections via RAG):\n{resume_context}\n\n"
            f"Job Description:\n{state['job_description']}"
        )),
    ])
    return {"skills_gap": response.content}
