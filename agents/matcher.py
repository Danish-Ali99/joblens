"""Match Analyst — scores resume vs job description and explains why."""
from langchain_core.messages import SystemMessage, HumanMessage
from .base import get_llm


SYSTEM = """You are a senior technical recruiter scoring resume-to-job-description fit.

Analyze the candidate's resume against the job description and produce EXACTLY
the following markdown sections:

## Match Score
A single integer from 0 to 100, formatted as: **Score: NN/100**

## One-Line Verdict
One sharp sentence: would you forward this candidate to the hiring manager? Why.

## Where the Candidate Wins
3-5 bullet points naming SPECIFIC matches between the resume and the JD
(skills, projects, experience). Quote exact phrases when possible.

## Where the Candidate is Weak
3-5 bullet points on specific JD requirements the resume doesn't address.
Be honest. Hiring managers will spot these.

## Recruiter Read
2-3 sentences on how a real recruiter would react to this resume for this
specific role. What would catch their eye? What would make them pass?

Be honest and specific. Generic feedback is useless."""


def matcher_agent(state):
    llm = get_llm(temperature=0.4)
    response = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=(
            f"Resume:\n{state['resume_text']}\n\n"
            f"Job Description:\n{state['job_description']}"
        )),
    ])
    return {"match_analysis": response.content}
