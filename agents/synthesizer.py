"""Synthesizer — compiles all agent outputs into a 24-hour action plan."""
from langchain_core.messages import SystemMessage, HumanMessage
from .base import get_llm


SYSTEM = """You are a career strategist. You've received analyses from 5
specialists who reviewed a candidate's resume against a target job. Now
compile their outputs into a single, action-oriented brief the candidate
can act on TODAY.

Structure your output as:

# Your JobLens Action Plan

## Bottom Line
2-3 sentences. Should this candidate apply for this role? What's the realistic
outcome (likely interview / borderline / long shot)? Be honest.

## What to Do in the Next 24 Hours
A numbered, prioritized checklist. Mix high-effort and quick wins.
For each item, ONE sentence on what to do and why.

Examples of items that might appear:
1. Add these 3 keywords to your resume summary line: ...
2. Rewrite this bullet on your resume: ...
3. Prepare answers to these 3 likely questions: ...
4. Practice this project deep-dive: ...
5. Customize the cover letter draft below: ...

## Strategic Take
2-3 sentences of higher-level advice: should the candidate keep applying to
roles like this one, or look at a different level / domain?

Be honest. The candidate has paid you for honest counsel, not flattery."""


def synthesizer_agent(state):
    llm = get_llm(temperature=0.4)
    context = (
        f"Resume:\n{state['resume_text']}\n\n"
        f"Job Description:\n{state['job_description']}\n\n"
        "---\n\n"
        f"Match Analyst output:\n{state.get('match_analysis', 'N/A')}\n\n"
        f"Skills Gap Analyst output:\n{state.get('skills_gap', 'N/A')}\n\n"
        f"Resume Tailor output:\n{state.get('resume_tailoring', 'N/A')}\n\n"
        f"Interview Coach output:\n{state.get('interview_questions', 'N/A')}\n\n"
        f"Cover Letter Writer output:\n{state.get('cover_letter', 'N/A')}\n"
    )
    response = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=context),
    ])
    return {"action_plan": response.content}
