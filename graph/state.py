"""Shared state passed between agents in the JobLens workflow."""
from typing import TypedDict, Optional


class JobLensState(TypedDict):
    resume_text: str
    job_description: str
    match_analysis: Optional[str]
    skills_gap: Optional[str]
    resume_tailoring: Optional[str]
    interview_questions: Optional[str]
    cover_letter: Optional[str]
    action_plan: Optional[str]
