"""Shared state passed between agents in the JobLens workflow."""
from typing import TypedDict, Optional, Any


class JobLensState(TypedDict):
    resume_text: str
    job_description: str
    retriever: Any  # utils.retriever.ResumeRetriever — set once at run start
    match_analysis: Optional[str]
    skills_gap: Optional[str]
    resume_tailoring: Optional[str]
    interview_questions: Optional[str]
    cover_letter: Optional[str]
    action_plan: Optional[str]
