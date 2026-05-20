"""LangGraph workflow: 5 specialists in parallel -> Synthesizer.

All five specialist agents (matcher, gap, tailor, interviewer, cover_letter)
read from the same inputs (resume + JD) and write to their own state fields
in parallel. LangGraph schedules them concurrently and only triggers the
synthesizer once all five have completed.
"""
from langgraph.graph import StateGraph, END

from agents import (
    matcher_agent,
    gap_agent,
    tailor_agent,
    interviewer_agent,
    cover_letter_agent,
    synthesizer_agent,
)
from .state import JobLensState


def build_graph():
    """Build and compile the JobLens LangGraph workflow."""
    graph = StateGraph(JobLensState)

    graph.add_node("matcher", matcher_agent)
    graph.add_node("gap", gap_agent)
    graph.add_node("tailor", tailor_agent)
    graph.add_node("interviewer", interviewer_agent)
    graph.add_node("cover_letter", cover_letter_agent)
    graph.add_node("synthesizer", synthesizer_agent)

    # Fan out: from START to all 5 specialists in parallel
    graph.add_edge("__start__", "matcher")
    graph.add_edge("__start__", "gap")
    graph.add_edge("__start__", "tailor")
    graph.add_edge("__start__", "interviewer")
    graph.add_edge("__start__", "cover_letter")

    # Fan in: synthesizer runs only after all 5 complete
    graph.add_edge("matcher", "synthesizer")
    graph.add_edge("gap", "synthesizer")
    graph.add_edge("tailor", "synthesizer")
    graph.add_edge("interviewer", "synthesizer")
    graph.add_edge("cover_letter", "synthesizer")

    graph.add_edge("synthesizer", END)

    return graph.compile()
