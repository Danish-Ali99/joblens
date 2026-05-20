"""Streamlit UI for JobLens — AI Job Intelligence Platform.

Heavy imports (langgraph, langchain, langchain_groq, rank_bm25, pypdf) are
deferred — they happen only when the user actually uploads a file or clicks
Analyze. This keeps the initial page render snappy on HF Spaces cold start.
"""
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# --- Lazy-loaded helpers (imports happen on first call, cached after) ---

def _extract_resume(uploaded_file) -> str:
    from utils.pdf import extract_resume_text
    return extract_resume_text(uploaded_file)


def _build_retriever(text: str):
    from utils.retriever import ResumeRetriever
    return ResumeRetriever(text)


REPO_URL = "https://github.com/Danish-Ali99/joblens"

SAMPLE_JDS = {
    "AI Engineer": """About the Role
We're hiring an AI Engineer to build LLM-powered products. You will own the
end-to-end design and shipping of retrieval-augmented agents.

Requirements
- 2+ years of Python production experience
- Hands-on with LangChain, LangGraph, or comparable orchestration frameworks
- Built and deployed at least one RAG pipeline (chunking, embeddings, vector DB)
- Comfortable with OpenAI / Anthropic / Groq APIs and prompt engineering
- Experience with agentic patterns (tool use, multi-agent, fan-out/fan-in)
- Familiarity with one of: pgvector, FAISS, Pinecone, Weaviate

Nice to have
- Streamlit or FastAPI experience
- Prior exposure to evaluation frameworks (LangSmith, Ragas)
- Open-source contributions to AI tooling""",

    "Full-Stack Developer": """About the Role
We need a full-stack engineer to build customer-facing web products at a
fast-moving startup.

Requirements
- 2+ years building production React or Next.js apps
- Strong TypeScript / JavaScript
- Comfortable with Node.js backends, REST APIs, and Postgres
- Experience with Tailwind CSS and modern UI patterns
- Familiar with Vercel, Netlify, or similar deploy platforms

Nice to have
- Worked at an early-stage startup
- Some Python / FastAPI exposure
- AI/LLM integration experience (Claude / OpenAI APIs)""",

    "Business Development Intern": """About the Role
We're an early-stage AI startup hiring a Business Development Intern who can
drive outbound outreach and close pilot deals.

Requirements
- Strong written and verbal English communication
- Comfortable with CRMs (HubSpot, Salesforce) and outbound tools
- Self-starter — can work without close supervision
- Final-year undergrad or recent grad
- Interest in AI / SaaS market

Nice to have
- Prior internship in sales, BD, or growth
- Personal projects or side hustles that show ownership
- Familiarity with AI tools and how they're sold""",
}


def get_secret(name: str, default: str = "") -> str:
    if os.environ.get(name):
        return os.environ[name]
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return st.secrets[name]
    except (FileNotFoundError, AttributeError):
        pass
    return default


@st.cache_resource
def get_compiled_graph():
    from graph.workflow import build_graph
    return build_graph()


def _set_jd(jd_text: str) -> None:
    st.session_state["jd_text"] = jd_text


st.set_page_config(
    page_title="JobLens — AI Job Intelligence",
    page_icon="🔍",
    layout="wide",
)

st.session_state.setdefault("jd_text", "")
st.session_state.setdefault("resume_text", "")

st.title("🔍 JobLens — AI Job Intelligence")
st.caption(
    "Upload your resume + a job description. 5 specialist AI agents review them in parallel "
    "using RAG over chunked resume sections, then a Synthesizer compiles a 24-hour action plan. "
    "Built with LangGraph + LangChain + BM25 retrieval."
)

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")

    provider = st.selectbox(
        "Provider",
        ["Groq (free)", "OpenAI"],
        index=0,
        help="Groq is free, no credit card. OpenAI requires billing.",
    )

    if provider == "Groq (free)":
        api_key = st.text_input(
            "Groq API Key",
            value=get_secret("GROQ_API_KEY"),
            type="password",
            help="Get a free key at console.groq.com — no credit card.",
        )
        model = st.selectbox(
            "Model",
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it"],
            index=0,
            help="8b-instant: fastest. 70b-versatile: higher quality but slower.",
        )
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            value=get_secret("OPENAI_API_KEY"),
            type="password",
        )
        model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
            index=0,
        )

    st.markdown("---")
    st.markdown("**Pipeline**")
    st.code(
        "Resume + JD\n    ↓\n┌───────────┐\n│ 5 agents  │\n│ parallel  │\n└─────┬─────┘\n      ↓\n Synthesizer\n      ↓\n Action Plan",
        language="text",
    )
    st.markdown("---")
    st.markdown(f"[View source on GitHub]({REPO_URL})")

# --- Main: inputs ---
col_resume, col_jd = st.columns([1, 1])

with col_resume:
    st.subheader("1. Your resume")
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Your resume in PDF format. Stays in this session, never stored.",
    )
    if uploaded_file:
        try:
            resume_text = _extract_resume(uploaded_file)
            st.session_state["resume_text"] = resume_text
            st.success(f"Extracted {len(resume_text)} characters from {uploaded_file.name}")
            with st.expander("Preview extracted text"):
                st.text(resume_text[:1500] + ("..." if len(resume_text) > 1500 else ""))
        except Exception as e:
            st.error(f"Couldn't parse PDF: {e}")

with col_jd:
    st.subheader("2. Job description")
    st.markdown("**Try a sample:**")
    btn_cols = st.columns(len(SAMPLE_JDS))
    for col, (label, jd) in zip(btn_cols, SAMPLE_JDS.items()):
        with col:
            st.button(
                label,
                key=f"sample-{label}",
                on_click=_set_jd,
                args=(jd,),
                use_container_width=True,
            )
    jd_text = st.text_area(
        "Paste the JD",
        height=260,
        placeholder="Paste the full job description here...",
        key="jd_text",
    )

run = st.button("🔍 Analyze fit", type="primary")

if run:
    if not api_key:
        st.error(f"Please provide your {provider.split()[0]} API key in the sidebar.")
        st.stop()
    resume_text = (st.session_state.get("resume_text") or "").strip()
    jd = (jd_text or "").strip()
    if not resume_text:
        st.error("Please upload your resume PDF in the left column.")
        st.stop()
    if len(jd) < 80:
        st.warning("⚠️ The job description seems too short. Paste the full JD for a good analysis.")
        st.stop()

    if provider == "Groq (free)":
        os.environ["LLM_PROVIDER"] = "groq"
        os.environ["GROQ_API_KEY"] = api_key
        os.environ["GROQ_MODEL"] = model
    else:
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_MODEL"] = model

    with st.spinner("Building RAG index for your resume..."):
        try:
            retriever = _build_retriever(resume_text)
            st.caption(
                f"📚 RAG index built — {retriever.num_chunks()} resume chunks indexed "
                "with BM25. Each agent retrieves only the top-k relevant chunks for its role."
            )
        except Exception as e:
            st.warning(
                f"RAG index couldn't build ({e}); falling back to full-resume context."
            )
            retriever = None

    initial = {
        "resume_text": resume_text,
        "job_description": jd,
        "retriever": retriever,
        "match_analysis": None,
        "skills_gap": None,
        "resume_tailoring": None,
        "interview_questions": None,
        "cover_letter": None,
        "action_plan": None,
    }

    app = get_compiled_graph()

    labels = {
        "matcher": "Match Analyst",
        "gap": "Skills Gap",
        "tailor": "Resume Tailor",
        "interviewer": "Interview Coach",
        "cover_letter": "Cover Letter",
        "synthesizer": "Synthesizer",
    }

    progress = st.progress(0, text="Starting agents...")
    completed_state = initial.copy()
    step_count = 0
    total_steps = 6

    try:
        for event in app.stream(initial, stream_mode="updates"):
            for node_name, update in event.items():
                step_count += 1
                progress.progress(
                    min(step_count / total_steps, 1.0),
                    text=f"Completed: {labels.get(node_name, node_name)}",
                )
                completed_state.update(update or {})
    except Exception as e:
        msg = str(e).lower()
        if any(s in msg for s in ("api key", "invalid_api_key", "unauthorized", "401", "authentication")):
            st.error(
                "API key was rejected. Double-check your key in the sidebar — "
                "Groq keys start with `gsk_`, OpenAI keys with `sk-`."
            )
        elif "rate" in msg and "limit" in msg:
            st.error(
                "Rate limit hit. Wait a minute, then retry. Switching to a smaller model "
                "in the sidebar (e.g. `llama-3.1-8b-instant`) raises your limits."
            )
        elif any(s in msg for s in ("connection", "timeout", "network")):
            st.error("Network issue reaching the model provider. Check your internet and retry.")
        else:
            st.error(f"Agent run failed: {e}")
        st.stop()

    progress.progress(1.0, text="All agents complete")
    st.success("Analysis complete.")

    # --- Action plan (hero) ---
    st.markdown("---")
    st.markdown(completed_state.get("action_plan") or "_No action plan._")

    fname_base = f"joblens-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    st.download_button(
        "📥 Download full action plan",
        completed_state.get("action_plan") or "",
        file_name=f"{fname_base}-action-plan.md",
        mime="text/markdown",
    )

    # --- Individual agent outputs ---
    st.markdown("---")
    st.markdown("## Specialist Reports")
    tab_match, tab_gap, tab_tailor, tab_int, tab_cover = st.tabs([
        "🎯 Match", "🔍 Skills Gap", "✏️ Resume Tailor", "🎤 Interview Prep", "📝 Cover Letter"
    ])
    with tab_match:
        st.markdown(completed_state.get("match_analysis") or "_empty_")
    with tab_gap:
        st.markdown(completed_state.get("skills_gap") or "_empty_")
    with tab_tailor:
        st.markdown(completed_state.get("resume_tailoring") or "_empty_")
    with tab_int:
        st.markdown(completed_state.get("interview_questions") or "_empty_")
    with tab_cover:
        st.markdown(completed_state.get("cover_letter") or "_empty_")
        st.download_button(
            "📥 Download cover letter (.md)",
            completed_state.get("cover_letter") or "",
            file_name=f"{fname_base}-cover-letter.md",
            mime="text/markdown",
            key="dl-cover",
        )

st.markdown("---")
st.caption(
    "⚠️ AI-generated career advice is a starting point, not a guarantee. Use your judgment. "
    f"Source on [GitHub]({REPO_URL})."
)
