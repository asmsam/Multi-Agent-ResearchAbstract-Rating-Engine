import os
import urllib.parse
import xml.etree.ElementTree as ET
import concurrent.futures
from typing import TypedDict, Annotated, List, cast

import requests
import streamlit as st
from pydantic import BaseModel, Field, SecretStr

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ==========================================
# 1. SCHEMAS & STATE DEFINITIONS
# ==========================================

class SkepticEvaluation(BaseModel):
    overlap_score: int = Field(
        description="Score 0-100 indicating how similar the user's idea is to retrieved papers. 100 = identical."
    )
    draft_question: str = Field(
        description="A challenging, critical question probing potential overlaps, identical methods, or redundant hypotheses."
    )

class AdvocateEvaluation(BaseModel):
    novelty_score: int = Field(
        description="Score 0-100 indicating how unique or innovative the idea's scope, methodology, or dataset is."
    )
    draft_question: str = Field(
        description="A supportive, constructive question guiding the user to highlight unique variables or pivot away from overlap."
    )

class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    retrieved_papers: List[dict]
    overlap_score: int
    novelty_score: int
    skeptic_draft_question: str
    advocate_draft_question: str
    turn_count: int
    max_turns: int
    final_report: str
    active_speaker: str

# ==========================================
# 2. HELPER: ARXIV PAPER RETRIEVER
# ==========================================

def fetch_arxiv_papers(query: str, max_results: int = 4) -> List[dict]:
    """Fetches real papers from arXiv API based on the user's initial idea."""
    cleaned_query = urllib.parse.quote(query[:200])
    url = f"http://export.arxiv.org/api/query?search_query=all:{cleaned_query}&start=0&max_results={max_results}"
    
    papers = []
    try:
        response = requests.get(url, timeout=8)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                link_elem = entry.find("{http://www.w3.org/2005/Atom}id")

                if title_elem is None or title_elem.text is None:
                    continue
                if summary_elem is None or summary_elem.text is None:
                    summary_text = "No abstract available."
                else:
                    summary_text = summary_elem.text
                if link_elem is None or link_elem.text is None:
                    continue

                title = title_elem.text.strip().replace("\n", " ")
                summary = summary_text.strip().replace("\n", " ")
                link = link_elem.text.strip()

                papers.append({"title": title, "summary": summary[:300] + ("..." if len(summary) > 300 else ""), "link": link})
    except Exception as e:
        st.warning(f"Note: Could not query arXiv ({e}). Proceeding with internal LLM knowledge.")
    return papers

# ==========================================
# 3. LANGGRAPH NODES & CHAINS
# ==========================================

LLM_TIMEOUT_SECONDS = 30

def evaluate_node(state: ResearchState, api_key: str) -> dict:
    """Runs Skeptic and Advocate evaluations concurrently."""
    messages = state["messages"]
    papers_context = "\n".join(
        [f"- {p['title']}: {p['summary']}" for p in state.get("retrieved_papers", [])]
    )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        api_key=SecretStr(api_key),
        timeout=LLM_TIMEOUT_SECONDS,
    )

    # Skeptic Chain
    skeptic_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Skeptic Agent (Research Devil's Advocate). 
Your job is to critically compare the user's idea against prior research papers.
Retrieved Prior Art:
{papers}

Assume the idea has already been done. Output an overlap score (0-100) and a sharp, probing question targeting potential redundancies."""),
        ("placeholder", "{messages}")
    ])
    skeptic_chain = skeptic_prompt.partial(papers=papers_context) | llm.with_structured_output(SkepticEvaluation)

    # Advocate Chain
    advocate_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Advocate Agent (Suggestion/Novelty Champion).
Your job is to identify gaps in the existing literature that the user can exploit.
Retrieved Prior Art:
{papers}

Assume the idea has groundbreaking potential. Output a novelty score (0-100) and a helpful question encouraging the user to carve out a distinct research niche."""),
        ("placeholder", "{messages}")
    ])
    advocate_chain = advocate_prompt.partial(papers=papers_context) | llm.with_structured_output(AdvocateEvaluation)

    # Execute in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_s = executor.submit(skeptic_chain.invoke, {"messages": messages})
        future_a = executor.submit(advocate_chain.invoke, {"messages": messages})
        try:
            skeptic_res = cast(SkepticEvaluation, future_s.result(timeout=LLM_TIMEOUT_SECONDS + 10))
            advocate_res = cast(AdvocateEvaluation, future_a.result(timeout=LLM_TIMEOUT_SECONDS + 10))
        except concurrent.futures.TimeoutError:
            st.error("The Groq API took too long to respond. Check your API key and network connection, then try again.")
            st.stop()

    return {
        "overlap_score": skeptic_res.overlap_score,
        "skeptic_draft_question": skeptic_res.draft_question,
        "novelty_score": advocate_res.novelty_score,
        "advocate_draft_question": advocate_res.draft_question,
    }

def orchestrator_router(state: ResearchState) -> str:
    """Decision matrix routing logic based on scores and turn count."""
    if state["turn_count"] >= state["max_turns"]:
        return "synthesis_node"

    overlap = state.get("overlap_score", 0)
    novelty = state.get("novelty_score", 0)

    # Matrix Decisions
    if overlap >= 75 and novelty >= 75:
        return "emit_skeptic_node"
    elif overlap >= 75:
        return "emit_advocate_node"
    elif novelty >= 75:
        return "emit_skeptic_node"
    elif overlap < 50 and novelty < 50:
        return "emit_advocate_node"
    else:
        return "emit_skeptic_node" if overlap > novelty else "emit_advocate_node"

def emit_skeptic_node(state: ResearchState) -> dict:
    q = state["skeptic_draft_question"]
    msg = AIMessage(content=q, name="Skeptic")
    return {"messages": [msg], "active_speaker": "🔴 Skeptic Agent"}

def emit_advocate_node(state: ResearchState) -> dict:
    q = state["advocate_draft_question"]
    msg = AIMessage(content=q, name="Advocate")
    return {"messages": [msg], "active_speaker": "🟢 Advocate Agent"}

def synthesis_node(state: ResearchState, api_key: str) -> dict:
    """Compiles the final structured Markdown report."""
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        api_key=SecretStr(api_key),
        timeout=LLM_TIMEOUT_SECONDS,
    )
    papers_context = "\n".join([f"- {p['title']} ({p['link']})" for p in state.get("retrieved_papers", [])])

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are the Synthesis Agent. Compile a comprehensive Research Novelty Report.
Retrieved Reference Papers:
{papers_context}

Structure your response strictly into these Markdown sections:
# 🔬 Research Novelty & Feasibility Report

## 1. Executive Verdict & Novelty Score
Provide a final verdict (Low, Moderate, High Novelty) and a 2-sentence summary.

## 2. Literature Overlap (Points of Convergence)
Detail where the proposed idea directly mirrors prior work.

## 3. The Novelty Gap (Points of Divergence)
Detail the exact variables, methodologies, or domains where the idea breaks new ground.

## 4. Strategic Recommendations
Provide 3 concrete, actionable ways to pivot or refine the research to ensure publication value.

## 5. Key Citations & Background Reading
List the relevant papers cited in this session."""),
        ("placeholder", "{messages}")
    ])

    chain = prompt | llm
    res = chain.invoke({"messages": state["messages"]})
    report_msg = AIMessage(content=res.content, name="Synthesis")

    return {
        "final_report": res.content,
        "messages": [report_message if (report_message := report_msg) else report_msg],
        "active_speaker": "🏁 Report Complete"
    }

# ==========================================
# 4. STREAMLIT USER INTERFACE
# ==========================================

st.set_page_config(page_title="Paper Novelty Pressure-Tester", page_icon="🔬", layout="wide")

st.title("🔬 Research Paper Novelty & Overlap Checker")
st.caption("Adversarial Multi-Agent Validation Architecture (Orchestrator + Skeptic + Advocate + Synthesis)")

# Sidebar Setup
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input("Groq API Key", type="password", help="Enter your Groq API key to run Llama 3.3 70B.")
    max_turns = st.slider("Maximum Interrogation Turns", min_value=2, max_value=8, value=4)
    
    st.divider()
    st.header("📊 Live Agent Metrics")
    
    overlap_val = st.session_state.get("overlap_score", 0)
    novelty_val = st.session_state.get("novelty_score", 0)
    
    st.metric("Literature Overlap (Skeptic)", f"{overlap_val}%")
    st.progress(overlap_val / 100)
    
    st.metric("Idea Novelty (Advocate)", f"{novelty_val}%")
    st.progress(novelty_val / 100)
    
    if st.button("Reset Session", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# Check for API Key
api_key = api_key_input or os.environ.get("GROQ_API_KEY")
if not api_key:
    st.info("👈 Please enter your Groq API key in the sidebar to begin.")
    st.stop()

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "state" not in st.session_state:
    st.session_state.state = None
if "final_report" not in st.session_state:
    st.session_state.final_report = None

# Step 1: Initial Idea Submission
if not st.session_state.state:
    st.subheader("What research paper idea do you want to test?")
    user_idea = st.text_area(
        "Describe your methodology, dataset, hypothesis, or core innovation:",
        placeholder="e.g., Using Physics-Informed Neural Networks (PINNs) trained on synthetic satellite imagery to predict micro-climate shifts in tropical rainforests...",
        height=140
    )
    
    if st.button("Start Adversarial Evaluation", type="primary"):
        if not user_idea.strip():
            st.warning("Please enter a brief description of your research idea.")
        else:
            with st.status("Fetching academic papers from arXiv...", expanded=True) as status:
                papers = fetch_arxiv_papers(user_idea)
                st.write(f"Found {len(papers)} candidate papers.")
                status.update(label="Initial analysis complete!", state="complete")
            
            # Initial state setup
            initial_state: ResearchState = {
                "messages": [HumanMessage(content=user_idea)],
                "retrieved_papers": papers,
                "overlap_score": 0,
                "novelty_score": 0,
                "skeptic_draft_question": "",
                "advocate_draft_question": "",
                "turn_count": 0,
                "max_turns": max_turns,
                "final_report": "",
                "active_speaker": "System"
            }
            
            # Run initial evaluation node
            with st.spinner("Skeptic and Advocate are reviewing the literature..."):
                eval_res = evaluate_node(initial_state, api_key)
                initial_state.update(eval_res)
                
                # Determine initial speaker
                next_node = orchestrator_router(initial_state)
                if next_node == "emit_skeptic_node":
                    emit_res = emit_skeptic_node(initial_state)
                else:
                    emit_res = emit_advocate_node(initial_state)
                
                initial_state.update(emit_res)
                initial_state["messages"].append(emit_res["messages"][0])

            st.session_state.state = initial_state
            st.session_state.overlap_score = initial_state["overlap_score"]
            st.session_state.novelty_score = initial_state["novelty_score"]
            st.rerun()

# Step 2: Interactive Interrogation & Report Display
else:
    state = st.session_state.state

    # Display Retrieved References
    with st.expander("📚 Retrieved Reference Papers (Context Window)", expanded=False):
        for p in state.get("retrieved_papers", []):
            st.markdown(f"**[{p['title']}]({p['link']})**")
            st.caption(p["summary"])

    # Render Chat Log
    st.subheader("Adversarial Debate Log")
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            speaker_icon = "🔴" if msg.name == "Skeptic" else "🟢" if msg.name == "Advocate" else "🏁"
            with st.chat_message("assistant", avatar=speaker_icon):
                st.markdown(f"**{msg.name or 'Agent'}:** {msg.content}")

    # Render Final Report if Complete
    if state.get("final_report"):
        st.divider()
        st.markdown(state["final_report"])
        
        # Download button for report
        st.download_button(
            label="📄 Download Report as Markdown",
            data=state["final_report"],
            file_name="research_novelty_report.md",
            mime="text/markdown"
        )
    else:
        # User input box for answering agent questions
        if prompt := st.chat_input("Answer the agent's question to clarify your paper idea..."):
            # Append user response
            state["messages"].append(HumanMessage(content=prompt))
            state["turn_count"] += 1
            
            # Re-evaluate with new context
            with st.spinner("Agents are analyzing your response..."):
                eval_res = evaluate_node(state, api_key)
                state.update(eval_res)
                
                # Dynamic Routing
                next_node = orchestrator_router(state)
                if next_node == "synthesis_node":
                    synth_res = synthesis_node(state, api_key)
                    state.update(synth_res)
                elif next_node == "emit_skeptic_node":
                    emit_res = emit_skeptic_node(state)
                    state.update(emit_res)
                else:
                    emit_res = emit_advocate_node(state)
                    state.update(emit_res)
                
                # Update Streamlit sidebar scores
                st.session_state.overlap_score = state["overlap_score"]
                st.session_state.novelty_score = state["novelty_score"]
                st.session_state.state = state
                st.rerun()