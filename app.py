import streamlit as st
from groq import Groq
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import gspread
from google.oauth2.service_account import Credentials
import requests
import json
from datetime import datetime
from typing import TypedDict, Optional

st.set_page_config(page_title="AI SDR Agent", page_icon="🤖", layout="centered")

st.markdown("""
<style>
    .hero {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 40px 24px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 32px;
    }
    .hero h1 { color: white; font-size: 2em; font-weight: 800; }
    .hero p { color: rgba(255,255,255,0.8); font-size: 1em; }
    .node-badge {
        display: inline-block;
        background: rgba(255,255,255,0.1);
        color: white;
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 0.8em;
        margin: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🤖 AI SDR Agent</h1>
    <p>Multi-agent system: Research → Outreach → Follow-up → CRM</p>
    <div>
        <span class="node-badge">🔍 Research Node</span>
        <span class="node-badge">✍️ Outreach Node</span>
        <span class="node-badge">📧 Follow-up Node</span>
        <span class="node-badge">💾 CRM Node</span>
    </div>
</div>
""", unsafe_allow_html=True)

# State
class SDRState(TypedDict):
    prospect_name: str
    job_title: str
    company: str
    stage: str
    pain_point: str
    offer: str
    benefit: str
    tone: str
    serper_key: str
    groq_key: str
    research: Optional[str]
    connection_request: Optional[str]
    followup: Optional[str]
    cold_email: Optional[str]
    saved: Optional[bool]

# Nodes
def research_node(state: SDRState) -> SDRState:
    query = f"{state['company']} {state['job_title']} {state['prospect_name']} startup news"
    headers = {"X-API-KEY": state["serper_key"], "Content-Type": "application/json"}
    data = {"q": query, "num": 3}
    try:
        res = requests.post("https://google.serper.dev/search", headers=headers, json=data)
        results = res.json().get("organic", [])
        snippets = "\n".join([r.get("snippet", "") for r in results[:3]])
        state["research"] = snippets if snippets else "No specific research found."
    except:
        state["research"] = "Research unavailable."
    return state

def outreach_node(state: SDRState) -> SDRState:
    llm = ChatGroq(api_key=state["groq_key"], model="llama-3.3-70b-versatile")
    prompt = f"""You are an elite B2B outreach copywriter.

RESEARCH CONTEXT:
{state['research']}

PROSPECT: {state['prospect_name']}, {state['job_title']} at {state['company']} ({state['stage']})
PAIN POINT: {state['pain_point']}
OFFER: {state['offer']}
BENEFIT: {state['benefit']}
TONE: {state['tone']}

RULES:
- Use research context to personalize
- NEVER use "hope you're doing well" or "love what you're building"
- Connection request: max 50 words, ends with soft question
- Sound like a founder, not a marketer
- Short sentences only

OUTPUT:
CONNECTION REQUEST:
[max 50 words]

COLD EMAIL:
[subject line, then 4-5 short lines]"""

    response = llm.invoke([HumanMessage(content=prompt)])
    result = response.content

    if "CONNECTION REQUEST:" in result:
        conn = result.split("CONNECTION REQUEST:")[1]
        if "COLD EMAIL:" in conn:
            state["connection_request"] = conn.split("COLD EMAIL:")[0].strip()
            state["cold_email"] = result.split("COLD EMAIL:")[1].strip()
        else:
            state["connection_request"] = conn.strip()
    return state

def followup_node(state: SDRState) -> SDRState:
    llm = ChatGroq(api_key=state["groq_key"], model="llama-3.3-70b-versatile")
    prompt = f"""Write a follow-up LinkedIn DM based on this context:

Prospect: {state['prospect_name']} at {state['company']}
Original offer: {state['offer']}
Benefit: {state['benefit']}

RULES:
- 3-4 lines max
- Lead with value, not pitch
- One soft CTA at end
- Sound human

Write only the message, no labels."""

    response = llm.invoke([HumanMessage(content=prompt)])
    state["followup"] = response.content.strip()
    return state

def crm_node(state: SDRState) -> SDRState:
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(st.secrets["SHEET_ID"])
        try:
            ws = sh.worksheet("SDR CRM")
        except:
            ws = sh.add_worksheet(title="SDR CRM", rows="1000", cols="20")
            ws.append_row(["Date", "Name", "Title", "Company", "Stage", "Pain Point", "Research", "Connection Request", "Follow-up", "Cold Email"])
        ws.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            state["prospect_name"], state["job_title"], state["company"],
            state["stage"], state["pain_point"],
            state.get("research", ""),
            state.get("connection_request", ""),
            state.get("followup", ""),
            state.get("cold_email", "")
        ])
        state["saved"] = True
    except Exception as e:
        state["saved"] = False
    return state

# Build Graph
def build_graph():
    graph = StateGraph(SDRState)
    graph.add_node("research", research_node)
    graph.add_node("outreach", outreach_node)
    graph.add_node("followup", followup_node)
    graph.add_node("crm", crm_node)
    graph.set_entry_point("research")
    graph.add_edge("research", "outreach")
    graph.add_edge("outreach", "followup")
    graph.add_edge("followup", "crm")
    graph.add_edge("crm", END)
    return graph.compile()

# UI
st.subheader("🎯 Prospect Info")
prospect_name = st.text_input("Prospect's Name", placeholder="e.g. Rahul Sharma")
job_title = st.text_input("Job Title", placeholder="e.g. Founder & CEO")
company = st.text_input("Company", placeholder="e.g. Razorpay")
stage = st.selectbox("Stage", ["Early-stage Startup", "Series A SaaS", "Growth-stage SaaS", "Bootstrapped SaaS"])
pain_point = st.text_input("Pain Point", placeholder="e.g. scaling outbound sales")

st.subheader("💼 Your Offer")
offer = st.text_input("What You're Offering", placeholder="e.g. AI agent that automates LinkedIn outreach")
benefit = st.text_input("Key Benefit", placeholder="e.g. saves 5 hours/week")
tone = st.radio("Tone", ["Professional", "Casual", "Founder-style", "Direct"], horizontal=True)

if st.button("🚀 Run AI SDR Agent", use_container_width=True):
    if not prospect_name or not company or not offer:
        st.error("Please fill Name, Company and Offer.")
    else:
        graph = build_graph()

        col1, col2, col3, col4 = st.columns(4)
        with col1: research_status = st.empty()
        with col2: outreach_status = st.empty()
        with col3: followup_status = st.empty()
        with col4: crm_status = st.empty()

        research_status.info("🔍 Researching...")

        state = SDRState(
            prospect_name=prospect_name,
            job_title=job_title,
            company=company,
            stage=stage,
            pain_point=pain_point,
            offer=offer,
            benefit=benefit,
            tone=tone,
            serper_key=st.secrets["SERPER_API_KEY"],
            groq_key=st.secrets["GROQ_API_KEY"],
            research=None,
            connection_request=None,
            followup=None,
            cold_email=None,
            saved=None
        )

        result = graph.invoke(state)

        research_status.success("🔍 Done")
        outreach_status.success("✍️ Done")
        followup_status.success("📧 Done")
        crm_status.success("💾 Saved")

        st.markdown("---")

        if result.get("research"):
            with st.expander("🔍 Research Found"):
                st.write(result["research"])

        if result.get("connection_request"):
            st.subheader("📨 Connection Request")
            st.code(result["connection_request"], language=None)

        if result.get("followup"):
            st.subheader("💬 Follow-up Message")
            st.code(result["followup"], language=None)

        if result.get("cold_email"):
            st.subheader("📧 Cold Email")
            st.code(result["cold_email"], language=None)

        if result.get("saved"):
            st.success("✅ Saved to Google Sheets CRM!")
            # Analytics Dashboard
st.markdown("---")
st.subheader("📊 Analytics Dashboard")

try:
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(st.secrets["SHEET_ID"])
    ws = sh.worksheet("SDR CRM")
    all_values = ws.get_all_values()
    
    if len(all_values) > 1:
        headers = all_values[0]
        rows = all_values[1:]
        
        reply_idx = headers.index("Reply") if "Reply" in headers else -1
        name_idx = headers.index("Name") if "Name" in headers else 1
        company_idx = headers.index("Company") if "Company" in headers else 3
        date_idx = headers.index("Date") if "Date" in headers else 0
        
        total = len(rows)
        replies = sum(1 for row in rows if reply_idx >= 0 and len(row) > reply_idx and row[reply_idx].strip().lower() == "yes")
        reply_rate = round((replies / total * 100), 1) if total > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📨 Outreach Generated", total)
        with col2:
            st.metric("↩️ Replies", replies)
        with col3:
            st.metric("📈 Reply Rate", f"{reply_rate}%")

        st.subheader("👥 Recent Prospects")
        for row in list(reversed(rows))[:5]:
            name = row[name_idx] if len(row) > name_idx else ""
            company = row[company_idx] if len(row) > company_idx else ""
            date = row[date_idx] if len(row) > date_idx else ""
            st.write(f"👤 **{name}** — {company} | {date}")
    else:
        st.info("No data yet.")

except Exception as e:
    st.error(f"Error: {e}")

st.markdown("---")
st.caption("Built by Pankaj Singh · AI SDR Agent · LangGraph + Groq + Streamlit")
