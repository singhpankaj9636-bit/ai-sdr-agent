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
    .stApp {
        background: linear-gradient(180deg, #0B1120 0%, #0F1729 100%);
        color: #E2E8F0;
    }
    .hero {
        background: linear-gradient(135deg, #0f0c29, #1E3A8A, #0B1120);
        padding: 40px 24px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 32px;
        border: 1px solid #1E293B;
    }
    .hero h1 { color: white; font-size: 2em; font-weight: 800; }
    .hero p { color: rgba(255,255,255,0.8); font-size: 1em; }
    .node-badge {
        display: inline-block;
        background: rgba(56,189,248,0.15);
        color: #38BDF8;
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 0.8em;
        margin: 4px;
        border: 1px solid rgba(56,189,248,0.3);
    }
    h1, h2, h3 { color: #F8FAFC !important; font-weight: 700 !important; }
    [data-testid="stMetric"] {
        background: #111C33;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    [data-testid="stMetricLabel"] { color: #94A3B8 !important; }
    [data-testid="stMetricValue"] { color: #38BDF8 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #2563EB, #1D4ED8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8, #1E40AF);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37,99,235,0.4);
    }
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        background-color: #111C33 !important;
        color: #E2E8F0 !important;
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
    }
    .lead-card {
        background: #111C33;
        border: 1px solid #1E293B;
        border-left: 4px solid #38BDF8;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .lead-card h4 { margin: 0 0 6px 0; color: #F8FAFC; }
    .lead-card p { margin: 2px 0; color: #94A3B8; font-size: 0.9rem; }
    .score-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .score-high { background: #064E3B; color: #34D399; }
    .score-med  { background: #78350F; color: #FBBF24; }
    .score-low  { background: #450A0A; color: #F87171; }
    [data-testid="stDataFrame"] { background: #111C33; border-radius: 10px; }
    hr { border-color: #1E293B !important; }
</style>
""", unsafe_allow_html=True)


def render_lead_card(name, company, score, category, reason=""):
    if category == "Hot":
        badge_class = "score-high"
    elif category == "Warm":
        badge_class = "score-med"
    else:
        badge_class = "score-low"

    st.markdown(f"""
    <div class="lead-card">
        <h4>{name} <span class="score-badge {badge_class}">{category} · {score}</span></h4>
        <p>🏢 {company}</p>
        <p>{reason}</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h1>🤖 AI SDR Agent</h1>
    <p>Multi-agent system: Research → Outreach → Follow-up → Scoring → CRM</p>
    <div>
        <span class="node-badge">🔍 Research Node</span>
        <span class="node-badge">✍️ Outreach Node</span>
        <span class="node-badge">📧 Follow-up Node</span>
        <span class="node-badge">🎯 Scoring Node</span>
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
    research_data: Optional[dict]
    connection_request: Optional[str]
    followup: Optional[str]
    cold_email: Optional[str]
    saved: Optional[bool]
    lead_score: Optional[int]
    lead_category: Optional[str]
    score_reason: Optional[str]
    lead_status: Optional[str]

# Nodes
def research_node(state: SDRState) -> SDRState:
    headers = {"X-API-KEY": state["serper_key"], "Content-Type": "application/json"}

    name = state["prospect_name"]
    company = state["company"]
    company_domain_guess = company.lower().replace(" ", "").replace(",", "").replace(".", "")

    # Targeted queries instead of loose OR queries
    queries = [
        f'site:linkedin.com/in "{name}" "{company}"',
        f'site:linkedin.com/company "{company}"',
        f'"{company}" funding OR raised OR "series a" OR "series b"',
        f'"{company}" hiring OR "we are hiring" OR "join our team"',
        f'"{company}" launch OR announces OR announcement',
        f'"{company}" official site',
    ]

    all_results = []
    seen_links = set()

    for q in queries:
        try:
            data = {"q": q, "num": 5}
            res = requests.post("https://google.serper.dev/search", headers=headers, json=data, timeout=10)
            organic = res.json().get("organic", [])
            for r in organic[:5]:
                link = r.get("link", "")
                if not link or link in seen_links:
                    continue
                seen_links.add(link)
                all_results.append({
                    "title": r.get("title", ""),
                    "link": link,
                    "snippet": r.get("snippet", "")
                })
        except Exception:
            pass

    # Source ranking: LinkedIn > Company site > News > Others
    def rank(r):
        link = r["link"].lower()
        if "linkedin.com/in" in link:
            return 0
        if "linkedin.com/company" in link:
            return 1
        if company_domain_guess in link:
            return 2
        if any(word in link for word in ["news", "techcrunch", "businesswire", "prnewswire", "forbes", "reuters"]):
            return 3
        return 4

    all_results.sort(key=rank)

    # Identity filter: discard /in/ profiles that don't mention the company
    filtered = []
    for r in all_results[:10]:
        if "linkedin.com/in" in r["link"].lower():
            text = (r["title"] + " " + r["snippet"]).lower()
            if company.lower() not in text:
                continue
        filtered.append(r)

    verified_profile_found = any("linkedin.com/in" in r["link"].lower() for r in filtered)
    company_site_found = any(company_domain_guess in r["link"].lower() for r in filtered)

    if not filtered:
        state["research"] = "No verified results found."
        state["research_data"] = {
            "person": name,
            "company": company,
            "recent_news": [],
            "hiring": False,
            "funding": False,
            "sources": [],
            "confidence": "Low"
        }
        return state

    raw_block = "\n\n".join(
        f"[SOURCE: {r['link']}]\nTITLE: {r['title']}\nSNIPPET: {r['snippet']}"
        for r in filtered[:8]
    )
    state["research"] = raw_block  # kept for the "Research Found" expander in UI

    # Structured extraction step: LLM converts raw results into strict JSON
    llm = ChatGroq(api_key=state["groq_key"], model="llama-3.3-70b-versatile")
    extraction_prompt = f"""Extract structured facts about this prospect from the search results below.
Return ONLY valid JSON, no markdown fences, no preamble, no explanation.

PROSPECT NAME: {name}
COMPANY: {company}

SEARCH RESULTS:
{raw_block}

Return JSON in exactly this shape:
{{
  "person": "{name}",
  "company": "{company}",
  "recent_news": ["short factual bullet", "..."],
  "hiring": true or false,
  "funding": true or false,
  "sources": ["url1", "url2"],
  "confidence": "High" or "Medium" or "Low"
}}

RULES:
- Only include facts explicitly stated in the search results above.
- If a result seems to refer to a different person or a different company, ignore it completely.
- "recent_news" should have at most 3 short bullets, or an empty list if nothing verified.
- "hiring" is true only if a source explicitly mentions hiring/open roles for this company.
- "funding" is true only if a source explicitly mentions a funding round or raise for this company.
- confidence = "High" if a LinkedIn profile AND company site/news are both verified, "Medium" if only one, "Low" if neither.
- If nothing can be verified, keep recent_news empty and confidence "Low". Do not guess or invent facts."""

    try:
        response = llm.invoke([HumanMessage(content=extraction_prompt)])
        raw_json = response.content.strip()
        if raw_json.startswith("```"):
            raw_json = raw_json.strip("`")
            if raw_json.lower().startswith("json"):
                raw_json = raw_json[4:].strip()
        structured = json.loads(raw_json)
    except Exception:
        structured = {
            "person": name,
            "company": company,
            "recent_news": [],
            "hiring": False,
            "funding": False,
            "sources": [r["link"] for r in filtered[:5]],
            "confidence": "Medium" if (verified_profile_found or company_site_found) else "Low"
        }

    state["research_data"] = structured
    return state


def outreach_node(state: SDRState) -> SDRState:
    llm = ChatGroq(api_key=state["groq_key"], model="llama-3.3-70b-versatile")
    rd = state.get("research_data", {}) or {}

    news_lines = "\n".join(f"- {n}" for n in rd.get("recent_news", [])) or "None verified"

    prompt = f"""You are an elite B2B outreach copywriter.

VERIFIED PROSPECT FACTS (confidence: {rd.get('confidence', 'Low')}):
- Hiring: {rd.get('hiring', False)}
- Funding: {rd.get('funding', False)}
- Recent news:
{news_lines}

PROSPECT: {state['prospect_name']}, {state['job_title']} at {state['company']} ({state['stage']})
PAIN POINT: {state['pain_point']}
OFFER: {state['offer']}
BENEFIT: {state['benefit']}
TONE: {state['tone']}

RULES:
- Only reference verified facts above. If confidence is "Low" or a fact is False/empty, do NOT mention it — personalize using role/company/stage/pain point instead.
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


def scoring_node(state: SDRState) -> SDRState:
    rd = state.get("research_data", {}) or {}
    score = 30
    stage = state.get("stage", "")

    if stage == "Growth-stage SaaS":
        score += 20
    elif stage == "Series A SaaS":
        score += 15
    elif stage == "Bootstrapped SaaS":
        score += 5
    elif stage == "Early-stage Startup":
        score += 0

    if rd.get("funding"):
        score += 20
    if rd.get("hiring"):
        score += 15
    if rd.get("recent_news"):
        score += 10
    if state.get("pain_point") and len(state["pain_point"].strip()) > 5:
        score += 10

    # Don't let unverified data inflate the score too much
    if rd.get("confidence") == "Low":
        score = min(score, 60)

    score = max(0, min(100, score))

    if score >= 70:
        category = "Hot"
    elif score >= 45:
        category = "Warm"
    else:
        category = "Cold"

    llm = ChatGroq(api_key=state["groq_key"], model="llama-3.3-70b-versatile")
    news_lines = "\n".join(f"- {n}" for n in rd.get("recent_news", [])) or "None verified"
    prompt = f"""A lead scored {score}/100 and was categorized as {category}.

PROSPECT: {state['prospect_name']}, {state['job_title']} at {state['company']} ({stage})
PAIN POINT: {state['pain_point']}
VERIFIED FACTS (confidence: {rd.get('confidence', 'Low')}):
- Hiring: {rd.get('hiring', False)}
- Funding: {rd.get('funding', False)}
- Recent news:
{news_lines}

In one short sentence, explain why this lead got this score. Only mention facts listed above. If confidence is Low, say research was inconclusive."""

    response = llm.invoke([HumanMessage(content=prompt)])
    reason = response.content.strip()

    state["lead_score"] = score
    state["lead_category"] = category
    state["score_reason"] = reason
    state["lead_status"] = "New"
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
            ws.append_row(["Date", "Name", "Title", "Company", "Stage", "Pain Point", "Research", "Connection Request", "Follow-up", "Cold Email", "Lead Score", "Lead Category", "Score Reason", "Lead Status"])
        ws.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            state["prospect_name"], state["job_title"], state["company"],
            state["stage"], state["pain_point"],
            state.get("research", ""),
            state.get("connection_request", ""),
            state.get("followup", ""),
            state.get("cold_email", ""),
            state.get("lead_score", ""),
            state.get("lead_category", ""),
            state.get("score_reason", ""),
            state.get("lead_status", "New")
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
    graph.add_node("scoring", scoring_node)
    graph.add_node("crm", crm_node)
    graph.set_entry_point("research")
    graph.add_edge("research", "outreach")
    graph.add_edge("outreach", "followup")
    graph.add_edge("followup", "scoring")
    graph.add_edge("scoring", "crm")
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

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: research_status = st.empty()
        with col2: outreach_status = st.empty()
        with col3: followup_status = st.empty()
        with col4: scoring_status = st.empty()
        with col5: crm_status = st.empty()

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
            research_data=None,
            connection_request=None,
            followup=None,
            cold_email=None,
            saved=None,
            lead_score=None,
            lead_category=None,
            score_reason=None,
            lead_status=None
        )

        result = graph.invoke(state)

        research_status.success("🔍 Done")
        outreach_status.success("✍️ Done")
        followup_status.success("📧 Done")
        scoring_status.success("🎯 Scored")
        crm_status.success("💾 Saved")

        st.markdown("---")

        if result.get("lead_score") is not None:
            category = result.get("lead_category", "Warm")
            score = result.get("lead_score", 50)
            reason = result.get("score_reason", "")

            render_lead_card(prospect_name, company, score, category, reason)
            st.progress(score / 100)

        if result.get("research_data"):
            with st.expander("🔍 Verified Research Facts"):
                st.json(result["research_data"])

        if result.get("research"):
            with st.expander("📄 Raw Search Results"):
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
        category_idx = headers.index("Lead Category") if "Lead Category" in headers else -1
        status_idx = headers.index("Lead Status") if "Lead Status" in headers else -1

        total = len(rows)
        replies = sum(1 for row in rows if reply_idx >= 0 and len(row) > reply_idx and row[reply_idx].strip().lower() == "yes")
        reply_rate = round((replies / total * 100), 1) if total > 0 else 0
        hot_leads = sum(1 for row in rows if category_idx >= 0 and len(row) > category_idx and row[category_idx].strip() == "Hot")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📨 Outreach Generated", total)
        with col2:
            st.metric("↩️ Replies", replies)
        with col3:
            st.metric("📈 Reply Rate", f"{reply_rate}%")
        with col4:
            st.metric("🔥 Hot Leads", hot_leads)

        # Pipeline funnel
        if status_idx >= 0:
            status_counts = {"New": 0, "Contacted": 0, "Replied": 0, "Meeting Booked": 0, "Won": 0, "Lost": 0}
            for row in rows:
                if len(row) > status_idx:
                    s = row[status_idx].strip()
                    if s in status_counts:
                        status_counts[s] += 1
                    elif s == "":
                        status_counts["New"] += 1

            st.subheader("🔄 Pipeline")
            pcols = st.columns(6)
            labels = ["New", "Contacted", "Replied", "Meeting Booked", "Won", "Lost"]
            for i, label in enumerate(labels):
                with pcols[i]:
                    st.metric(label, status_counts[label])

        st.subheader("👥 Recent Prospects")
        for row in list(reversed(rows))[:5]:
            name = row[name_idx] if len(row) > name_idx else ""
            company = row[company_idx] if len(row) > company_idx else ""
            date = row[date_idx] if len(row) > date_idx else ""
            cat = row[category_idx] if category_idx >= 0 and len(row) > category_idx else ""
            status = row[status_idx] if status_idx >= 0 and len(row) > status_idx and row[status_idx] else "New"
            cat_display = f" | {cat}" if cat else ""
            st.write(f"👤 **{name}** — {company} | {date}{cat_display} | 📋 {status}")
    else:
        st.info("No data yet.")

except Exception as e:
    st.error(f"Error: {e}")

st.markdown("---")
st.caption("Built by Pankaj Singh · AI SDR Agent · LangGraph + Groq + Streamlit")
