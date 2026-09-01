import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import gspread
from google.oauth2.service_account import Credentials
import requests
import json
from datetime import datetime
from typing import TypedDict, Optional
from bs4 import BeautifulSoup
from urllib.parse import urlparse

st.set_page_config(
    page_title="AI SDR Agent",
    page_icon="🤖",
    layout="centered"
)

# =========================================================
# UI
# =========================================================

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


def safe_json_from_llm(text, fallback):
    """Safely parse JSON returned by the LLM."""
    if not text:
        return fallback

    raw = str(text).strip()

    if raw.startswith("```"):
        raw = raw.replace("```json", "", 1)
        raw = raw.replace("```JSON", "", 1)
        raw = raw.replace("```", "")
        raw = raw.strip()

    try:
        return json.loads(raw)
    except Exception:
        # Try extracting the first JSON object.
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                pass

    return fallback


def normalize_status(value):
    """
    Normalize hiring/funding into:
    {
        "status": "true" | "false" | "unknown",
        "evidence": [...]
    }
    """
    if not isinstance(value, dict):
        return {"status": "unknown", "evidence": []}

    status = str(value.get("status", "unknown")).strip().lower()

    # Backward compatibility if an older LLM returns boolean/string.
    if status in ("true", "yes", "confirmed"):
        status = "true"
    elif status in ("false", "no", "not confirmed"):
        status = "false"
    else:
        status = "unknown"

    evidence = value.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []

    clean_evidence = []
    for item in evidence:
        if isinstance(item, dict):
            clean_evidence.append({
                "claim": str(item.get("claim", "")).strip(),
                "source": str(item.get("source", "")).strip(),
                "date": str(item.get("date", "")).strip()
            })
        elif isinstance(item, str):
            clean_evidence.append({
                "claim": item.strip(),
                "source": "",
                "date": ""
            })

    return {
        "status": status,
        "evidence": clean_evidence
    }


def normalize_research(structured, name, company, job_title, linkedin_url, fallback_sources):
    """
    Enforce a stable research schema and immutable identity.
    The LLM is never allowed to overwrite identity fields.
    """
    if not isinstance(structured, dict):
        structured = {}

    recent_news = structured.get("recent_news", [])
    if not isinstance(recent_news, list):
        recent_news = []

    clean_news = []
    for item in recent_news:
        if isinstance(item, dict):
            clean_news.append({
                "claim": str(item.get("claim", "")).strip(),
                "source": str(item.get("source", "")).strip(),
                "date": str(item.get("date", "")).strip()
            })
        elif isinstance(item, str):
            clean_news.append({
                "claim": item.strip(),
                "source": "",
                "date": ""
            })

    sources = structured.get("sources", [])
    if not isinstance(sources, list):
        sources = []

    sources = [str(s).strip() for s in sources if str(s).strip()]

    confidence = str(structured.get("confidence", "Low")).strip().title()
    if confidence not in {"High", "Medium", "Low"}:
        confidence = "Low"

    # Immutable identity: always use upstream values.
    normalized = {
        "person": name,
        "company": company,
        "job_title": job_title,
        "lead_url": linkedin_url,
        "recent_news": clean_news[:3],
        "hiring": normalize_status(structured.get("hiring")),
        "funding": normalize_status(structured.get("funding")),
        "sources": list(dict.fromkeys(sources))[:10],
        "confidence": confidence
    }

    if not normalized["sources"]:
        normalized["sources"] = fallback_sources[:10]

    return normalized


# =========================================================
# State
# =========================================================

class SDRState(TypedDict):
    lead_url: Optional[str]
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


# =========================================================
# Extraction Node
# =========================================================

def extraction_node(state: SDRState) -> SDRState:
    """
    Extract lead identity from the exact user-supplied URL.

    Important:
    - Research node will NOT re-search the person's name when a URL exists.
    - LinkedIn pages often block direct scraping, so Serper is used as a
      discovery layer for the exact profile URL.
    """
    url = (state.get("lead_url") or "").strip()

    if not url:
        return state

    headers = {
        "X-API-KEY": state["serper_key"],
        "Content-Type": "application/json"
    }

    is_linkedin = "linkedin.com/in/" in url.lower()
    snippet_text = ""
    page_text = ""

    # -----------------------------------------------------
    # Search exact URL / domain
    # -----------------------------------------------------
    try:
                if is_linkedin:
            parsed = urlparse(url)
            profile_slug = parsed.path.rstrip("/").split("/")[-1]
            query = f'site:linkedin.com/in "{profile_slug.replace("-", " ")}"'
        else:
            domain = urlparse(url).netloc or url
            query = f'"{domain}" official company'
            

        data = {
            "q": query,
            "num": 10
        }

        res = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            json=data,
            timeout=10
        )

        organic = res.json().get("organic", [])

        # Prefer an exact URL match for LinkedIn.
        if is_linkedin:
            target = url.rstrip("/").lower()
            organic = sorted(
                organic,
                key=lambda r: (
                    0 if r.get("link", "").rstrip("/").lower() == target else 1
                )
            )

        snippet_text = "\n".join(
            f"TITLE: {r.get('title', '')}\n"
            f"LINK: {r.get('link', '')}\n"
            f"SNIPPET: {r.get('snippet', '')}"
            for r in organic[:10]
        )

    except Exception as e:
        print("EXTRACTION SEARCH ERROR:", e)

    # -----------------------------------------------------
    # Company website direct fetch
    # -----------------------------------------------------
    if not is_linkedin:
        try:
            r = requests.get(
                url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            soup = BeautifulSoup(r.text, "html.parser")

            meta = soup.find(
                "meta",
                attrs={"name": "description"}
            )

            meta_text = (
                meta.get("content", "")
                if meta
                else ""
            )

            title_text = (
                soup.title.get_text(" ", strip=True)
                if soup.title
                else ""
            )

            body = " ".join(
                p.get_text(" ", strip=True)
                for p in soup.find_all("p")
            )[:2500]

            page_text = (
                f"TITLE: {title_text}\n"
                f"META: {meta_text}\n"
                f"BODY: {body}"
            )

        except Exception as e:
            print("WEBSITE FETCH ERROR:", e)

    # -----------------------------------------------------
    # LLM extraction
    # -----------------------------------------------------

    llm = ChatGroq(
        api_key=state["groq_key"],
        model="openai/gpt-oss-120b"
    )

    prompt = f"""
You are extracting a B2B prospect identity.

USER-SUPPLIED URL:
{url}

IS LINKEDIN PROFILE:
{is_linkedin}

SEARCH RESULTS:
{snippet_text}

PAGE TEXT:
{page_text}

Return ONLY valid JSON:

{{
  "prospect_name": "",
  "job_title": "",
  "company": "",
  "stage": "",
  "pain_point": ""
}}

STRICT RULES:

1. If this is a LinkedIn profile URL, use only evidence that clearly belongs
   to this exact profile URL.
2. Do NOT substitute another person with the same name.
3. Do NOT infer a company from an unrelated search result.
4. If company/title cannot be verified, return an empty string.
5. Do not invent a person's identity.
6. stage must be one of:
   - Early-stage Startup
   - Series A SaaS
   - Growth-stage SaaS
   - Bootstrapped SaaS
   - Unknown
7. pain_point must be a conservative role/industry-based hypothesis.
"""

    fallback = {
        "prospect_name": "",
        "job_title": "",
        "company": "",
        "stage": "Unknown",
        "pain_point": ""
    }

    try:
        response = llm.invoke(
            [HumanMessage(content=prompt)]
        )
        extracted = safe_json_from_llm(
            response.content,
            fallback
        )
    except Exception as e:
        print("EXTRACTION LLM ERROR:", e)
        extracted = fallback

    state["prospect_name"] = str(
        extracted.get("prospect_name", "")
    ).strip()

    state["job_title"] = str(
        extracted.get("job_title", "")
    ).strip()

    state["company"] = str(
        extracted.get("company", "")
    ).strip()

    stage = str(
        extracted.get("stage", "Unknown")
    ).strip()

    valid_stages = {
        "Early-stage Startup",
        "Series A SaaS",
        "Growth-stage SaaS",
        "Bootstrapped SaaS",
        "Unknown"
    }

    state["stage"] = (
        stage if stage in valid_stages
        else "Unknown"
    )

    state["pain_point"] = str(
        extracted.get("pain_point", "")
    ).strip()

    return state


# =========================================================
# Research Node
# =========================================================

def research_node(state: SDRState) -> SDRState:
    headers = {
        "X-API-KEY": state["serper_key"],
        "Content-Type": "application/json"
    }

    name = (state.get("prospect_name") or "").strip()
    company = (state.get("company") or "").strip()
    job_title = (state.get("job_title") or "").strip()
    linkedin_url = (state.get("lead_url") or "").strip()

    all_results = []
    seen_links = set()

    def search_serper(query, num=5):
        try:
            data = {
                "q": query,
                "num": num
            }

            res = requests.post(
                "https://google.serper.dev/search",
                headers=headers,
                json=data,
                timeout=10
            )

            return res.json().get("organic", [])

        except Exception as e:
            print("SEARCH ERROR:", e)
            return []

    def add_results(results, source_type):
        for r in results:
            link = str(r.get("link", "")).strip()

            if not link or link in seen_links:
                continue

            seen_links.add(link)

            all_results.append({
                "title": str(r.get("title", "")),
                "link": link,
                "snippet": str(r.get("snippet", "")),
                "source_type": source_type
            })

    # -----------------------------------------------------
    # 0. IDENTITY LOCK
    # -----------------------------------------------------
    # If the user supplied a LinkedIn URL, DO NOT perform
    # a name-only LinkedIn search.
    # The URL/identity came from the upstream extraction node.
    # -----------------------------------------------------

    if linkedin_url:
        all_results.append({
            "title": f"{name} — {job_title}".strip(" —"),
            "link": linkedin_url,
            "snippet": (
                f"{name} — verified target profile "
                f"from user-supplied URL"
            ),
            "source_type": "linkedin_profile_verified"
        })
        seen_links.add(linkedin_url)

    elif name:
        # Fallback only if there is no supplied profile URL.
        identity_query = (
            f'site:linkedin.com/in "{name}"'
        )

        print(
            "IDENTITY QUERY (fallback, no URL):",
            identity_query
        )

        results = search_serper(
            identity_query,
            10
        )

        add_results(
            results,
            "linkedin_profile_unverified"
        )

    # -----------------------------------------------------
    # 1. COMPANY RESEARCH
    # -----------------------------------------------------

    if company:
        add_results(
            search_serper(
                f'"{company}" official website',
                5
            ),
            "company"
        )

        add_results(
            search_serper(
                f'"{company}" '
                f'(funding OR raised OR "series A" OR '
                f'"series B" OR investment)',
                5
            ),
            "funding"
        )

        add_results(
            search_serper(
                f'"{company}" '
                f'(hiring OR "open roles" OR careers OR jobs)',
                5
            ),
            "hiring"
        )

        add_results(
            search_serper(
                f'"{company}" '
                f'(news OR launches OR announces OR announcement)',
                5
            ),
            "news"
        )

    # -----------------------------------------------------
    # 2. NO RESULTS
    # -----------------------------------------------------

    if not all_results:
        state["research"] = "No verified results found."

        state["research_data"] = {
            "person": name,
            "company": company,
            "job_title": job_title,
            "lead_url": linkedin_url,
            "recent_news": [],
            "hiring": {
                "status": "unknown",
                "evidence": []
            },
            "funding": {
                "status": "unknown",
                "evidence": []
            },
            "sources": [],
            "confidence": "Low"
        }

        return state

    # -----------------------------------------------------
    # 3. FILTER UNVERIFIED LINKEDIN RESULTS
    # -----------------------------------------------------

    relevant_results = []

    for r in all_results:
        text = (
            r["title"] + " " +
            r["snippet"]
        ).lower()

        if r["source_type"] == "linkedin_profile_unverified":
            name_parts = [
                p.lower()
                for p in name.split()
                if len(p) > 2
            ]

            if name_parts and not all(
                part in text
                for part in name_parts
            ):
                continue

        relevant_results.append(r)

    relevant_results = relevant_results[:15]

    print(
        "RESEARCH RESULTS:",
        len(relevant_results)
    )

    raw_block = "\n\n".join(
        f"""
[SOURCE TYPE: {r['source_type']}]
[SOURCE: {r['link']}]
TITLE: {r['title']}
SNIPPET: {r['snippet']}
"""
        for r in relevant_results
    )

    state["research"] = raw_block

    # -----------------------------------------------------
    # 4. LLM VERIFICATION
    # -----------------------------------------------------

    llm = ChatGroq(
        api_key=state["groq_key"],
        model="openai/gpt-oss-120b"
    )

    extraction_prompt = f"""
You are a strict B2B research verification system.

TARGET PERSON:
{name}

TARGET COMPANY:
{company}

TARGET JOB TITLE:
{job_title}

VERIFIED LINKEDIN URL:
{linkedin_url or "not provided"}

SEARCH RESULTS:
{raw_block}

Return ONLY valid JSON:

{{
  "person": "{name}",
  "company": "{company}",
  "recent_news": [],
  "hiring": {{
    "status": "unknown",
    "evidence": []
  }},
  "funding": {{
    "status": "unknown",
    "evidence": []
  }},
  "sources": [],
  "confidence": "Low"
}}

RULES:

1. [SOURCE TYPE: linkedin_profile_verified] is the identity
   source of truth. Never replace the target person/company/title
   with another person.

2. The verified LinkedIn source MUST NOT be used as evidence
   for company news, hiring, or funding.

3. Source-type restrictions are strict:
   - funding → funding claims ONLY
   - hiring → hiring claims ONLY
   - news → recent_news ONLY
   - company → general company facts ONLY
   - linkedin_profile_verified → identity ONLY

4. Never mix facts from another person with the same name.

5. "Founder of X", "works at X", or job-title history is NOT
   recent_news.

6. recent_news must be a real company event, not a profile fact.

7. Only classify something as recent_news if it appears to be
   from the last 90 days.

8. If a news item's publication date is unavailable or clearly
   older than 90 days, do NOT include it.

9. hiring.status:
   - "true" ONLY with explicit evidence from a hiring source
     that the TARGET COMPANY is hiring or has open roles.
   - "false" ONLY if a source explicitly says hiring is frozen,
     stopped, or closed.
   - otherwise "unknown".

10. funding.status:
   - "true" ONLY with explicit evidence from a funding source
     that the TARGET COMPANY raised funding/investment.
   - "false" ONLY with explicit evidence that there was no funding.
   - otherwise "unknown".

11. Every recent_news item MUST be an object:
   {{
      "claim": "...",
      "source": "URL",
      "date": "date if available"
   }}

12. Every hiring/funding evidence item MUST be an object:
   {{
      "claim": "...",
      "source": "URL",
      "date": "date if available"
   }}

13. Evidence must be traceable to the supplied search results.

14. Never invent dates, URLs, funding, hiring, or news.

15. confidence:
   - High = verified identity + strong company evidence
   - Medium = verified identity + weak/partial company evidence
   - Low = weak/conflicting evidence
"""

    fallback = {
        "person": name,
        "company": company,
        "recent_news": [],
        "hiring": {
            "status": "unknown",
            "evidence": []
        },
        "funding": {
            "status": "unknown",
            "evidence": []
        },
        "sources": [
            r["link"]
            for r in relevant_results[:5]
        ],
        "confidence": "Low"
    }

    try:
        response = llm.invoke(
            [HumanMessage(content=extraction_prompt)]
        )

        structured = safe_json_from_llm(
            response.content,
            fallback
        )

    except Exception as e:
        print("LLM RESEARCH ERROR:", e)
        structured = fallback

    # -----------------------------------------------------
    # 5. NORMALIZE + IMMUTABLE IDENTITY LOCK
    # -----------------------------------------------------

    structured = normalize_research(
        structured,
        name,
        company,
        job_title,
        linkedin_url,
        [r["link"] for r in relevant_results]
    )

    state["research_data"] = structured

    print("FINAL RESEARCH:")
    print(
        json.dumps(
            structured,
            indent=2,
            ensure_ascii=False
        )
    )

    return state


# =========================================================
# Outreach Node
# =========================================================

def outreach_node(state: SDRState) -> SDRState:
    llm = ChatGroq(
        api_key=state["groq_key"],
        model="openai/gpt-oss-120b"
    )

    rd = state.get("research_data", {}) or {}

    funding_status = (
        (rd.get("funding") or {})
        .get("status", "unknown")
    )

    hiring_status = (
        (rd.get("hiring") or {})
        .get("status", "unknown")
    )

    news_lines = "\n".join(
        f"- {n.get('claim', '')}"
        for n in rd.get("recent_news", [])
        if isinstance(n, dict)
    )

    if not news_lines:
        news_lines = "None verified"

    prompt = f"""
You are an elite B2B outreach copywriter.

VERIFIED PROSPECT:
- Name: {state['prospect_name']}
- Job title: {state['job_title']}
- Company: {state['company']}
- Stage: {state['stage']}
- Research confidence: {rd.get('confidence', 'Low')}

VERIFIED COMPANY FACTS:
- Hiring status: {hiring_status}
- Funding status: {funding_status}
- Recent verified news:
{news_lines}

PAIN POINT:
{state['pain_point']}

OFFER:
{state['offer']}

BENEFIT:
{state['benefit']}

TONE:
{state['tone']}

RULES:
- Only reference verified facts.
- Never mention "unknown" in outreach.
- Do not mention hiring/funding unless status is "true".
- Do not mention recent news unless it exists above.
- Never invent a company event.
- NEVER use "hope you're doing well".
- NEVER use "love what you're building".
- Connection request: maximum 50 words.
- Connection request ends with a soft question.
- Sound like a founder, not a marketer.
- Short sentences only.

OUTPUT:

CONNECTION REQUEST:
[message]

COLD EMAIL:
[subject line]
[email body]
"""

    try:
        response = llm.invoke(
            [HumanMessage(content=prompt)]
        )
        result = response.content.strip()
    except Exception as e:
        print("OUTREACH ERROR:", e)
        return state

    if "CONNECTION REQUEST:" in result:
        conn = result.split(
            "CONNECTION REQUEST:",
            1
        )[1]

        if "COLD EMAIL:" in conn:
            state["connection_request"] = (
                conn.split("COLD EMAIL:", 1)[0]
                .strip()
            )
            state["cold_email"] = (
                result.split("COLD EMAIL:", 1)[1]
                .strip()
            )
        else:
            state["connection_request"] = conn.strip()

    return state


# =========================================================
# Follow-up Node
# =========================================================

def followup_node(state: SDRState) -> SDRState:
    llm = ChatGroq(
        api_key=state["groq_key"],
        model="openai/gpt-oss-120b"
    )

    prompt = f"""
Write a follow-up LinkedIn DM.

Prospect:
{state['prospect_name']} at {state['company']}

Original offer:
{state['offer']}

Benefit:
{state['benefit']}

RULES:
- 3-4 lines maximum.
- Lead with value, not pitch.
- One soft CTA at the end.
- Sound human.
- Do not invent company facts.
- Write only the message.
"""

    try:
        response = llm.invoke(
            [HumanMessage(content=prompt)]
        )
        state["followup"] = response.content.strip()
    except Exception as e:
        print("FOLLOWUP ERROR:", e)

    return state


# =========================================================
# Scoring Node
# =========================================================

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

    # IMPORTANT:
    # hiring/funding are now dictionaries, so status must be read.
    funding_status = (
        (rd.get("funding") or {})
        .get("status", "unknown")
    )

    hiring_status = (
        (rd.get("hiring") or {})
        .get("status", "unknown")
    )

    if funding_status == "true":
        score += 20

    if hiring_status == "true":
        score += 15

    if rd.get("recent_news"):
        score += 10

    if (
        state.get("pain_point")
        and len(state["pain_point"].strip()) > 5
    ):
        score += 10

    if rd.get("confidence") == "Low":
        score = min(score, 60)

    score = max(
        0,
        min(100, score)
    )

    if score >= 70:
        category = "Hot"
    elif score >= 45:
        category = "Warm"
    else:
        category = "Cold"

    llm = ChatGroq(
        api_key=state["groq_key"],
        model="openai/gpt-oss-120b"
    )

    news_lines = "\n".join(
        f"- {n.get('claim', '')}"
        for n in rd.get("recent_news", [])
        if isinstance(n, dict)
    )

    if not news_lines:
        news_lines = "None verified"

    prompt = f"""
A lead scored {score}/100 and was categorized as {category}.

PROSPECT:
{state['prospect_name']}, {state['job_title']}
at {state['company']} ({stage})

PAIN POINT:
{state['pain_point']}

VERIFIED FACTS:
- Hiring: {hiring_status}
- Funding: {funding_status}
- Recent news:
{news_lines}
- Research confidence: {rd.get('confidence', 'Low')}

Write ONE short sentence explaining the score.

RULES:
- Only mention facts listed above.
- Do not invent facts.
- If confidence is Low, say research was inconclusive.
"""

    try:
        response = llm.invoke(
            [HumanMessage(content=prompt)]
        )
        reason = response.content.strip()
    except Exception:
        reason = (
            "Score based on available stage, research, "
            "pain-point and verified company signals."
        )

    state["lead_score"] = score
    state["lead_category"] = category
    state["score_reason"] = reason
    state["lead_status"] = "New"

    return state


# =========================================================
# CRM Node
# =========================================================

def crm_node(state: SDRState) -> SDRState:
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]

        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=scopes
        )

        gc = gspread.authorize(creds)

        sh = gc.open_by_key(
            st.secrets["SHEET_ID"]
        )

        try:
            ws = sh.worksheet("SDR CRM")
        except Exception:
            ws = sh.add_worksheet(
                title="SDR CRM",
                rows="1000",
                cols="20"
            )

            ws.append_row([
                "Date",
                "Name",
                "Title",
                "Company",
                "LinkedIn URL",
                "Stage",
                "Pain Point",
                "Research",
                "Connection Request",
                "Follow-up",
                "Cold Email",
                "Lead Score",
                "Lead Category",
                "Score Reason",
                "Lead Status"
            ])

        ws.append_row([
            datetime.now().strftime(
                "%Y-%m-%d %H:%M"
            ),
            state["prospect_name"],
            state["job_title"],
            state["company"],
            state.get("lead_url", ""),
            state["stage"],
            state["pain_point"],
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
        print("CRM ERROR:", e)
        state["saved"] = False

    return state


# =========================================================
# Build Graph
# =========================================================

def build_graph():
    graph = StateGraph(SDRState)

    graph.add_node(
        "extraction",
        extraction_node
    )

    graph.add_node(
        "research",
        research_node
    )

    graph.add_node(
        "outreach",
        outreach_node
    )

    graph.add_node(
        "followup",
        followup_node
    )

    graph.add_node(
        "scoring",
        scoring_node
    )

    graph.add_node(
        "crm",
        crm_node
    )

    graph.set_entry_point("extraction")

    graph.add_edge(
        "extraction",
        "research"
    )

    graph.add_edge(
        "research",
        "outreach"
    )

    graph.add_edge(
        "outreach",
        "followup"
    )

    graph.add_edge(
        "followup",
        "scoring"
    )

    graph.add_edge(
        "scoring",
        "crm"
    )

    graph.add_edge(
        "crm",
        END
    )

    return graph.compile()


# =========================================================
# Header
# =========================================================

st.markdown("""
<div class="hero">
    <h1>🤖 AI SDR Agent</h1>
    <p>Multi-agent system: Extract → Research → Outreach → Follow-up → Scoring → CRM</p>
    <div>
        <span class="node-badge">🌐 Extraction Node</span>
        <span class="node-badge">🔍 Research Node</span>
        <span class="node-badge">✍️ Outreach Node</span>
        <span class="node-badge">📧 Follow-up Node</span>
        <span class="node-badge">🎯 Scoring Node</span>
        <span class="node-badge">💾 CRM Node</span>
    </div>
</div>""", unsafe_allow_html=True)


# =========================================================
# UI
# =========================================================

st.subheader("🎯 Prospect Info")

lead_url = st.text_input(
    "LinkedIn Profile URL or Company Website",
    placeholder=(
        "e.g. https://linkedin.com/in/rahulsharma "
        "or https://razorpay.com"
    )
)

st.subheader("💼 Your Offer")

offer = st.text_input(
    "What You're Offering",
    placeholder=(
        "e.g. AI agent that automates LinkedIn outreach"
    )
)

benefit = st.text_input(
    "Key Benefit",
    placeholder=(
        "e.g. saves 5 hours/week"
    )
)

tone = st.radio(
    "Tone",
    [
        "Professional",
        "Casual",
        "Founder-style",
        "Direct"
    ],
    horizontal=True
)


# =========================================================
# RUN
# =========================================================

if st.button(
    "🚀 Run AI SDR Agent",
    use_container_width=True
):

    if not lead_url or not offer:
        st.error(
            "Please provide a LinkedIn/website URL and your Offer."
        )

    else:
        graph = build_graph()

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            extraction_status = st.empty()

        with col2:
            research_status = st.empty()

        with col3:
            outreach_status = st.empty()

        with col4:
            followup_status = st.empty()

        with col5:
            scoring_status = st.empty()

        with col6:
            crm_status = st.empty()

        extraction_status.info(
            "🌐 Extracting..."
        )

        state = SDRState(
            lead_url=lead_url,
            prospect_name="",
            job_title="",
            company="",
            stage="",
            pain_point="",
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

        try:
            result = graph.invoke(state)

            extraction_status.success("🌐 Done")
            research_status.success("🔍 Done")
            outreach_status.success("✍️ Done")
            followup_status.success("📧 Done")
            scoring_status.success("🎯 Scored")

            if result.get("saved"):
                crm_status.success("💾 Saved")
            else:
                crm_status.warning("💾 Not saved")

            st.markdown("---")

            # -------------------------------------------------
            # Extracted lead
            # -------------------------------------------------

            with st.expander(
                "🌐 Auto-extracted lead info"
            ):
                st.write(
                    f"**Name:** "
                    f"{result.get('prospect_name', '—')}"
                )

                st.write(
                    f"**Title:** "
                    f"{result.get('job_title', '—')}"
                )

                st.write(
                    f"**Company:** "
                    f"{result.get('company', '—')}"
                )

                st.write(
                    f"**Stage:** "
                    f"{result.get('stage', '—')}"
                )

                st.write(
                    f"**Pain point:** "
                    f"{result.get('pain_point', '—')}"
                )

                st.write(
                    f"**LinkedIn URL:** "
                    f"{result.get('lead_url', '—')}"
                )

            # -------------------------------------------------
            # Score
            # -------------------------------------------------

            if result.get("lead_score") is not None:

                category = result.get(
                    "lead_category",
                    "Warm"
                )

                score = result.get(
                    "lead_score",
                    50
                )

                reason = result.get(
                    "score_reason",
                    ""
                )

                render_lead_card(
                    result.get(
                        "prospect_name",
                        ""
                    ),
                    result.get(
                        "company",
                        ""
                    ),
                    score,
                    category,
                    reason
                )

                st.progress(
                    score / 100
                )

            # -------------------------------------------------
            # Research
            # -------------------------------------------------

            if result.get("research_data"):

                with st.expander(
                    "🔍 Verified Research Facts"
                ):

                    st.json(
                        result["research_data"]
                    )

            # -------------------------------------------------
            # Raw research
            # -------------------------------------------------

            if result.get("research"):

                with st.expander(
                    "📄 Raw Search Results"
                ):

                    st.write(
                        result["research"]
                    )

            # -------------------------------------------------
            # Outreach
            # -------------------------------------------------

            if result.get(
                "connection_request"
            ):

                st.subheader(
                    "📨 Connection Request"
                )

                st.code(
                    result["connection_request"],
                    language=None
                )

            if result.get("followup"):

                st.subheader(
                    "💬 Follow-up Message"
                )

                st.code(
                    result["followup"],
                    language=None
                )

            if result.get("cold_email"):

                st.subheader(
                    "📧 Cold Email"
                )

                st.code(
                    result["cold_email"],
                    language=None
                )

            if result.get("saved"):

                st.success(
                    "✅ Saved to Google Sheets CRM!"
                )

            else:

                st.warning(
                    "⚠️ Research/outreach completed, "
                    "but CRM save failed. Check terminal logs."
                )

        except Exception as e:

            extraction_status.error("🌐 Failed")
            research_status.error("🔍 Failed")
            outreach_status.error("✍️ Failed")
            followup_status.error("📧 Failed")
            scoring_status.error("🎯 Failed")
            crm_status.error("💾 Failed")

            st.error(
                f"Agent execution error: {e}"
            )


# =========================================================
# Analytics Dashboard
# =========================================================

st.markdown("---")
st.subheader("📊 Analytics Dashboard")

try:
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes
    )

    gc = gspread.authorize(creds)

    sh = gc.open_by_key(
        st.secrets["SHEET_ID"]
    )

    ws = sh.worksheet("SDR CRM")

    all_values = ws.get_all_values()

    if len(all_values) > 1:

        headers = all_values[0]
        rows = all_values[1:]

        reply_idx = (
            headers.index("Reply")
            if "Reply" in headers
            else -1
        )

        name_idx = (
            headers.index("Name")
            if "Name" in headers
            else 1
        )

        company_idx = (
            headers.index("Company")
            if "Company" in headers
            else 3
        )

        date_idx = (
            headers.index("Date")
            if "Date" in headers
            else 0
        )

        category_idx = (
            headers.index("Lead Category")
            if "Lead Category" in headers
            else -1
        )

        status_idx = (
            headers.index("Lead Status")
            if "Lead Status" in headers
            else -1
        )

        total = len(rows)

        replies = sum(
            1
            for row in rows
            if (
                reply_idx >= 0
                and len(row) > reply_idx
                and row[reply_idx]
                .strip()
                .lower() == "yes"
            )
        )

        reply_rate = (
            round(
                replies / total * 100,
                1
            )
            if total > 0
            else 0
        )

        hot_leads = sum(
            1
            for row in rows
            if (
                category_idx >= 0
                and len(row) > category_idx
                and row[category_idx]
                .strip() == "Hot"
            )
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📨 Outreach Generated",
                total
            )

        with col2:
            st.metric(
                "↩️ Replies",
                replies
            )

        with col3:
            st.metric(
                "📈 Reply Rate",
                f"{reply_rate}%"
            )

        with col4:
            st.metric(
                "🔥 Hot Leads",
                hot_leads
            )

        if status_idx >= 0:

            status_counts = {
                "New": 0,
                "Contacted": 0,
                "Replied": 0,
                "Meeting Booked": 0,
                "Won": 0,
                "Lost": 0
            }

            for row in rows:

                if len(row) > status_idx:

                    s = row[status_idx].strip()

                    if s in status_counts:
                        status_counts[s] += 1

                    elif s == "":
                        status_counts["New"] += 1

            st.subheader("🔄 Pipeline")

            pcols = st.columns(6)

            labels = [
                "New",
                "Contacted",
                "Replied",
                "Meeting Booked",
                "Won",
                "Lost"
            ]

            for i, label in enumerate(labels):

                with pcols[i]:

                    st.metric(
                        label,
                        status_counts[label]
                    )

        st.subheader(
            "👥 Recent Prospects"
        )

        for row in list(
            reversed(rows)
        )[:5]:

            name = (
                row[name_idx]
                if len(row) > name_idx
                else ""
            )

            company = (
                row[company_idx]
                if len(row) > company_idx
                else ""
            )

            date = (
                row[date_idx]
                if len(row) > date_idx
                else ""
            )

            cat = (
                row[category_idx]
                if (
                    category_idx >= 0
                    and len(row) > category_idx
                )
                else ""
            )

            status = (
                row[status_idx]
                if (
                    status_idx >= 0
                    and len(row) > status_idx
                    and row[status_idx]
                )
                else "New"
            )

            cat_display = (
                f" | {cat}"
                if cat
                else ""
            )

            st.write(
                f"👤 **{name}** — "
                f"{company} | {date}"
                f"{cat_display} | 📋 {status}"
            )

    else:

        st.info(
            "No data yet."
        )

except Exception as e:

    st.error(
        f"Analytics error: {e}"
    )

st.markdown("---")
