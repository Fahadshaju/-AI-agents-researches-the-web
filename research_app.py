import re
import time
from datetime import datetime

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from agents import (
    build_search_agent,
    build_scrape_agent,
    writer_chain,
    critics_prompt,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "gemma4:12B"

st.set_page_config(
    page_title="DeepScout · AI Research Studio",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLES = [
    "Solid-state battery breakthroughs",
    "State of AI regulation worldwide",
    "How CRISPR is changing medicine",
    "Future of quantum computing",
]

STEPS = [
    ("🔍", "Search", "Finding reliable sources"),
    ("📄", "Scrape", "Reading the best page"),
    ("✍️", "Write", "Drafting the report"),
    ("🧐", "Critique", "Reviewing quality"),
]

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
#MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1200px;}

.hero {
  padding: 2.2rem 2rem; border-radius: 20px; margin-bottom: 1.5rem;
  background: radial-gradient(1200px 400px at 0% 0%, rgba(139,108,255,.35), transparent 60%),
              radial-gradient(900px 400px at 100% 100%, rgba(0,212,255,.22), transparent 60%),
              #141925;
  border: 1px solid rgba(255,255,255,.07);
}
.hero h1 {
  font-size: 2.6rem; margin: 0 0 .3rem 0; font-weight: 800; letter-spacing: -.02em;
  background: linear-gradient(90deg,#fff,#b9a7ff 50%,#6fe3ff);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p {margin: 0; color: #a9b0c3; font-size: 1.05rem;}
.badge {
  display:inline-block; padding: .2rem .7rem; border-radius: 999px; font-size: .75rem;
  background: rgba(139,108,255,.18); color:#cbbcff; border:1px solid rgba(139,108,255,.35);
  margin-bottom: .8rem;
}

.stepper {display:grid; grid-template-columns: repeat(4,1fr); gap:.8rem; margin: 1rem 0 1.4rem 0;}
.step {
  border-radius: 16px; padding: 1rem; background:#141925;
  border: 1px solid rgba(255,255,255,.07); transition: all .3s ease;
}
.step .ico {font-size: 1.5rem;}
.step .ttl {font-weight: 700; margin-top:.3rem;}
.step .sub {font-size:.8rem; color:#8c94a8;}
.step.active {
  border-color:#8B6CFF; box-shadow: 0 0 0 1px #8B6CFF, 0 0 24px rgba(139,108,255,.35);
  animation: pulse 1.4s ease-in-out infinite;
}
.step.done {border-color: rgba(52,211,153,.5); background: rgba(52,211,153,.07);}
.step.done .ttl::after {content:" ✓"; color:#34d399;}
.step.error {border-color: rgba(248,113,113,.7); background: rgba(248,113,113,.08);}
@keyframes pulse {50% {box-shadow: 0 0 0 1px #8B6CFF, 0 0 38px rgba(139,108,255,.6);}}

.stat {
  border-radius: 16px; padding: 1rem 1.2rem; background:#141925;
  border: 1px solid rgba(255,255,255,.07);
}
.stat .v {font-size: 1.7rem; font-weight: 800;}
.stat .l {font-size: .8rem; color:#8c94a8; text-transform: uppercase; letter-spacing:.06em;}

.source {
  padding:.65rem .9rem; border-radius: 12px; margin-bottom:.5rem;
  background:#141925; border:1px solid rgba(255,255,255,.07); word-break: break-all;
}
.source a {color:#9fd8ff; text-decoration:none;}
.source a:hover {text-decoration: underline;}

.report-card {
  background:#141925; border:1px solid rgba(255,255,255,.07);
  border-radius: 18px; padding: 1.6rem 2rem;
}
div.stButton > button {border-radius: 12px; font-weight: 600;}
@media (max-width: 800px) {.stepper {grid-template-columns: repeat(2,1fr);}}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def get_llm():
    return ChatOllama(model=MODEL_NAME, temperature=0, disable_streaming=True)


def render_stepper(placeholder, states):
    html = '<div class="stepper">'
    for (icon, title, sub), s in zip(STEPS, states):
        html += (
            f'<div class="step {s}"><div class="ico">{icon}</div>'
            f'<div class="ttl">{title}</div><div class="sub">{sub}</div></div>'
        )
    html += "</div>"
    placeholder.markdown(html, unsafe_allow_html=True)


def extract_urls(*texts):
    seen, urls = set(), []
    for t in texts:
        for u in re.findall(r"https?://[^\s\)\]\}\"'<>]+", t or ""):
            u = u.rstrip(".,;:")
            if u not in seen:
                seen.add(u)
                urls.append(u)
    return urls


def stat(col, value, label):
    col.markdown(
        f'<div class="stat"><div class="v">{value}</div><div class="l">{label}</div></div>',
        unsafe_allow_html=True,
    )


def to_text(x):
    return x if isinstance(x, str) else str(x)


def set_example(text):
    st.session_state.topic_input = text


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("result", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("topic_input", "")

# ---------------------------------------------------------------------------
# Pipeline with live UI
# ---------------------------------------------------------------------------
def run_pipeline(topic, stepper_ph, preview_ph):
    llm = get_llm()
    state = {"topic": topic, "created": datetime.now().strftime("%d %b %Y, %H:%M")}
    timings = {}
    states = ["pending"] * 4
    current = 0

    def mark(i, s):
        states[i] = s
        render_stepper(stepper_ph, states)

    try:
        # 1. Search
        current = 0
        mark(0, "active")
        t = time.time()
        res = build_search_agent().invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Search the web for recent and reliable information "
                        f"on the topic: {topic}.",
                    )
                ]
            }
        )
        state["search_results"] = to_text(res["messages"][-1].content)
        timings["Search"] = time.time() - t
        mark(0, "done")
        with preview_ph.container():
            with st.expander("🔍 Search preview", expanded=False):
                st.markdown(state["search_results"][:700] + "…")

        # 2. Scrape
        current = 1
        mark(1, "active")
        t = time.time()
        res = build_scrape_agent().invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Based on the following search about '{topic}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{state['search_results'][:700]}",
                    )
                ]
            }
        )
        state["scraped_content"] = to_text(res["messages"][-1].content)
        timings["Scrape"] = time.time() - t
        mark(1, "done")

        # 3. Write
        current = 2
        mark(2, "active")
        t = time.time()
        combined = (
            f"Search results:\n{state['search_results']}\n\n"
            f"Detailed scraped content:\n{state['scraped_content']}"
        )
        state["report"] = to_text(
            writer_chain.invoke({"topic": topic, "research": combined})
        )
        timings["Write"] = time.time() - t
        mark(2, "done")

        # 4. Critique
        current = 3
        mark(3, "active")
        t = time.time()
        critic_chain = critics_prompt | llm | StrOutputParser()
        state["feedback"] = to_text(critic_chain.invoke({"report": state["report"]}))
        timings["Critique"] = time.time() - t
        mark(3, "done")

        state["timings"] = timings
        return state

    except Exception as e:
        mark(current, "error")
        raise e


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛰️ DeepScout")
    st.caption("Multi-agent research studio")
    st.markdown(f"**Critic model:** `{MODEL_NAME}`  \n**Runtime:** Ollama (local)")

    st.divider()
    st.markdown("#### 🕘 History")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            label = item["topic"][:34] + ("…" if len(item["topic"]) > 34 else "")
            if st.button(f"📄 {label}", key=f"h{i}", use_container_width=True):
                st.session_state.result = item
        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()
    else:
        st.caption("Your research runs will appear here.")

    st.divider()
    st.caption("Tip: the first run can be slow while Ollama loads the model.")

# ---------------------------------------------------------------------------
# Hero + input
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <div class="badge">⚡ Search → Scrape → Write → Critique</div>
  <h1>DeepScout Research Studio</h1>
  <p>Give it a topic. A team of AI agents researches the web, reads the best source,
  writes a report and then reviews its own work — all running locally.</p>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
with c1:
    topic = st.text_input(
        "Research topic",
        key="topic_input",
        placeholder="e.g. Latest advances in solid-state batteries",
        label_visibility="collapsed",
    )
with c2:
    run_clicked = st.button("Run research 🚀", type="primary", use_container_width=True)

st.caption("Try an example:")
ex_cols = st.columns(len(EXAMPLES))
for col, ex in zip(ex_cols, EXAMPLES):
    col.button(ex, key=f"ex_{ex}", on_click=set_example, args=(ex,), use_container_width=True)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
stepper_ph = st.empty()
preview_ph = st.empty()

if run_clicked:
    clean_topic = topic.strip()
    if not clean_topic:
        st.warning("Please enter a research topic first.")
    else:
        try:
            result = run_pipeline(clean_topic, stepper_ph, preview_ph)
            st.session_state.result = result
            st.session_state.history.append(result)
            st.toast("Research complete!", icon="✅")
        except Exception as e:
            st.error(f"The pipeline failed: {e}")
            st.info(
                "Checklist: Ollama is running (`ollama serve`), the model is pulled "
                "(`ollama pull gemma4:12B`), and the API keys in `.env` are set."
            )
elif st.session_state.result is None:
    render_stepper(stepper_ph, ["pending"] * 4)
else:
    render_stepper(stepper_ph, ["done"] * 4)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
res = st.session_state.result
if res:
    report = res["report"]
    words = len(report.split())
    urls = extract_urls(res["search_results"], res["scraped_content"])
    timings = res.get("timings", {})
    total = sum(timings.values()) if timings else 0

    st.markdown(f"## {res['topic']}")
    st.caption(f"Generated {res.get('created', '')}")

    s1, s2, s3, s4 = st.columns(4)
    stat(s1, f"{words:,}", "Words")
    stat(s2, f"{max(1, round(words / 200))} min", "Reading time")
    stat(s3, len(urls), "Sources found")
    stat(s4, f"{total:.0f}s" if total else "—", "Total time")

    st.write("")
    tab_report, tab_feedback, tab_sources, tab_raw, tab_perf = st.tabs(
        ["📝 Report", "🧐 Critic feedback", "🔗 Sources", "🗂️ Raw research", "⏱️ Performance"]
    )

    with tab_report:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown(report)
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button(
            "⬇️ Markdown",
            data=report,
            file_name="research_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        full = (
            f"# {res['topic']}\n\n{report}\n\n---\n\n## Critic feedback\n\n{res['feedback']}"
        )
        d2.download_button(
            "⬇️ Report + critique",
            data=full,
            file_name="research_report_full.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with tab_feedback:
        st.info("An independent critic pass reviewed the report above.", icon="🧐")
        st.markdown(res["feedback"])

    with tab_sources:
        if urls:
            for u in urls:
                st.markdown(
                    f'<div class="source">🔗 <a href="{u}" target="_blank">{u}</a></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No URLs were detected in the agent output.")

    with tab_raw:
        with st.expander("Search agent output", expanded=False):
            st.markdown(res["search_results"])
        with st.expander("Scrape agent output", expanded=False):
            st.markdown(res["scraped_content"])

    with tab_perf:
        if timings:
            import pandas as pd

            df = pd.DataFrame(
                {"Step": list(timings.keys()), "Seconds": [round(v, 1) for v in timings.values()]}
            ).set_index("Step")
            st.bar_chart(df, color="#8B6CFF")
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("No timing data available.")
