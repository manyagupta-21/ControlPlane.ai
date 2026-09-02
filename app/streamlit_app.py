"""ControlPlane demo UI.

Run locally: streamlit run app/streamlit_app.py

Lets you push any AI response through the control layer, see each detector's
risk, the tiered decision (and why), and capture a human override that is
written back to the audit trail, closing the feedback loop live on screen.

Two independent checks are shown, run at two different pipeline stages.
The input gate runs on the QUERY, before any generation happens. The output
pipeline runs on the AI RESPONSE, after generation. They can disagree,
because they are looking at different text.
"""
import json, os, sys
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from controlplane import ControlPlane, Interaction

st.set_page_config(page_title="ControlPlane.ai", page_icon="🛡️", layout="wide")
PURPLE = "#A100FF"
ACTIONS = ["allow", "edit", "review", "block"]
ACTION_COLOR = {"allow": "#1FA36B", "edit": "#E39A1C", "review": "#E39A1C", "block": "#D24545"}
ACTION_TINT = {"allow": "#E8F8F1", "edit": "#FDF3E3", "review": "#FDF3E3", "block": "#FBEAEA"}


def banner(label, action, sub=None):
    """One solid-color action banner plus a tinted, same-color detail strip
    directly under it, so the explanation reads as part of the verdict
    instead of fading into default gray caption text."""
    color, tint = ACTION_COLOR[action], ACTION_TINT[action]
    html = (f"<div style='padding:12px 16px;border-radius:8px 8px 0 0;"
            f"background:{color};color:white;font-weight:700'>{label}: {action.upper()}</div>")
    if sub:
        html += (f"<div style='padding:10px 16px;border-radius:0 0 8px 8px;"
                 f"background:{tint};color:#333;border-left:4px solid {color};"
                 f"font-size:14px'>{sub}</div>")
    st.markdown(html, unsafe_allow_html=True)


@st.cache_resource
def engine():
    return ControlPlane("config/policies.yaml")


@st.cache_data
def samples():
    rows = {}
    with open("data/interactions.jsonl", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            rows[f"[{r['category']}] {r['response'][:55]}..."] = r
    return rows


st.markdown(f"<h1 style='color:{PURPLE}'>🛡️ ControlPlane.ai</h1>"
            "<p>A real-time control layer that scores every AI response for "
            "<b>performance</b>, <b>cost</b> and <b>responsibility</b> risk, "
            "then allows, edits, reviews, or blocks it.</p>", unsafe_allow_html=True)

cp = engine()
data = samples()

if "gate_result" not in st.session_state:
    st.session_state.gate_result = None
if "decision_result" not in st.session_state:
    st.session_state.decision_result = None

left, right = st.columns([1, 1])
with left:
    pick = st.selectbox("Load a sample interaction", ["Custom"] + list(data.keys()))
    r = data[pick] if pick != "Custom" else None

    st.subheader("1. Query, checked by the input gate before generation")
    c1, c2, c3 = st.columns(3)
    with c1:
        use_case = st.selectbox("Use case", ["customer_facing", "internal_copilot", "regulated_decision"],
                                index=["customer_facing", "internal_copilot", "regulated_decision"].index(r["use_case"]) if r else 0)
    with c2:
        jurisdiction = st.selectbox("Jurisdiction", ["US", "EU", "IN"])
    with c3:
        sector = st.selectbox("Sector", ["general", "healthcare", "finance"])

    EXAMPLE_QUERIES = {
        "Custom or from sample": None,
        "Clean question": "How do refunds work?",
        "Query containing PII": "My email is jane.doe@examplemail.com and my number is 9876543210, please cancel my order",
        "Toxic query": "You are such an idiot, this bot is useless, I hate dealing with you",
    }
    example_pick = st.selectbox("Try an example query", list(EXAMPLE_QUERIES.keys()))
    default_query = EXAMPLE_QUERIES[example_pick] or (r["query"] if r else "How do refunds work?")
    query = st.text_input("User query", value=default_query, key=f"query_{pick}_{example_pick}")

    if st.button("Check input gate"):
        st.session_state.gate_result = cp.check_input(
            query, use_case=use_case, jurisdiction=jurisdiction, sector=sector)

    if st.session_state.gate_result is not None:
        gate = st.session_state.gate_result
        sub = ("Rules fired: " + ", ".join(gate.fired_rules)) if gate.fired_rules else \
              "No rule fired for this use case, jurisdiction, and sector combination."
        sub += "<br>This checks the query only. It cannot see the AI response below."
        banner("Gate (query stage)", gate.action, sub)

    st.divider()
    st.subheader("2. AI response, checked by the output pipeline after generation")
    response = st.text_area("AI response", r["response"] if r else
                            "You can return items any time, no receipt needed.", height=120)
    context = st.text_area("Source context (for grounding)",
                           r.get("context", "") if r else
                           "Refunds are allowed within 30 days of delivery with a receipt.", height=80)
    if st.button("Run ControlPlane (output side)", type="primary"):
        x = Interaction(id="live", use_case=use_case, query=query, response=response,
                        context=context, samples=[response], model_used="large",
                        jurisdiction=jurisdiction, sector=sector)
        st.session_state.decision_result = cp.process(x, log=True)

with right:
    st.subheader("3. Decision on the response")
    d = st.session_state.decision_result
    if d is None:
        st.info("Pick a sample or write your own, then click Run ControlPlane (output side).")
    else:
        banner("Final decision", d.action, "This checks the AI response, independent of the gate on the left.")

        m1, m2 = st.columns(2)
        m1.metric("P(harm)", f"{d.p_harm:.3f}",
                  help="Calibrated probability that this response is harmful if served.")
        m2.metric("Overall risk", f"{d.overall_risk:.2f}",
                  help="Weighted per-dimension score, diagnostic, not the decision input.")

        if d.expected_loss:
            st.caption("Expected loss per action. The cost-model pick is the argmin, "
                       "before any hard rule can override it.")
            best = min(d.expected_loss, key=d.expected_loss.get)
            st.dataframe(
                [{"action": a, "E[loss]": round(v, 2),
                  "": "◀ cost-model pick" if a == best else ""}
                 for a, v in d.expected_loss.items()],
                hide_index=True, use_container_width=True)
            if ACTIONS.index(d.action) > ACTIONS.index(best):
                st.markdown(
                    f"<div style='padding:10px 14px;border-radius:8px;background:#FDF3E3;"
                    f"border-left:4px solid #E39A1C;font-size:14px;color:#333'>"
                    f"The cost model alone would have picked <b>{best}</b> here. A hard rule "
                    f"fired below and escalated the final decision to <b>{d.action.upper()}</b>. "
                    f"Hard rules always override cost optimisation.</div>", unsafe_allow_html=True)
            th = d.thresholds_used
            st.caption(f"Bands derived from this use case's cost model: "
                      f"edit >= {th['edit']:.3f}, review >= {th['review']:.3f}, block >= {th['block']:.3f}")

        st.caption("Per-dimension risk")
        for dim, val in d.risk_scores.items():
            st.progress(min(1.0, val), text=f"{dim}: {val:.2f}")

        st.caption("Why")
        for reason in d.reasons:
            st.write("-", reason)
        if d.fired_rules:
            st.warning("Rules fired: " + ", ".join(d.fired_rules))

        st.subheader("4. Human override (feedback loop)")
        new = st.selectbox("Override action", ACTIONS, index=ACTIONS.index(d.action))
        if st.button("Log override"):
            cp.audit.record_override(d.interaction_id, d.action, new, reviewer="demo_user")
            st.success(f"Override logged: {d.action} to {new} (written to audit trail)")

st.divider()
st.caption("Every decision and override is appended to data/audit_log.jsonl, "
          "the audit trail behind governance and the learning loop.")