"""ControlPlane demo UI.

Run locally:  streamlit run app/streamlit_app.py

Lets you push any AI response through the control layer, see each detector's
risk, the tiered decision (and why), and capture a human override that is
written back to the audit trail — closing the feedback loop live on screen.
"""
import json, os, sys
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from controlplane import ControlPlane, Interaction

st.set_page_config(page_title="ControlPlane.ai", page_icon="🛡️", layout="wide")
PURPLE = "#A100FF"
ACTION_COLOR = {"allow": "#1FA36B", "edit": "#E39A1C", "review": "#E39A1C", "block": "#D24545"}

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
            "<b>performance</b>, <b>cost</b> and <b>responsibility</b> risk — "
            "then allows, edits, reviews, or blocks it.</p>", unsafe_allow_html=True)

cp = engine()
data = samples()

left, right = st.columns([1, 1])
with left:
    st.subheader("1 · Choose or write a response")
    pick = st.selectbox("Load a sample interaction", ["— custom —"] + list(data.keys()))
    if pick != "— custom —":
        r = data[pick]
        use_case = st.selectbox("Use case", ["customer_facing", "internal_copilot", "regulated_decision"],
                                index=["customer_facing", "internal_copilot", "regulated_decision"].index(r["use_case"]))
        query = st.text_input("User query", r["query"])
        response = st.text_area("AI response", r["response"], height=120)
        context = st.text_area("Source context (for grounding)", r.get("context", ""), height=80)
    else:
        use_case = st.selectbox("Use case", ["customer_facing", "internal_copilot", "regulated_decision"])
        query = st.text_input("User query", "How do refunds work?")
        response = st.text_area("AI response", "You can return items any time, no receipt needed.", height=120)
        context = st.text_area("Source context (for grounding)",
                               "Refunds are allowed within 30 days of delivery with a receipt.", height=80)
    go = st.button("Run ControlPlane", type="primary")

with right:
    st.subheader("2 · Decision")
    if go:
        x = Interaction(id="live", use_case=use_case, query=query, response=response,
                        context=context, samples=[response], model_used="large")
        d = cp.process(x, log=True)
        c = ACTION_COLOR[d.action]
        st.markdown(f"<div style='padding:14px;border-radius:10px;background:{c};color:white;"
                    f"font-size:22px;font-weight:700'>Decision: {d.action.upper()}</div>",
                    unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("P(harm)", f"{d.p_harm:.3f}",
                  help="Calibrated probability that this response is harmful if served.")
        m2.metric("Overall risk", f"{d.overall_risk:.2f}",
                  help="Weighted per-dimension score (diagnostic, not the decision input).")

        if d.expected_loss:
            st.caption("Expected loss per action — the decision is the argmin, not a threshold")
            best = min(d.expected_loss, key=d.expected_loss.get)
            st.dataframe(
                [{"action": a, "E[loss]": round(v, 2),
                  "": "◀ chosen" if a == best else ""}
                 for a, v in d.expected_loss.items()],
                hide_index=True, use_container_width=True)
            th = d.thresholds_used
            st.caption(
                f"Bands derived from this use case's cost model — "
                f"edit ≥ {th['edit']:.3f} · review ≥ {th['review']:.3f} · block ≥ {th['block']:.3f}")
        st.caption("Per-dimension risk")
        for dim, val in d.risk_scores.items():
            st.progress(min(1.0, val), text=f"{dim}: {val:.2f}")
        st.caption("Why")
        for reason in d.reasons:
            st.write("•", reason)
        if d.fired_rules:
            st.warning("Rules fired: " + ", ".join(d.fired_rules))

        st.subheader("3 · Human override (feedback loop)")
        new = st.selectbox("Override action", ["allow", "edit", "review", "block"],
                           index=["allow", "edit", "review", "block"].index(d.action))
        if st.button("Log override"):
            cp.audit.record_override("live", d.action, new, reviewer="demo_user")
            st.success(f"Override logged: {d.action} → {new} (written to audit trail)")
    else:
        st.info("Pick a sample or write your own, then click **Run ControlPlane**.")

st.divider()
st.caption("Every decision and override is appended to data/audit_log.jsonl — "
           "the audit trail behind governance and the learning loop.")
