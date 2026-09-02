# ControlPlane.ai

A real-time control layer for enterprise AI that evaluates requests before generation, scores responses across performance, responsibility, and cost, and selects **ALLOW, EDIT, REVIEW, or BLOCK** using configurable policy and expected-loss minimisation.

ControlPlane.ai is designed as a general-purpose AI control layer rather than a RAG-specific checker. It operates at the input/output layer, making it suitable for API-based foundation models where model internals are not directly accessible.

## Table of contents

* [Problem](#problem)
* [Solution](#solution)
* [How it works](#how-it-works)
* [Statistical decision framework](#statistical-decision-framework)
* [Risk dimensions](#risk-dimensions)
* [Statistical anomaly detection](#statistical-anomaly-detection)
* [AI-as-judge second opinion](#ai-as-judge-second-opinion)
* [Policy and governance](#policy-and-governance)
* [Multi-turn and session control](#multi-turn-and-session-control)
* [Monitoring and feedback](#monitoring-and-feedback)
* [Evaluation](#evaluation)
* [Project structure](#project-structure)
* [Requirements](#requirements)
* [Installation](#installation)
* [Running the application](#running-the-application)
* [Running the evaluation suite](#running-the-evaluation-suite)
* [Configuration](#configuration)
* [Troubleshooting](#troubleshooting)

---

## Problem

Enterprise AI systems operate under different risk tolerances, latency budgets, data environments, and governance requirements.

A customer-facing assistant, an internal knowledge copilot, and a regulated decision-support system should not necessarily apply the same control threshold.

At the same time:

* Hallucination, privacy, bias, and operational risks can overlap.
* Reliable real-time ground truth is often unavailable.
* Over-flagging creates alert fatigue.
* Under-flagging creates liability.
* Multi-turn interactions can compound exposure.
* AI agents can propagate one questionable response into downstream actions.
* Regulatory requirements vary across jurisdictions and sectors.
* Foundation models are often consumed through APIs, limiting access to model internals.

The core problem is therefore not simply:

> "Is this response risky?"

It is:

> **"Given the evidence, uncertainty, context, and cost of being wrong, what should the control layer do?"**

---

## Solution

ControlPlane.ai treats AI guardrailing as a **decision problem rather than only a scoring problem**.

For each interaction, the system:

1. Applies a pre-generation input gate.
2. Generates the response when the request is permitted.
3. Evaluates the response across three control dimensions.
4. Applies an AI-as-judge second opinion alongside the primary grounding score.
5. Calibrates the performance-related probability estimate.
6. Detects statistical behaviour outside validated operating profiles.
7. Applies use-case, jurisdiction, and sector policies.
8. Minimises expected loss to select an action.
9. Records the decision in an audit trail.
10. Monitors cumulative session exposure and population-level drift.

The resulting action is one of:

**ALLOW | EDIT | REVIEW | BLOCK**

The system is designed so that the business specifies the costs of different outcomes, while the policy engine derives the corresponding decision bands.

---

## How it works

```text
                         USER REQUEST
                              |
                              v
                    +-------------------+
                    |    INPUT GATE     |
                    | use case          |
                    | jurisdiction      |
                    | sector            |
                    +---------+---------+
                              |
                    +---------+---------+
                    |                   |
                  BLOCK               ALLOW
                    |                   |
                    v                   v
                  AUDIT          FOUNDATION MODEL
                                        |
                                        v
                         +-------------------------+
                         |   RESPONSE CONTROL      |
                         |                         |
                         | Performance             |
                         |  - TF-IDF grounding     |
                         |  - AI-as-judge (async)  |
                         |  - Self-consistency     |
                         |  - Calibration          |
                         |  - Anomaly detection    |
                         |                         |
                         | Responsibility          |
                         | Cost                    |
                         +------------+------------+
                                      |
                                      v
                            RISK + UNCERTAINTY
                                      |
                                      v
                           +---------------------+
                           |    POLICY ENGINE    |
                           |                     |
                           | use case            |
                           | x                   |
                           | jurisdiction        |
                           | x                   |
                           | sector              |
                           +----------+----------+
                                      |
                                      v
                            EXPECTED-LOSS ACTION
                                      |
                     +----------------+----------------+
                     |                |                |
                     v                v                v
                   ALLOW         EDIT / REVIEW        BLOCK
                                      |
                                      v
                              AUDIT + MONITORING
```

### Pre-generation input control

The input gate operates before the foundation model is called.

A request is evaluated against the configured governance context:

* use case
* jurisdiction
* sector

If the input gate produces a blocking decision, generation can be skipped entirely.

Input decisions are recorded using the same audit mechanism as response decisions.

### Post-generation response control

Once a response is generated, the three control dimensions run concurrently.

Each dimension produces its own evidence and risk signal. The policy engine then combines these signals with the configured decision costs.

The architecture keeps detection separate from policy so that individual detectors can be replaced without redesigning the decision layer.

---

## Statistical decision framework

A central design principle of ControlPlane.ai is to separate **risk estimation**, **uncertainty**, and **action selection**.

The system first estimates the probability of harm from detector evidence:

$$
P(\mathrm{harm}\mid x)
$$

The policy layer then considers the consequences of each available action.

The selected action is:

$$
a^* =
\arg\min_a
E[L(a,Y)\mid x]
$$

where:

* \(a\) is an available control action
* \(Y\) represents the underlying state of the response
* \(L(a,Y)\) is the configured loss associated with taking action \(a\) when the true state is \(Y\)

This means that ControlPlane.ai does not rely on a single manually selected probability threshold.

Instead, decision bands are derived from the stated business costs.

For example, the same estimated harm probability can lead to different actions for an internal copilot and a regulated decision-support system because the cost of serving an incorrect answer is different.

### Hard governance rules

Some risks should not be traded against ordinary business costs.

Hard rules can therefore escalate a decision regardless of the expected-loss calculation.

Examples include configured PII and toxicity rules for high-risk contexts.

This creates two layers of control:

```text
Expected-loss policy
        |
        v
Ordinary risk trade-offs

Hard governance rules
        |
        v
Non-negotiable escalation
```

---

## Risk dimensions

### 1. Performance

The performance control evaluates whether the response is supported by the available context and whether its behaviour is internally consistent.

It includes:

* Context grounding via TF-IDF claim-level faithfulness scoring (validated on RAGTruth)
* AI-as-judge LLM second opinion (see dedicated section below)
* Self-consistency checks across resampled generations
* Calibrated hallucination probability
* Statistical anomaly detection on response structure and support profile

The performance probability is calibrated using held-out data rather than treating the raw detector score as a directly interpretable probability.

### 2. Responsibility

The responsibility control focuses on risks that can create privacy, safety, or fairness exposure.

It includes:

* Email detection
* Indian phone number detection
* PAN detection
* Aadhaar detection
* Payment card detection
* SSN detection
* Toxicity and bias signals
* Statistical monitoring of PII and toxicity rates

Optional integrations can provide stronger specialised detection where available.

### 3. Cost

The cost control evaluates the operational cost associated with generating and serving responses.

It includes:

* Token estimation
* Illustrative model-serving cost
* Response verbosity
* Regeneration and rework behaviour
* Conditional response-length modelling
* Regeneration-rate anomaly detection

Cost is treated as part of the decision problem rather than simply as a reporting metric.

---

## Statistical anomaly detection

Statistical anomaly detection is integrated into the three control dimensions rather than treated as a separate risk category.

The purpose of an anomaly signal is to identify behaviour that is unusual relative to the validated operating profile.

An unusual observation is **not automatically considered harmful**.

Instead:

```text
Observed behaviour
       |
       v
Statistical anomaly detection
       |
       v
Distributional uncertainty
       |
       v
Policy decision
```

This distinction is important because unusual behaviour can be legitimate.

ControlPlane.ai therefore uses anomaly evidence to widen uncertainty around the decision rather than simply adding anomaly severity to the estimated harm probability.

### Performance anomaly model

The performance profile uses response and support features including number of claims, mean and minimum support, support variability, fraction unsupported, log response length, type-token ratio, digit density, sentence length, and repetition behaviour.

The profile combines response-length modelling with multivariate distance using the squared Mahalanobis statistic.

### Responsibility anomaly model

Responsibility monitoring tracks population behaviour: PII occurrence rates, toxicity occurrence rates, and population-rate drift via EWMA control charts.

### Cost anomaly model

Cost monitoring evaluates response length conditional on query length (log-token residuals) and regeneration frequency via Poisson tail probabilities.

### Held-out anomaly validation

The anomaly profiles were fitted using 3,470 clean QA-train responses against 740 held-out clean QA responses at a nominal significance level of 1%.

The held-out clean-data flag rate was **0.81%**: close to the nominal level, confirming the profile is not over-flagging clean observations.

---

## AI-as-judge second opinion

ControlPlane includes an LLM-based grounding auditor that runs alongside the primary TF-IDF faithfulness scorer as a second opinion.

For each response with available context, the judge is given the full context and response and returns a structured JSON verdict:

```json
{
  "verdict":            "supported | unsupported | contradicted",
  "unsupported_claims": ["specific claim text"],
  "contradicted_claims": ["specific claim text"],
  "reasoning":          "one-line explanation",
  "confidence":         0.99
}
```

The judge can do something the TF-IDF and NLI backends cannot: it names the **specific claims** it doubts and gives a **plain-English reason**, making the audit trail inspectable rather than just a score.

### Architecture

The judge runs asynchronously inside the performance detector, it never adds to the inline latency the user waits for. Its verdict lands in `DetectorResult.detail["judge"]` and is recorded in every audit trail entry.

The judge does **not** replace the calibrated TF-IDF risk used for the decision. The TF-IDF score is validated on RAGTruth and sits on a calibrated probability scale. The judge's verdict is a qualitative second opinion, not a calibrated probability, and mixing the two would corrupt the expected-loss arithmetic. The correct channel is the audit trail.

### `judge_overrides_tfidf` flag

When TF-IDF scores a response as acceptable but the judge returns `unsupported` or `contradicted`, a `judge_overrides_tfidf` flag is set on the decision. These are the highest-priority cases for human review which the primary model missed something the judge caught.

In the evaluation demo, TF-IDF scored a fabricated South region revenue figure at 0.611 (below the block threshold). The judge identified it as unsupported with 0.99 confidence and named the exact claim. `judge_overrides_tfidf=True` was set. Judge latency: approximately 660ms on Groq's free tier.

Note: `judge_overrides_tfidf` flags *disagreement*, not correctness. In our
evaluation, the judge was sometimes stricter than TF-IDF (e.g., treating an
inferred connection between two facts as unsupported when a human reader would
accept it). This is expected and desirable: the flag exists so a human makes
the final call on genuinely ambiguous cases, rather than the system silently
picking one AI's judgment over another's.

### Overlap detection

When a response is simultaneously hallucinated and contains PII, ControlPlane sets a `hallucination_pii_overlap` flag and records `overlap:hallucination_pii` in the fired rules. This names the joint incident in the audit trail so it can be routed to a different escalation path from either condition alone.

### Fail-open design

If `GROQ_API_KEY` is absent, the `groq` package is not installed, or the API call exceeds the timeout, the judge logs `verdict: unavailable` and the pipeline continues with TF-IDF as the sole score. No decisions change.

### A known reasoning-model failure mode

`openai/gpt-oss-20b` is a chain-of-thought model: it spends part of its token
budget on hidden internal reasoning before writing the final answer. On some
inputs it can spend the *entire* budget reasoning and return empty content
with `finish_reason: "length"`, not an error, just nothing written. This is
a documented Groq/gpt-oss behaviour, not a bug in this codebase.

The judge handles it explicitly: `reasoning_effort="low"` reduces how much
budget the model spends reasoning, the token budget is set generously (1024,
retried at 2048), and if the model still returns nothing, the verdict is
logged as `reasoning_exhausted` rather than a generic failure — so the audit
trail states plainly what happened.

### Setup

```bash
pip install groq
# Free key at https://console.groq.com (30 seconds to sign up)
$env:GROQ_API_KEY = "your_key_here"   # PowerShell
python scripts/test_judge.py
```

### Backend selection

```bash
CONTROLPLANE_GROUNDING=tfidf          # TF-IDF only (default, offline)
CONTROLPLANE_GROUNDING=judge          # judge only
CONTROLPLANE_GROUNDING=tfidf+judge    # TF-IDF primary + judge second opinion
```

---

## Policy and governance

Policy configuration is stored in `config/policies.yaml`.

The policy layer separates business context from detector implementation.

Policies can vary across use case, jurisdiction, sector, risk appetite, latency budget, action costs, and hard governance rules.

### Three-axis composition

The policy engine composes three independent axes:

```text
use_case  x  jurisdiction  x  sector
```

Each axis contributes hard rules that escalate the action independently. A healthcare deployment in the EU under a regulated decision use case fires all three axes simultaneously. Rules from each axis are tagged separately in the audit trail.

### Current use cases

**Customer-facing** — lower tolerance for harmful responses, tighter response-time requirements.

**Internal copilot** — wider trade-off because an expert user can review or correct outputs before downstream use.

**Regulated decision** — substantially higher costs for serving harmful responses, producing the most conservative decision behaviour.

### Jurisdictions

**EU** — EU AI Act. Human-in-the-loop obligation for uncertain grounding in high-risk use cases.

**IN** — India DPDP Act. PII disclosure is a hard stop across all use cases. Bulk PII disclosure is treated as a distinct incident.

**US** — Baseline. No additional jurisdiction-specific obligations beyond each use case.

### Sectors

**Healthcare** — Hallucination and PII overlap is explicitly handled. An ungrounded claim in a clinical context triggers review even where the use case alone would not.

**Finance** — Ungrounded claims feeding financial decisions carry the same operational-risk tail as the regulated decision use case.

**General** — No sector-specific obligations beyond use case and jurisdiction.

### Derived decision bands

Current decision-band analysis:

| Use case           | Cost ratio |  Edit | Review | Block |
| ------------------ | ---------: | ----: | -----: | ----: |
| Customer-facing    |       4.2x | 0.051 |  0.151 | 0.552 |
| Internal copilot   |       0.6x | 0.071 |  0.300 | 0.898 |
| Regulated decision |      33.3x | 0.008 |  0.017 | 0.300 |

These values are derived from the configured loss structure rather than manually entered probability thresholds.

---

## Multi-turn and session control

A response-level guardrail cannot capture all risks created by a sequence of individually acceptable responses.

ControlPlane.ai therefore maintains session-level exposure:

$$
1-\prod_{i=1}^{n}(1-p_i)
$$

This allows the system to detect situations where no individual response crosses a blocking threshold, but the probability of at least one harmful served response becomes significant.

Agent actions can receive additional exposure weighting because their outputs can feed downstream actions rather than simply being read by a user.

---

## Monitoring and feedback

ControlPlane.ai includes population-level monitoring in addition to response-level controls.

### Drift monitoring

The monitoring layer compares a live or simulated reference window using Population Stability Index and two-sample Kolmogorov-Smirnov test.

In the current simulated drift experiment, retrieval degradation injected from week 9 produced:

```text
PSI = 2.853
KS p-value < 0.0001
Verdict = MATERIAL SHIFT
```

### Feedback loop

The feedback module (`controlplane/feedback.py`) attributes each human override back to the specific fired rule that produced the decision being overridden and not just the action.

This is the distinction that matters operationally: two `review` decisions can be driven by completely different rules. A rule-level override rate tells you exactly which entry in `config/policies.yaml` is losing the desk's trust. The recommendation output names the axis (use_case / jurisdiction / sector) and the specific condition, pointing directly to the config edit required.

### Audit trail

Every decision, including input gate decisions, is recorded with: input and response, risk evidence, policy context, fired rules (tagged by axis), selected action, reasons, and session information.

---

## Evaluation

### RAGTruth benchmark

The performance/grounding component is evaluated on the RAGTruth benchmark (900 QA test responses, 17.8% hallucinated base rate).

Using the default TF-IDF grounding backend:

| Metric          | Result |
| --------------- | -----: |
| AUROC           |  0.708 |
| AUPRC           |  0.271 |
| AUPRC base rate |  0.178 |
| AUPRC lift      |  1.52x |

At the Youden-optimal threshold of 0.692:

| Metric    | Result |
| --------- | -----: |
| Precision |   0.29 |
| Recall    |   0.76 |
| F1        |   0.42 |

### Probability calibration

| Metric      |    Raw | Calibrated |
| ----------- | -----: | ---------: |
| Brier score | 0.3757 |     0.1518 |
| ECE         | 0.4830 |     0.1248 |

A reject-option experiment showed accuracy rising from 82% (full coverage) to 90% (60% automatic coverage), supporting use of human escalation for lower-confidence cases.

### Loss backtest

| Policy            | Mean realised loss |
| ----------------- | -----------------: |
| No guardrail      |            1217.67 |
| Block everything  |              54.83 |
| Static thresholds |             191.20 |
| ControlPlane      |          **26.67** |
| Oracle            |               0.00 |

ControlPlane captured **97.8% of the avoidable loss** between the unguarded system and the oracle benchmark.

### Prevalence sensitivity

| Policy                          |       2% |       5% |      10% |      50% |
| ------------------------------- | -------: | -------: | -------: | -------: |
| No guardrail                    |     37.5 |     93.7 |     187.3 |    936.7 |
| Block everything                |    153.5 |    148.8 |     141.0 |     78.3 |
| Static thresholds               |      6.8 |     15.6 |      30.2 |    147.3 |
| ControlPlane                    |     27.6 |     27.5 |      27.5 |     26.9 |
| ControlPlane + prior correction | **16.3** | **21.0** |  **28.2** | **30.1** |

### Fairness audit

| Attribute   | Counterfactual flip rate | Disparate impact | Four-fifths rule |
| ----------- | -----------------------: | ---------------: | ---------------- |
| Gender      |                     0.0% |            1.000 | PASS             |
| Region      |                     0.0% |            1.000 | PASS             |
| Name origin |                     2.8% |            0.938 | PASS             |

### Multi-turn exposure

An eight-turn session experiment demonstrated cumulative risk: no individual turn was blocked, but cumulative probability of at least one harmful served response reached approximately 19%, triggering a session-level control action.

---

## Project structure

```text
ControlPlane.ai/
│
├── app/
│   ├── api.py
│   ├── monitoring.py
│   └── streamlit_app.py
│
├── config/
│   └── policies.yaml
│
├── controlplane/
│   ├── audit.py
│   ├── calibration.py
│   ├── decision_theory.py
│   ├── detectors.py
│   ├── dimension_anomaly.py
│   ├── feedback.py
│   ├── grounding.py
│   ├── input_gate.py
│   ├── judge.py
│   ├── llm.py
│   ├── pipeline.py
│   ├── policy.py
│   ├── router.py
│   ├── schemas.py
│   ├── scoring.py
│   ├── session.py
│   └── stats.py
│
├── data/
│   ├── anomaly_profiles.json
│   ├── calibrator.json
│   └── ragtruth/
│
├── scripts/
│   ├── fit_profiles.py
│   ├── calibrate.py
│   ├── evaluate.py
│   ├── evaluate_ragtruth.py
│   ├── loss_backtest.py
│   ├── threshold_sensitivity.py
│   ├── fairness_audit.py
│   ├── drift_monitor.py
│   ├── session_demo.py
│   ├── cost_frontier.py
│   ├── feedback_report.py
│   ├── test_input_gate.py
│   ├── test_jurisdiction.py
│   ├── test_sector.py
│   └── test_judge.py
│
└── README.md
```

### Core modules

| Module                 | Purpose                                                           |
| ---------------------- | ----------------------------------------------------------------- |
| `pipeline.py`          | Orchestrates input control, response evaluation, policy and audit |
| `input_gate.py`        | Pre-generation request control                                    |
| `detectors.py`         | Performance, responsibility and cost controls                     |
| `judge.py`             | AI-as-judge LLM grounding second opinion                          |
| `dimension_anomaly.py` | Statistical anomaly profiles for the control dimensions           |
| `calibration.py`       | Probability calibration and OOD interval widening                 |
| `decision_theory.py`   | Harm probability and expected-loss calculations                   |
| `policy.py`            | Three-axis policy (use case × jurisdiction × sector)              |
| `session.py`           | Cumulative multi-turn exposure                                    |
| `audit.py`             | Decision logging                                                  |
| `feedback.py`          | Rule-level feedback and override attribution                      |
| `monitoring.py`        | Population-level drift monitoring                                 |
| `grounding.py`         | TF-IDF, embedding, and NLI grounding backends                     |
| `stats.py`             | Statistical kernel: Mahalanobis, EWMA, Poisson, residual models   |

---

## Requirements

Python virtual environment. Primary dependencies: scikit-learn, numpy, pandas, fastapi, streamlit, pyyaml.

Optional: `groq` (AI-as-judge), `transformers` + `torch` (NLI and embedding backends), `sentence-transformers` (embedding backend).

---

## Installation

```bash
git clone https://github.com/manyagupta-21/ControlPlane.ai.git
cd ControlPlane.ai
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

---

## Running the application

### Start the FastAPI service

```bash
uvicorn app.api:app --reload
```

### Start the Streamlit interface

The AI-as-judge feature requires a Groq API key. Set `GROQ_API_KEY` before starting Streamlit.

```powershell
$env:GROQ_API_KEY = "your_free_key"
streamlit run app/streamlit_app.py
```

### Typical workflow

```text
1. Select or write an interaction
2. Apply the input gate
3. Generate or provide the AI response
4. Run ControlPlane
5. Inspect performance, responsibility and cost evidence
6. Inspect anomaly information and judge verdict
7. View the selected action
8. Inspect the recorded decision information
```

---

## Running the evaluation suite

### Fit anomaly profiles

```bash
python scripts/fit_profiles.py
```

### Calibrate the performance probability

```bash
python scripts/calibrate.py
```

### Evaluate on RAGTruth

```bash
python scripts/evaluate_ragtruth.py
```

### Evaluate the full control layer

```bash
python scripts/evaluate.py
```

### Backtest realised policy loss

```bash
python scripts/loss_backtest.py
```

### Evaluate decision-band sensitivity

```bash
python scripts/threshold_sensitivity.py
```

### Run the fairness audit

```bash
python scripts/fairness_audit.py
```

### Monitor population drift

```bash
python scripts/drift_monitor.py
```

### Demonstrate multi-turn exposure

```bash
python scripts/session_demo.py
```

### Evaluate the cost frontier

```bash
python scripts/cost_frontier.py
```

### Run the feedback report

```bash
python scripts/simulate_traffic.py --reset   # generate audit data
python scripts/feedback_report.py            # rule-level override attribution
```

### Test the AI-as-judge

```bash
pip install groq
$env:GROQ_API_KEY = "your_free_key"   # from https://console.groq.com
python scripts/test_judge.py
```

Without `GROQ_API_KEY` the demo runs in fail-open mode showing `verdict: unavailable`. The pipeline continues normally in all cases.

### Test governance axes

```bash
python scripts/test_input_gate.py    # use-case-aware pre-generation gate
python scripts/test_jurisdiction.py  # EU / IN / US jurisdiction composition
python scripts/test_sector.py        # healthcare / finance sector composition
```

---

## Configuration

The main policy configuration is `config/policies.yaml`.

The important design principle: **business costs are inputs to the policy rather than manually selected probability thresholds**. When the cost assumptions change, the corresponding decision bands are recomputed automatically.

---

## Troubleshooting

### FastAPI starts but `/` returns 404

The API service is not required to expose a root webpage. Use the API routes in `app/api.py` or the Streamlit interface.

### PowerShell blocks virtual-environment activation

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1
```

### Evaluation outputs are missing

Run the profile and calibration steps first:

```bash
python scripts/fit_profiles.py
python scripts/calibrate.py
```

### AI-as-judge returns `unavailable`

Install the Groq package and set your API key:

```bash
pip install groq
$env:GROQ_API_KEY = "your_key"   # free key at https://console.groq.com
```

Without the key the pipeline continues normally with TF-IDF as the sole grounding score.

### Toxicity backend selection

```powershell
$env:CONTROLPLANE_TOXICITY = "lexicon"    # default, offline
$env:CONTROLPLANE_TOXICITY = "detoxify"   # requires pip install detoxify
```

### Grounding backend selection

```powershell
$env:CONTROLPLANE_GROUNDING = "tfidf"         # default, offline
$env:CONTROLPLANE_GROUNDING = "tfidf+judge"   # TF-IDF + LLM judge (requires groq)
$env:CONTROLPLANE_GROUNDING = "nli"           # NLI cross-encoder (CPU: ~5s/response)
```

---

## Maintainers

- Niraj Mhatre, M.Sc. Statistics, IIT Kanpur
- Manya Gupta, M.Sc. Statistics, IIT Kanpur

ControlPlane.ai was developed for the **Accenture Innovation Challenge 2026, Round 2, Problem Track 1**.

Licensed under the [MIT License](LICENSE).
