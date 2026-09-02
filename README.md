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
* [Team](#team)

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

1. Applies a **pre-generation input gate**.
2. Generates the response when the request is permitted.
3. Evaluates the response across three control dimensions.
4. Calibrates the performance-related probability estimate.
5. Detects statistical behaviour outside validated operating profiles.
6. Applies use-case, jurisdiction, and sector policies.
7. Minimises expected loss to select an action.
8. Records the decision in an audit trail.
9. Monitors cumulative session exposure and population-level drift.

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
                         | Responsibility           |
                         | Cost                    |
                         |                         |
                         | Statistical anomaly     |
                         | detection within each  |
                         | dimension              |
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

* Context grounding
* TF-IDF based support scoring
* Self-consistency checks
* Calibrated hallucination probability
* Statistical anomaly detection

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

For example:

* A correct response may be unusually long.
* A legitimate query may have an unusual structure.
* A rare user request may not represent harmful behaviour.
* A new but valid operating condition may fall outside the historical profile.

ControlPlane.ai therefore uses anomaly evidence to widen uncertainty around the decision rather than simply adding anomaly severity to the estimated harm probability.

### Performance anomaly model

The performance profile uses response and support features such as:

* Number of claims
* Mean support
* Minimum support
* Support variability
* Fraction unsupported
* Support range
* Log response length
* Type-token ratio
* Digit density
* Sentence length
* Repetition behaviour

The profile combines response-length modelling with multivariate distance using the squared Mahalanobis statistic.

### Responsibility anomaly model

Responsibility monitoring tracks population behaviour such as:

* PII occurrence rates
* Toxicity occurrence rates
* Population-rate drift

### Cost anomaly model

Cost monitoring evaluates:

* Response length conditional on query length
* Log-token residuals
* Regeneration frequency

### Held-out anomaly validation

The anomaly profiles were fitted using:

* **3,470 clean QA-train responses**
* **740 held-out clean QA responses**
* Nominal significance level: **1%**

The held-out clean-data flag rate was:

**0.81%**

A realised rate close to the nominal significance level provides a useful sanity check that the statistical profile is not excessively over-flagging clean observations.

---

## Policy and governance

Policy configuration is stored in:

```text
config/policies.yaml
```

The policy layer separates business context from detector implementation.

Policies can vary across:

* Use case
* Jurisdiction
* Sector
* Risk appetite
* Latency budget
* Action costs
* Hard governance rules

The current prototype includes:

### Customer-facing

Designed for lower tolerance for harmful responses and tighter response-time requirements.

### Internal copilot

Allows a different trade-off because an expert user can review or correct outputs before downstream use.

### Regulated decision

Uses substantially higher costs for serving harmful responses, producing much more conservative decision behaviour.

### Derived decision bands

Current decision-band analysis gives:

| Use case           | Cost ratio |  Edit | Review | Block |
| ------------------ | ---------: | ----: | -----: | ----: |
| Customer-facing    |       4.2x | 0.051 |  0.151 | 0.552 |
| Internal copilot   |       0.6x | 0.071 |  0.300 | 0.898 |
| Regulated decision |      33.3x | 0.008 |  0.017 | 0.300 |

These values are derived from the configured loss structure rather than manually entered probability thresholds.

---

## Multi-turn and session control

A response-level guardrail cannot capture all risks created by a sequence of individually acceptable responses.

ControlPlane.ai therefore maintains session-level exposure.

For a sequence of served responses with estimated harmful probabilities \(p_1,\ldots,p_n\), cumulative exposure is represented as:

$$
1-\prod_{i=1}^{n}(1-p_i)
$$

This allows the system to detect situations where no individual response crosses a blocking threshold, but the probability of at least one harmful served response becomes significant.

Agent actions can receive additional exposure weighting because their outputs can feed downstream actions rather than simply being read by a user.

---

## Monitoring and feedback

ControlPlane.ai includes population-level monitoring in addition to response-level controls.

### Drift monitoring

The monitoring layer compares a live or simulated reference window using:

* Population Stability Index
* Two-sample Kolmogorov-Smirnov test

In the current simulated drift experiment:

* Reference window: weeks 1 to 4
* Retrieval degradation injected from week 9
* First material distributional breach: **week 9**

At week 9:

```text
PSI = 2.853
KS p-value < 0.0001
Verdict = MATERIAL SHIFT
```

The monitoring layer therefore detects a material population shift before relying only on downstream harm observations.

### Audit trail

Decisions are recorded through the audit layer, providing information about:

* Input and response
* Risk evidence
* Policy context
* Selected action
* Reasons
* Session information

This supports post-hoc investigation and operational monitoring.

---

## Evaluation

ControlPlane.ai is a general AI control framework. Different components are therefore evaluated using datasets and experiments appropriate to the property being tested.

### RAGTruth benchmark

**The performance/grounding component is evaluated on the RAGTruth benchmark.**

RAGTruth provides externally labelled real LLM responses for evaluating hallucination and grounding behaviour.

The current evaluation uses:

* **900 RAGTruth QA test responses**
* Hallucinated-response base rate: **17.8%**

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

The benchmark demonstrates meaningful discrimination above random performance, while also showing that grounding remains an imperfect signal. The control layer therefore does not depend on hallucination detection alone.

### Probability calibration

The performance probability was evaluated before and after isotonic calibration.

| Metric      |    Raw | Calibrated |
| ----------- | -----: | ---------: |
| Brier score | 0.3757 |     0.1518 |
| ECE         | 0.4830 |     0.1248 |

Both metrics decreased substantially after calibration.

A reject-option experiment also showed:

| Automatic answer rate | Accuracy |
| --------------------: | -------: |
|                  100% |     0.82 |
|                   80% |     0.85 |
|                   60% |     0.90 |

This supports the use of human escalation for lower-confidence cases.

### Loss backtest

The policy is evaluated by realised loss after the true state of each interaction is known.

| Policy            | Mean realised loss |
| ----------------- | -----------------: |
| No guardrail      |            1217.67 |
| Block everything  |              54.83 |
| Static thresholds |             191.20 |
| ControlPlane      |          **26.67** |
| Oracle            |               0.00 |

The current cost-derived ControlPlane policy captured:

**97.8% of the avoidable loss between the unguarded system and the oracle benchmark.**

The block-everything baseline is intentionally included. It demonstrates why minimising harm alone is insufficient: a system can reduce loss by refusing to serve anything while destroying the utility of the AI system.

### Prevalence sensitivity

The evaluation dataset has a much higher harmful-response prevalence than expected live traffic.

The prototype therefore evaluates the effect of different harmful base rates:

| Policy                          |       2% |       5% |      10% |      50% |
| ------------------------------- | -------: | -------: | -------: | -------: |
| No guardrail                    |     37.5 |     93.7 |    187.3 |    936.7 |
| Block everything                |    153.5 |    148.8 |    141.0 |     78.3 |
| Static thresholds               |      6.8 |     15.6 |     30.2 |    147.3 |
| ControlPlane                    |     27.6 |     27.5 |     27.5 |     26.9 |
| ControlPlane + prior correction | **16.3** | **21.0** | **28.2** | **30.1** |

This makes the prevalence assumption explicit rather than implicitly treating an evaluation-set class balance as production reality.

### Fairness audit

The prototype includes a counterfactual fairness audit using matched responses where one protected-attribute token is changed while the underlying response meaning remains the same.

Results:

| Attribute   | Counterfactual flip rate | Disparate impact | Four-fifths rule |
| ----------- | -----------------------: | ---------------: | ---------------- |
| Gender      |                     0.0% |            1.000 | PASS             |
| Region      |                     0.0% |            1.000 | PASS             |
| Name origin |                     2.8% |            0.938 | PASS             |

The name-origin experiment identified a small but non-zero decision flip rate despite a very small mean probability difference.

This demonstrates an important property of the audit: a detector can be nearly invariant in score while a small perturbation near a decision boundary can still change the final action.

### Multi-turn exposure

An eight-turn session experiment demonstrates cumulative risk.

No individual turn was blocked, but cumulative probability of at least one harmful served response reached approximately **19%**.

The session controller detected increasing cumulative exposure and triggered a session-level control action.

### Robustness

The decision layer was also tested under a detector substitution.

Replacing the toxicity component with Detoxify changed the ControlPlane loss-recovery result only marginally, while the static-threshold baseline changed more substantially.

This supports the design principle of separating detector outputs from policy logic.

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
│   ├── drift/
│   ├── fairness/
│   ├── session/
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
│   └── cost_frontier.py
│
└── README.md
```

### Core modules

| Module                 | Purpose                                                           |
| ---------------------- | ----------------------------------------------------------------- |
| `pipeline.py`          | Orchestrates input control, response evaluation, policy and audit |
| `input_gate.py`        | Pre-generation request control                                    |
| `detectors.py`         | Performance, responsibility and cost controls                     |
| `dimension_anomaly.py` | Statistical anomaly profiles for the control dimensions           |
| `calibration.py`       | Probability calibration                                           |
| `decision_theory.py`   | Harm probability and expected-loss calculations                   |
| `policy.py`            | Context-specific policy and action selection                      |
| `session.py`           | Cumulative multi-turn exposure                                    |
| `audit.py`             | Decision logging                                                  |
| `feedback.py`          | Feedback and override handling                                    |
| `monitoring.py`        | Population-level monitoring                                       |
| `grounding.py`         | Context grounding logic                                           |

---

## Requirements

The prototype is implemented in Python and uses standard machine learning, statistical modelling, API, and dashboard libraries.

The primary environment is a Python virtual environment.

The default prototype is designed to support local, reproducible execution using the repository's included data and deterministic components.

---

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/manyagupta-21/ControlPlane.ai.git
cd ControlPlane.ai

python -m venv .venv
```

Activate the environment.

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the application

### Start the FastAPI service

```bash
uvicorn app.api:app --reload
```

The API runs locally on:

```text
http://127.0.0.1:8000
```

### Start the Streamlit interface

In a second terminal with the virtual environment activated:

```bash
streamlit run app/streamlit_app.py
```

The Streamlit interface provides the interactive ControlPlane workflow for testing sample or custom interactions.

### Typical workflow

```text
1. Select or write an interaction
2. Apply the input gate
3. Generate or provide the AI response
4. Run ControlPlane
5. Inspect performance, responsibility and cost evidence
6. Inspect anomaly information
7. View the selected action
8. Inspect the recorded decision information
```

---

## Running the evaluation suite

The repository includes scripts for reproducing the main statistical evaluations.

### Fit anomaly profiles

```bash
python scripts/fit_profiles.py
```

This fits the dimension-specific anomaly profiles and writes:

```text
data/anomaly_profiles.json
```

### Calibrate the performance probability

```bash
python scripts/calibrate.py
```

This evaluates calibration and writes:

```text
data/calibrator.json
```

### Evaluate on RAGTruth

```bash
python scripts/evaluate_ragtruth.py
```

This evaluates the performance/grounding component against the RAGTruth QA test set.

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

---

## Configuration

The main policy configuration is:

```text
config/policies.yaml
```

The policy file controls the business context used by the decision layer.

The current configuration distinguishes:

```text
customer_facing
internal_copilot
regulated_decision
```

and supports contextual variation through:

```text
use case
jurisdiction
sector
```

The configuration also contains:

* Risk weights
* Action costs
* Latency budgets
* Hard governance rules

The important design principle is that **business costs are inputs to the policy rather than manually selected probability thresholds**.

When the cost assumptions change, the corresponding decision bands are recomputed.

---

## Troubleshooting

### FastAPI starts but `/` returns 404

The API service is not required to expose a root webpage. A `404 Not Found` response at `/` does not necessarily indicate that the service has failed.

Use the API routes exposed by `app/api.py` or the Streamlit interface.

### PowerShell blocks virtual-environment activation

Run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Evaluation outputs are missing

Run the profile and calibration steps before the dependent evaluations:

```bash
python scripts/fit_profiles.py
python scripts/calibrate.py
```

This generates the required statistical artefacts under `data/`.

### Optional model dependencies

Some specialised detectors can use optional external libraries. The default prototype retains local components so that the core control pipeline can be demonstrated without requiring access to a proprietary foundation model or model internals.

---


## Maintainers

- Niraj Mhatre, M.Sc. Statistics, IIT Kanpur
- Manya Gupta, M.Sc. Statistics, IIT Kanpur

ControlPlane.ai was developed for the **Accenture Innovation Challenge 2026, Round 2, Problem Track 1**.

Licensed under the [MIT License](LICENSE).
