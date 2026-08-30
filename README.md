# ControlPlane

ControlPlane is a real time control layer that sits between a large language
model and the person using it. It reads every response the model produces,
estimates the probability that the response is harmful, and then decides whether
to serve it, attach a caveat, send it to a human reviewer, or block it.

The idea behind it is narrow and specific. A guardrail is a decision problem
under uncertainty, so it should be built like one. Existing guardrail products
produce a risk score and compare it against a number that somebody picked. If
you ask why that number is 0.8 rather than 0.7, there is no answer. ControlPlane
estimates P(harm) on a calibrated scale, asks the business what each outcome
costs, and then lets expected loss minimisation choose the action. The
thresholds are derived rather than chosen, and every decision carries the
arithmetic that produced it. This is the same reasoning a bank uses to set a
credit cutoff.

Built for the Accenture Innovation Challenge 2026, Problem Track 1
(ControlPlane.ai), Round 2.


## Table of contents

- Requirements
- Optional detector upgrades
- Installation
- Configuration
- Architecture
- Results and evidence
- Reproducing every number
- Known limitations
- Troubleshooting
- FAQ
- Maintainers


## Requirements

Python 3.10 or newer. Core dependencies are listed in `requirements.txt`:

- `pyyaml`, `numpy`, `scikit-learn`, `matplotlib` for the pipeline and evaluation
- `fastapi`, `uvicorn`, `httpx`, `pydantic` for the governance gateway
- `streamlit` for the demo and monitoring interfaces
- `groq` for the optional live LLM provider

No API key and no network access are needed to reproduce any result in this
README. The LLM provider defaults to a deterministic mock and the grounding
backend defaults to an offline lexical scorer. Clone the repository, install the
requirements, and every script below will run.


## Optional detector upgrades

Each detector returns a risk score in `[0, 1]`, so a heavier backend can be
swapped in without touching the decision, session, or anomaly layers. These are
left commented out in `requirements.txt` because they add substantial weight and
a network dependency on first run, which would break the zero setup guarantee
above.

- [Detoxify](https://github.com/unitaryai/detoxify), a classifier trained on the
  Jigsaw toxic comment corpus, replacing the built in lexicon. Enable with
  `CONTROLPLANE_TOXICITY=detoxify`.
- [sentence-transformers](https://www.sbert.net/) for semantic grounding, which
  scores AUROC 0.740 against 0.708 for the lexical default. Enable with
  `CONTROLPLANE_GROUNDING=embedding`.
- [Presidio](https://microsoft.github.io/presidio/) for production PII
  detection, replacing the regex patterns. Enable with
  `CONTROLPLANE_PII=presidio`.

Each one falls back to the built in implementation if the package is
unavailable, so enabling any of them cannot break the pipeline.

All numbers reported below were produced with the defaults. If you enable an
optional backend, expect the numbers to move, since you have changed the
detector feeding the decision layer.


## Installation

```bash
git clone <repository-url>
cd controlplane
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

To reproduce the RAGTruth benchmark results, build the dataset first. The source
data is not redistributed in this repository.

```bash
python data/ragtruth/fetch_ragtruth.py
python data/ragtruth/build_ragtruth.py
```


## Configuration

All governance behaviour lives in `config/policies.yaml`. It is the only file a
business needs to edit, and it contains no thresholds, only costs.

1.  State what each outcome costs, per use case:

    ```yaml
    customer_facing:
      costs:
        serve_bad: 500      # a harmful answer reaches a customer who acts on it
        block_good: 120     # a good answer is withheld
        review: 40          # one reviewer, one item
        caveat: 16          # an unnecessary warning, including fatigue cost
        resid_edit: 0.40    # fraction of harm surviving a caveat
        resid_review: 0.05  # fraction a human reviewer still misses
    ```

1.  The decision bands are then derived from those costs when the policy loads.
    Nothing is set by hand:

    | Use case | Cost ratio | edit at | review at | block at |
    |---|---|---|---|---|
    | `internal_copilot` | 0.6x | 0.071 | 0.300 | 0.898 |
    | `customer_facing` | 4.2x | 0.051 | 0.151 | 0.552 |
    | `regulated_decision` | 33.3x | 0.008 | 0.017 | 0.300 |

    The regulated tool blocks from 0.300 because a bad answer there costs 33
    times a withheld one. The internal copilot tolerates up to 0.898 because an
    expert reader catches the remainder and blocking wastes their time. Same
    engine, different stated costs.

1.  Add rules that escalate regardless of the cost arithmetic, for obligations
    that are not negotiable at any price:

    ```yaml
      hard_rules:
        - {if: pii_detected, action: block}
        - {if: toxicity_high, action: block}
    ```

1.  Check the derivation at any time:

    ```bash
    python scripts/threshold_sensitivity.py
    ```

Optional environment variables: `CONTROLPLANE_LLM` (`mock` by default, or
`groq`), `CONTROLPLANE_GROUNDING` (`tfidf` or `embedding`),
`CONTROLPLANE_TOXICITY` (`lexicon` or `detoxify`), and `CONTROLPLANE_PII`
(`regex` or `presidio`).


## Architecture

```
(query, response, context)
        |
        v
  DETECTORS (run concurrently)
    performance     grounding + self consistency        [async]
    responsibility  PII regex + toxicity                [INLINE]
    cost            tokens, rework                      [async]
    anomaly         Mahalanobis distance, chi squared   [async]
        |
        v
  PROBABILITY ASSEMBLY
    isotonic calibration, noisy OR, prior shift correction  =  p_harm
        |
        v
  DECISION LAYER
    a* = argmin over actions of E[L(a, S)], then hard rules escalate
        |
        +--> allow / edit / review / block
        +--> session monitor: cumulative exposure across turns
        +--> append only audit log, including human overrides
```

Only inline detectors sit on the path the user waits for. Everything heavier
runs concurrently in the background. Measured inline p99 is 0.07 ms against an
800 ms budget.


## Results and evidence

### Hallucination detection

Validated on [RAGTruth](https://github.com/ParticleMedia/RAGTruth) (ACL 2024),
900 real LLM responses carrying human word level annotations.

| Backend | AUROC | Latency | Outcome |
|---|---|---|---|
| Lexical TF-IDF (default) | 0.708 | ~1.5 ms | shipped |
| Semantic embeddings | 0.740 | ~1.9 ms | validated upgrade |
| NLI entailment | 0.578 | ~12,000 ms | rejected, worse and far slower |

The NLI row is worth reading. It was the most sophisticated option and it lost
on both accuracy and speed, so it was dropped.

### Calibration

Isotonic regression fitted on the RAGTruth test split. A raw similarity score is
not a probability, and the decision layer multiplies probabilities by costs, so
this step is what makes the arithmetic legitimate.

| Metric | Raw | Calibrated |
|---|---|---|
| Brier score | 0.376 | 0.152 |
| Expected calibration error | 0.483 | 0.125 |

Abstaining on the least confident 20% of cases lifts accuracy from 82% to 85%.
Abstaining on 40% lifts it to 90%, which says the cases the system is unsure
about really are the ones it gets wrong.

### Loss back test

Policies are scored on the cost they actually realise once the true state of
each response is known, rather than against labels we wrote ourselves.

| Policy | Mean loss | Savings captured |
|---|---|---|
| No guardrail | 1217.67 | 0.0% |
| Block everything | 54.83 | 95.5% |
| Static thresholds (v1) | 191.20 | 84.3% |
| ControlPlane | 26.67 | 97.8% |
| Oracle | 0.00 | 100% |

Blocking everything is in the table on purpose. It captures 95.5% of the
available savings and is useless, because it turns the AI off. Any guardrail
that cannot beat it clearly is not worth its latency.

### Robustness across detector backends

Re running the back test with Detoxify in place of the lexicon leaves the cost
derived policy almost unchanged, moving from 97.8% to 97.7%, while the static
threshold baseline degrades from 191.20 to 228.07. Expected loss decisions built
on calibrated probabilities carry across detectors. Thresholds tuned against one
detector do not.

### Fairness audit of the control layer itself

288 matched pair responses, identical except for one protected attribute token.
Because the pairs mean the same thing, identical treatment is the correct
behaviour, so the experiment supplies its own ground truth.

| Axis | Flip rate | Disparate impact | 95% CI | Four fifths rule |
|---|---|---|---|---|
| Gender | 0.0% | 1.000 | [1.00, 1.00] | pass |
| Region | 0.0% | 1.000 | [1.00, 1.00] | pass |
| Name origin | 2.8% | 0.938 | [0.81, 1.00] | pass |

The mean score gap on the name origin axis is 0.005, which is essentially
nothing, and yet 2.8% of decisions still flip. That is threshold proximity.
Responses sitting close to a band edge get pushed across it by a name alone,
because the lexical backend tokenises some names differently from others. The
defect is in the decision layer rather than the detector, and no aggregate
accuracy metric would reveal it.

Confidence intervals come from a cluster bootstrap that resamples templates
rather than individual responses, since the variants of one template are a
matched set and not independent observations.

### Drift monitoring

16 simulated weeks, reference window frozen at weeks 1 to 4, retrieval quality
degraded from week 9 onward.

| Weeks | PSI | KS p value | Verdict |
|---|---|---|---|
| 1 to 8 | 0.005 to 0.071 | 0.41 to 0.99 | stable |
| 9 | 2.853 | < 0.0001 | material shift |
| 12 | 5.516 | < 0.0001 | material shift |

Silent for eight weeks, then fires on the exact week of injection. PSI is paired
with a two sample KS test because PSI has no null distribution and its 0.10 and
0.25 cutoffs are convention rather than inference. Running both is what caught a
binning problem during development, when PSI was raising alarms in weeks 5 to 7
while KS reported nothing significant.

### Multi turn exposure

An eight turn session in which no individual turn is ever blocked. Cumulative
exposure, computed as 1 minus the product of (1 - p) across served turns,
reaches 19% and breaches the session limit at turn 7. A guardrail that scores one
response at a time cannot see this failure mode. Turns that carry agent actions
are weighted 1.5x, since their output feeds a downstream step rather than simply
being read.

### Statistical anomaly detection

Squared Mahalanobis distance across six structural features, tested against a
chi squared distribution with 6 degrees of freedom at alpha = 0.01. The null
hypothesis is that the response comes from the same distribution as normal
traffic.

| Case | d squared | p value | Reject null |
|---|---|---|---|
| Held out clean | 0.07 | 0.9999 | no |
| Truncated output | 230.56 | < 0.0001 | yes |
| Numeric dump | 1992.86 | < 0.0001 | yes |
| Repetition loop | 49405.47 | < 0.0001 | yes |
| Injected instruction | 636.57 | < 0.0001 | yes |

Every flagged case would either pass a grounding check or be invisible to one. A
repetition loop repeats a true statement, and an injected instruction contains no
claims to verify. The detector is orthogonal to grounding rather than a
duplicate of it. It contributes 0.0 to `p_harm` by design, because unusual is
not the same as harmful, and promoting it to a blocking rule is a one line change
in the policy file.

Robust z scores use the median and MAD rather than the mean and standard
deviation, since the mean and standard deviation are themselves distorted by the
outliers the detector is meant to find.

### Cost of ownership

Model routing was implemented, priced, and then retired. Compute turns out to be
0.05% of total cost of ownership. Routing all traffic to the smaller model saves
$1.79 per 1,000 interactions and costs $2,243 in additional harm and review. The
cost lever that matters is review load, which the cost model sets directly.


## Reproducing every number

```bash
# Evidence
python scripts/evaluate_ragtruth.py      # AUROC on the external benchmark
python scripts/calibrate.py              # Brier, ECE, abstention curve
python scripts/threshold_sensitivity.py  # where the decision bands come from
python scripts/loss_backtest.py          # realised cost comparison
python scripts/fairness_audit.py         # counterfactual matched pair audit
python scripts/drift_monitor.py          # temporal PSI with a KS cross check
python scripts/session_demo.py           # cumulative multi turn exposure
python scripts/anomaly_demo.py           # Mahalanobis anomaly test
python scripts/cost_frontier.py          # total cost of ownership
python scripts/evaluate.py               # detector metrics, triage asymmetry

# Interfaces
streamlit run app/streamlit_app.py                  # score one response live
python scripts/simulate_traffic.py --reset --n 400  # populate the audit log
streamlit run app/monitoring.py                     # operations view
uvicorn app.api:app --reload                        # governance gateway
```


## Known limitations

These are stated plainly, because a system that hides its limitations is less
trustworthy than one that names them.

- The PII and toxicity metrics in `scripts/evaluate.py` are circular.
  `data/generate_dataset.py` builds its positive examples from the same lexicon
  and regex patterns that the detectors match against, so F1 = 1.00 holds by
  construction and measures nothing about detection quality. That script is an
  integration test. The detection evidence is RAGTruth and the fairness audit.
- Grounding is validated on RAG question answering only. The architecture
  accepts any `(query, response, context)` triple, but AUROC 0.740 is evidence
  for that task shape specifically. Summarisation, code generation, and open
  ended chat would each need separate validation against benchmarks such as
  FRANK, HaluEval, or FActScore.
- The system measures faithfulness rather than factuality. It checks whether a
  response is consistent with the context supplied to it. If that context is
  itself stale or wrong, a faithful response to it will pass. No automated
  system can check against a truth it has no access to, which is part of why the
  human review tier exists for high stakes use cases.
- The cost figures are illustrative. The method is the contribution, not the
  numbers. A real deployment would substitute its own incident cost data, after
  which the thresholds recompute on their own.
- Reviewer overrides in the monitoring view are simulated. Every decision in the
  audit log is real and went through the actual pipeline, but the override
  labels are synthetic, because there is no review desk behind this prototype.
- The anomaly reference profile is narrow, fitted on 18 clean responses. A
  deployment would fit it on a rolling window of live traffic, which widens the
  profile and reduces false positives.


## Troubleshooting

**Activation fails on Windows with a `PSSecurityException`.** PowerShell blocks
script execution by default. Either run
`Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process` for the current
session, or skip activation and call `.venv\Scripts\python.exe` directly.

**Results do not match the tables above.** Check whether an optional backend is
still enabled from an earlier session. `echo $env:CONTROLPLANE_TOXICITY` on
Windows, or `env | grep CONTROLPLANE` on Linux and macOS. All reported numbers
use the defaults.

**`scripts/evaluate_ragtruth.py` reports a missing data file.** The RAGTruth
source data is not committed. Run `python data/ragtruth/fetch_ragtruth.py` and
then `python data/ragtruth/build_ragtruth.py`.

**The monitoring view reports an empty audit log.** Populate it first with
`python scripts/simulate_traffic.py --reset --n 400`.

**An optional backend prints "unavailable" and carries on.** That is the
designed fallback rather than an error. Install the relevant package from the
commented section of `requirements.txt` to enable it.


## FAQ

**Q: Why derive thresholds from costs instead of tuning them on validation
data?**

**A:** Tuning optimises whichever metric you picked, at whatever prevalence your
validation set happened to have. Deriving from costs optimises what the business
actually loses, at any prevalence, and produces a threshold that can be defended
line by line to a risk committee. It also transfers: three use cases share one
engine and differ only in their stated costs.

**Q: The detectors were fitted where 65% of responses are harmful, but live
traffic is only a few percent. Does that break the probabilities?**

**A:** It would, which is why a label shift correction is applied before any
decision is taken. In odds form the likelihood ratio is reweighted by the ratio
of priors, so the same loss matrix stays optimal at any base rate. Without the
correction, cost at a 2% base rate is 27.6. With it, 16.3.

**Q: Why is "block everything" included in the back test?**

**A:** Because it captures 95.5% of available savings while being completely
useless. Including it is what makes the 97.8% figure mean something.

**Q: Is the fairness finding a problem?**

**A:** It is a real defect, found in our own system, with an identified
mechanism, and it clears the four fifths test. Reporting it is the point. It is
invisible to every aggregate accuracy metric, which is exactly why the audit
exists.

**Q: Does this work with models other than the one you tested against?**

**A:** The layer never inspects model internals. It takes `(query, response,
context)` as text and returns a decision, so it works with any model consumed
through an API, which is the realistic enterprise constraint the problem brief
describes.


## Maintainers

- Niraj Mhatre, M.Sc. Statistics, IIT Kanpur
- Manya Gupta, M.Sc. Statistics, IIT Kanpur

Licensed under the [MIT License](LICENSE).
