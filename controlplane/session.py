"""Multi-turn risk: a per-turn limit is not a session limit.

Every guardrail on the market scores one response at a time. That is a per-trade
limit with no book-level control, and it fails the same way: a run of positions
each individually inside its limit can still take the book past its drawdown
cap. In a conversation, six turns at P(harm)=0.08 are each comfortably below any
sensible per-turn threshold, yet the chance the user has been told at least one
harmful thing is 1 - 0.92^6 = 39%.

Agents make this worse, because a questionable output does not just get read —
it is fed into the next step, so errors compound rather than merely accumulate.

So we track two limits, as a risk desk would:

  * TURN limit    — the expected-loss decision from decision_theory.py, unchanged.
  * SESSION limit — cumulative exposure across the conversation, which triggers
                    escalation even when no single turn ever did.

Cumulative exposure is the noisy-OR survival form:

      C_n = 1 - prod_{i<=n} (1 - p_i)

i.e. the probability that at least one turn in the session was harmful. It is
monotone in n, which is the point: long conversations are riskier than short
ones, and a control layer that cannot express that is blind to its main failure
mode. We also track an EWMA of per-turn risk, which unlike C_n can come back
down — that distinguishes "this session went bad early and recovered" from
"this session is getting worse", and only the latter should force a hard stop.

For agents, an `action_multiplier` inflates the contribution of turns whose
output is consumed downstream rather than merely displayed.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json


@dataclass
class SessionState:
    session_id: str
    use_case: str
    cumulative: float = 0.0          # 1 - prod(1 - p_i): P(at least one harmful turn)
    ewma: float = 0.0                # recent-risk trend, mean-reverting
    turns: int = 0
    escalations: list = field(default_factory=list)
    history: list = field(default_factory=list)


class SessionMonitor:
    """Cumulative exposure control across the turns of one conversation."""

    def __init__(self, limit: float = 0.35, ewma_alpha: float = 0.4,
                 ewma_limit: float = 0.30, action_multiplier: float = 1.5):
        self.limit = limit
        self.alpha = ewma_alpha
        self.ewma_limit = ewma_limit
        self.action_multiplier = action_multiplier
        self._sessions: dict[str, SessionState] = {}

    def get(self, session_id: str, use_case: str = "internal_copilot") -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id, use_case)
        return self._sessions[session_id]

    def update(self, session_id: str, p_harm: float, turn_action: str,
               use_case: str = "internal_copilot", is_agent_action: bool = False) -> dict:
        """Fold one turn into the session and return the session-level verdict."""
        s = self.get(session_id, use_case)
        p = min(1.0, p_harm * (self.action_multiplier if is_agent_action else 1.0))

        # A blocked turn never reached the user, so it contributes no exposure.
        # Anything served — including an edited or reviewed answer — does.
        effective = 0.0 if turn_action == "block" else p

        s.turns += 1
        s.cumulative = 1.0 - (1.0 - s.cumulative) * (1.0 - effective)
        s.ewma = effective if s.turns == 1 else self.alpha * effective + (1 - self.alpha) * s.ewma

        session_action, reason = "continue", None
        if s.cumulative >= self.limit and s.ewma >= self.ewma_limit:
            session_action = "halt_session"
            reason = (f"cumulative exposure {s.cumulative:.3f} >= {self.limit} "
                      f"AND risk still rising (EWMA {s.ewma:.3f} >= {self.ewma_limit})")
        elif s.cumulative >= self.limit:
            session_action = "require_human"
            reason = (f"cumulative exposure {s.cumulative:.3f} >= {self.limit}, "
                      f"but recent turns are improving (EWMA {s.ewma:.3f}) — "
                      f"hand to a reviewer rather than terminating")
        elif s.cumulative >= 0.6 * self.limit:
            session_action = "warn"
            reason = f"cumulative exposure {s.cumulative:.3f} approaching limit {self.limit}"

        rec = dict(turn=s.turns, p_harm=round(p_harm, 4), effective=round(effective, 4),
                   is_agent_action=is_agent_action, turn_action=turn_action,
                   cumulative=round(s.cumulative, 4), ewma=round(s.ewma, 4),
                   session_action=session_action, reason=reason)
        s.history.append(rec)
        if session_action != "continue":
            s.escalations.append(rec)
        return rec

    def report(self, session_id: str) -> str:
        return json.dumps(asdict(self.get(session_id)), indent=2)
