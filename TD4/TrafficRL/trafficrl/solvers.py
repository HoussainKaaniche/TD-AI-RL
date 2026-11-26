
"""Exact dynamic-programming planners for the supplied finite TrafficMDP.

No Flask dependency. Iteration callbacks receive actual solver measurements.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from .mdp import TrafficMDP, State


@dataclass
class SolveResult:
    values: dict[State, float]
    policy: dict[State, str]
    history: list[dict]
    converged: bool
    iterations: int
    elapsed_seconds: float


def _validate(gamma: float, epsilon: float, max_iterations: int) -> None:
    if not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0, 1)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")


def _prepare(mdp: TrafficMDP):
    """Cache transitions once; no repeated Poisson calculations per iteration."""
    return {
        s: {a: mdp.transitions(s, a) for a in mdp.valid_actions(s)}
        for s in mdp.states
    }


# TODO 1
def _q(outcomes, values, gamma):
    total = 0.0

    for p, next_state, reward in outcomes:
        total += p * (reward + gamma * values[next_state])

    return total


def action_values(
    mdp: TrafficMDP, state: State, values: dict[State, float], gamma: float
):
    """Q(s,a) for each valid action under a supplied value function."""
    if not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0, 1)")
    return {
        a: _q(mdp.transitions(state, a), values, gamma)
        for a in mdp.valid_actions(state)
    }


# TODO 2
def _greedy(actions, values, gamma):
    best_action = None
    best_score = float("-inf")

    for action, outcomes in actions.items():
        score = _q(outcomes, values, gamma)

        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def value_iteration(
    mdp: TrafficMDP,
    gamma: float = 0.95,
    epsilon: float = 1e-3,
    max_iterations: int = 1000,
    on_iteration: Callable[[dict], None] | None = None,
) -> SolveResult:

    _validate(gamma, epsilon, max_iterations)
    start = perf_counter()
    transitions = _prepare(mdp)
    values = dict.fromkeys(mdp.states, 0.0)
    policy = {s: next(iter(transitions[s])) for s in mdp.states}
    history = []
    converged = False

    for iteration in range(1, max_iterations + 1):

        # TODO 3a: Bellman optimality update
        updated = {}

        for s in mdp.states:
            updated[s] = max(
                _q(outcomes, values, gamma)
                for outcomes in transitions[s].values()
            )

        # TODO 3b: Maximum difference
        delta = max(
            abs(updated[s] - values[s])
            for s in mdp.states
        )

        # TODO 3c: Extract greedy policy
        new_policy = {
            s: _greedy(transitions[s], updated, gamma)
            for s in mdp.states
        }

        changes = sum(new_policy[s] != policy[s] for s in mdp.states)
        values, policy = updated, new_policy

        record = {
            "iteration": iteration,
            "delta": delta,
            "policy_changes": changes,
            "elapsed_seconds": perf_counter() - start,
        }

        history.append(record)

        if on_iteration:
            on_iteration(record.copy())

        if delta < epsilon:
            converged = True
            break

    return SolveResult(
        values, policy, history, converged,
        len(history), perf_counter() - start
    )


def policy_iteration(
    mdp: TrafficMDP,
    gamma: float = 0.95,
    epsilon: float = 1e-3,
    max_iterations: int = 1000,
    evaluation_epsilon: float = 1e-5,
    max_evaluation_iterations: int = 10000,
    on_iteration: Callable[[dict], None] | None = None,
) -> SolveResult:

    _validate(gamma, epsilon, max_iterations)

    if evaluation_epsilon <= 0 or max_evaluation_iterations < 1:
        raise ValueError(
            "evaluation_epsilon and max_evaluation_iterations must be positive"
        )

    start = perf_counter()
    transitions = _prepare(mdp)
    values = dict.fromkeys(mdp.states, 0.0)
    policy = {s: next(iter(actions)) for s, actions in transitions.items()}
    history = []
    converged = False

    for iteration in range(1, max_iterations + 1):

        evaluation_converged = False

        for evaluation_steps in range(1, max_evaluation_iterations + 1):

            # TODO 4a: Policy Evaluation
            updated = {}

            for s in mdp.states:
                action = policy[s]
                outcomes = transitions[s][action]
                updated[s] = _q(outcomes, values, gamma)

            delta = max(
                abs(updated[s] - values[s])
                for s in mdp.states
            )

            values = updated

            if delta < evaluation_epsilon:
                evaluation_converged = True
                break

        # TODO 4b: Policy Improvement
        improved = {
            s: _greedy(transitions[s], values, gamma)
            for s in mdp.states
        }

        changes = sum(improved[s] != policy[s] for s in mdp.states)
        policy = improved

        record = {
            "iteration": iteration,
            "delta": delta,
            "policy_changes": changes,
            "evaluation_iterations": evaluation_steps,
            "evaluation_converged": evaluation_converged,
            "elapsed_seconds": perf_counter() - start,
        }

        history.append(record)

        if on_iteration:
            on_iteration(record.copy())

        if changes == 0 and evaluation_converged:
            converged = True
            break

        if not evaluation_converged:
            break

    return SolveResult(
        values, policy, history, converged,
        len(history), perf_counter() - start
    )
