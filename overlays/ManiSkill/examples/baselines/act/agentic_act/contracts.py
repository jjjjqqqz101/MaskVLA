"""Small, strict contracts between a VLM planner and the robot executor.

The task contract is supplied separately from the model output. A plausible
model response cannot change the requested order or authorize a new skill.
Physical success is checked from a fresh observation after execution.
"""

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple


OBJECT_IDS = frozenset(("cube1", "cube2", "cube3"))
REGION_IDS = frozenset(("zone_A", "zone_B", "zone_C"))
SKILL = "pick_place"


class StopRequested(RuntimeError):
    """Cooperative simulation cancellation, without recovery motion."""


class ContractError(ValueError):
    """An external response or observation violates the execution contract."""


@dataclass(frozen=True)
class Step:
    object_id: str
    region_id: str

    def __post_init__(self):
        if not isinstance(self.object_id, str) or self.object_id not in OBJECT_IDS:
            raise ContractError("Unknown object_id: %r" % (self.object_id,))
        if not isinstance(self.region_id, str) or self.region_id not in REGION_IDS:
            raise ContractError("Unknown region_id: %r" % (self.region_id,))

    def as_dict(self):
        return {"skill": SKILL, "object_id": self.object_id,
                "region_id": self.region_id}


@dataclass(frozen=True)
class TaskSpec:
    instruction: str
    steps: Tuple[Step, ...]

    def __post_init__(self):
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ContractError("Task instruction must be non-empty text")
        object.__setattr__(self, "steps", tuple(self.steps))
        if not 1 <= len(self.steps) <= 3 or any(not isinstance(s, Step) for s in self.steps):
            raise ContractError("This prototype requires one to three Step objects")
        if len({s.object_id for s in self.steps}) != len(self.steps):
            raise ContractError("The task must select distinct objects")
        if len({s.region_id for s in self.steps}) != len(self.steps):
            raise ContractError("The task must select distinct destination regions")

    def as_dict(self):
        return {"instruction": self.instruction,
                "steps": [s.as_dict() for s in self.steps]}


def parse_json_object(text: str):
    """Parse one JSON object; reject prose, duplicate keys and NaN/Infinity."""
    if not isinstance(text, str):
        raise ContractError("Model content must be text containing a JSON object")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError("Duplicate JSON key: %s" % key)
            result[key] = value
        return result

    def invalid_constant(value):
        raise ContractError("Non-finite JSON constant: %s" % value)

    try:
        payload = json.loads(text, object_pairs_hook=unique_pairs,
                             parse_constant=invalid_constant)
    except (json.JSONDecodeError, TypeError) as error:
        raise ContractError("Model response is not a single valid JSON object") from error
    if not isinstance(payload, dict):
        raise ContractError("Model response must be a JSON object")
    return payload


def _object(payload: Any, required, allowed, label):
    if not isinstance(payload, dict):
        raise ContractError("%s must be an object" % label)
    missing = set(required) - payload.keys()
    extra = payload.keys() - set(allowed)
    if missing or extra:
        raise ContractError("%s fields invalid (missing=%s, extra=%s)" %
                            (label, sorted(missing), sorted(extra)))


def validate_plan(payload: Any, task: TaskSpec):
    """Return validated steps, preserving exactly the externally requested order."""
    _object(payload, ("steps",), ("steps", "reason"), "Plan")
    if not isinstance(payload["steps"], list) or len(payload["steps"]) != len(task.steps):
        raise ContractError("Plan must contain exactly the requested number of ordered steps")
    parsed = []
    for index, row in enumerate(payload["steps"]):
        _object(row, ("skill", "object_id", "region_id"),
                ("skill", "object_id", "region_id"), "Plan step %d" % index)
        if row["skill"] != SKILL:
            raise ContractError("Unsupported skill: %r" % row["skill"])
        step = Step(row["object_id"], row["region_id"])
        if step != task.steps[index]:
            raise ContractError("Plan step %d changes the requested object, region or order" % index)
        parsed.append(step)
    return tuple(parsed)


def validate_next_action(payload: Any, task: TaskSpec, completed: int,
                         attempts: Optional[Mapping[int, int]] = None,
                         max_attempts: int = 3):
    """Gate a VLM decision against executor progress and bounded retries.

    ``completed`` is the number of consecutively verified steps, not a model
    estimate. ``attempts`` counts calls already started for each zero-based
    step. Retry requires a previous attempt; execute is reserved for the first.
    Final completion must additionally pass ``verify_final_state``.
    """
    if type(completed) is not int or not 0 <= completed <= len(task.steps):
        raise ContractError("completed must be an integer verified-prefix length")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ContractError("max_attempts must be a positive integer")
    attempts = {} if attempts is None else attempts
    if not isinstance(attempts, Mapping):
        raise ContractError("attempts must map step indices to attempt counts")
    if any(type(k) is not int or not 0 <= k < len(task.steps) or
           type(v) is not int or v < 0 for k, v in attempts.items()):
        raise ContractError("Invalid attempt counter")
    _object(payload, ("decision", "step_index", "reason"),
            ("decision", "step_index", "reason", "object_id", "region_id"), "Decision")
    decision = payload["decision"]
    if decision not in ("execute", "retry", "abort", "complete"):
        raise ContractError("Unknown decision")
    if type(payload["step_index"]) is not int or payload["step_index"] != completed:
        raise ContractError("Decision must address the current verified-prefix index")
    if not isinstance(payload["reason"], str) or not payload["reason"].strip():
        raise ContractError("Decision reason must be non-empty text")
    if len(payload["reason"]) > 2000:
        raise ContractError("Decision reason is too long")
    if decision == "complete":
        if completed != len(task.steps):
            raise ContractError("Cannot complete before every requested step is verified")
        if "object_id" in payload or "region_id" in payload:
            raise ContractError("Completion does not select another object or region")
    elif decision in ("execute", "retry"):
        if completed == len(task.steps):
            raise ContractError("No pending skill remains")
        current = task.steps[completed]
        for key in ("object_id", "region_id"):
            if key in payload and payload[key] != getattr(current, key):
                raise ContractError("Decision changes the current task parameters")
        count = attempts.get(completed, 0)
        if count >= max_attempts:
            raise ContractError("Attempt budget exhausted")
        if decision == "retry" and count == 0:
            raise ContractError("Cannot retry before the first attempt")
        if decision == "execute" and count != 0:
            raise ContractError("A repeated skill call must be an explicit retry")
    return dict(payload)


def verify_final_state(task: TaskSpec, observation: Any):
    """Check all current placement predicates, including release and stability.

    Expected observation: ``objects`` maps stable object IDs to dictionaries
    containing ``placed_region`` (string/null), ``stable`` (bool), ``held``
    (bool). Global ``gripper_empty`` must be bool. Missing or malformed evidence
    fails closed; previously recorded success does not substitute for a fresh
    final observation.
    """
    if not isinstance(observation, dict):
        raise ContractError("Observation must be an object")
    objects = observation.get("objects")
    if not isinstance(objects, dict):
        raise ContractError("Observation objects must be an object-ID mapping")
    empty = observation.get("gripper_empty")
    if type(empty) is not bool:
        raise ContractError("gripper_empty must be a boolean")
    checks = []
    for index, step in enumerate(task.steps):
        reasons = []
        row = objects.get(step.object_id)
        if not isinstance(row, dict):
            reasons.append("object_observation_missing")
        else:
            if row.get("placed_region") != step.region_id:
                reasons.append("outside_requested_region")
            if type(row.get("stable")) is not bool:
                raise ContractError("Object stable must be a boolean")
            if type(row.get("held")) is not bool:
                raise ContractError("Object held must be a boolean")
            if not row["stable"]:
                reasons.append("object_not_stable")
            if row["held"]:
                reasons.append("object_still_held")
        if not empty:
            reasons.append("gripper_not_empty")
        checks.append({"step_index": index, "object_id": step.object_id,
                       "region_id": step.region_id, "passed": not reasons,
                       "reasons": reasons})
    return {"success": all(row["passed"] for row in checks),
            "gripper_empty": empty, "checks": checks}
