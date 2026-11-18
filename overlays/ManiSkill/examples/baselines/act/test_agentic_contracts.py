"""Execution-contract tests; no model weights, API requests, or simulation."""

import copy
import unittest

from agentic_act.contracts import (
    ContractError, Step, TaskSpec, parse_json_object, validate_next_action,
    validate_plan, verify_final_state,
)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.task = TaskSpec("先将 cube1 放入 A 区，再将 cube2 放入 B 区",
                             (Step("cube1", "zone_A"), Step("cube2", "zone_B")))
        self.plan = {"steps": [step.as_dict() for step in self.task.steps]}
        self.state = {"gripper_empty": True, "objects": {
            "cube1": {"placed_region": "zone_A", "stable": True, "held": False},
            "cube2": {"placed_region": "zone_B", "stable": True, "held": False},
        }}

    def decision(self, decision="execute", step_index=0):
        return {"decision": decision, "step_index": step_index, "reason": "Next verified step"}

    def test_valid_plan_preserves_task(self):
        before = copy.deepcopy(self.plan)
        self.assertEqual(validate_plan(self.plan, self.task), self.task.steps)
        self.assertEqual(self.plan, before)

    def test_plan_cannot_change_order_target_region_or_skill(self):
        variants = []
        row = copy.deepcopy(self.plan); row["steps"].reverse(); variants.append(row)
        row = copy.deepcopy(self.plan); row["steps"][0]["object_id"] = "cube3"; variants.append(row)
        row = copy.deepcopy(self.plan); row["steps"][0]["region_id"] = "zone_B"; variants.append(row)
        row = copy.deepcopy(self.plan); row["steps"][0]["skill"] = "open_drawer"; variants.append(row)
        row = copy.deepcopy(self.plan); row["steps"][0]["reset_env"] = True; variants.append(row)
        row = copy.deepcopy(self.plan); row["steps"][0]["object_id"] = ["cube1"]; variants.append(row)
        for payload in variants:
            with self.subTest(payload=payload), self.assertRaises(ContractError):
                validate_plan(payload, self.task)

    def test_strict_json_rejects_ambiguous_or_non_json_output(self):
        self.assertEqual(parse_json_object('{"steps": []}'), {"steps": []})
        for text in ('{"decision":"execute","decision":"complete"}',
                     '```json\n{}\n```', '{} trailing', '[{}]', '{"x":NaN}'):
            with self.subTest(text=text), self.assertRaises(ContractError):
                parse_json_object(text)

    def test_only_current_step_can_execute(self):
        self.assertEqual(validate_next_action(self.decision(), self.task, 0)["decision"], "execute")
        with self.assertRaises(ContractError):
            validate_next_action(self.decision(step_index=1), self.task, 0)
        row = self.decision(); row["object_id"] = "cube2"
        with self.assertRaises(ContractError):
            validate_next_action(row, self.task, 0)
        with self.assertRaises(ContractError):
            validate_next_action(self.decision(step_index=True), self.task, 1)

    def test_premature_completion_is_rejected(self):
        for completed in (0, 1):
            with self.assertRaises(ContractError):
                validate_next_action(self.decision("complete", completed), self.task, completed)
        self.assertEqual(validate_next_action(self.decision("complete", 2), self.task, 2)["decision"], "complete")
        with self.assertRaises(ContractError):
            validate_next_action(self.decision("execute", 2), self.task, 2)

    def test_retries_are_explicit_and_bounded(self):
        with self.assertRaises(ContractError):
            validate_next_action(self.decision("retry"), self.task, 0)
        self.assertEqual(validate_next_action(self.decision("retry"), self.task, 0,
                                             attempts={0: 1}, max_attempts=2)["decision"], "retry")
        with self.assertRaises(ContractError):
            validate_next_action(self.decision(), self.task, 0, attempts={0: 1})
        with self.assertRaises(ContractError):
            validate_next_action(self.decision("retry"), self.task, 0,
                                 attempts={0: 2}, max_attempts=2)
        self.assertEqual(validate_next_action(self.decision("abort"), self.task, 0,
                                             attempts={0: 2}, max_attempts=2)["decision"], "abort")

    def test_final_placement_requires_current_release_and_stability(self):
        self.assertTrue(verify_final_state(self.task, self.state)["success"])
        for field, value, expected_reason in (
                ("held", True, "object_still_held"),
                ("stable", False, "object_not_stable"),
                ("placed_region", "zone_A", "outside_requested_region")):
            state = copy.deepcopy(self.state); state["objects"]["cube2"][field] = value
            result = verify_final_state(self.task, state)
            self.assertFalse(result["success"])
            self.assertIn(expected_reason, result["checks"][1]["reasons"])
        state = copy.deepcopy(self.state); state["gripper_empty"] = False
        self.assertFalse(verify_final_state(self.task, state)["success"])

    def test_missing_evidence_and_truthy_strings_do_not_count_as_success(self):
        state = copy.deepcopy(self.state); del state["objects"]["cube1"]
        self.assertFalse(verify_final_state(self.task, state)["success"])
        for container, field in ((self.state, "gripper_empty"),
                                 (self.state["objects"]["cube1"], "stable"),
                                 (self.state["objects"]["cube1"], "held")):
            old = container[field]; container[field] = "false"
            try:
                with self.assertRaises(ContractError):
                    verify_final_state(self.task, self.state)
            finally:
                container[field] = old


if __name__ == "__main__":
    unittest.main()
