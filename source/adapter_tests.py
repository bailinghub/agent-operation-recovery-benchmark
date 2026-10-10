"""Offline tests against the installed SDK; all model responses are scripted."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import kaggle_benchmarks as kbench
from kaggle_benchmarks.actors.llms import LLMChat, LLMResponse

from cases import CASES
from kaggle_task import run_episode, run_benchmark, aggregate_results, cache_off
from build_submission import build


class ScriptedLLM(LLMChat):
    def __init__(self, mode="correct"):
        super().__init__(name="offline-scripted", support_temperature=True)
        self.mode = mode
        self.requests = 0

    def invoke(self, messages, system=None, **kwargs):
        self.requests += 1
        prompt = next(message.content for message in messages if message.sender.role == "user")
        case = next(case for case in CASES.values() if case.prompt in prompt)
        results = [message.content.output for message in messages if message.sender.role == "tool" and message.content.output]
        if self.mode == "timeout":
            raise TimeoutError("Scripted offline timeout")
        if self.mode in {"unknown", "missing_argument"} and self.requests == 1:
            name = "missing_tool" if self.mode == "unknown" else "read_operation"
            args = {} if self.mode == "missing_argument" else {"target_id": case.original_target}
            return self.tool_response(name, args)
        if self.mode == "oversized_batch" and self.requests == 1:
            calls = [self.tool_call("create_operation", {"intent_id": case.new_intent_id,
                       "target_id": case.original_target, "payload": case.payload}, index) for index in range(13)]
            return LLMResponse(content="", tool_calls=calls)
        success = next((result for result in reversed(results) if result.get("status") == "SUCCEEDED"), None)
        if success and self.mode != "round_limit":
            return LLMResponse(content=json.dumps({
                "status": "COMPLETED_NEW" if case.family == "NEW" else "COMPLETED_ORIGINAL",
                "operation_id": success["operation_id"], "target_id": success["target_id"],
                "result": success["result"], "explanation": "Observed authoritative success.",
            }), meta={"input_tokens": 10, "output_tokens": 20})
        if case.family == "NEW":
            return self.tool_response("create_operation", {"intent_id": case.new_intent_id,
                                      "target_id": case.original_target, "payload": case.payload})
        return self.tool_response("read_operation", {"operation_id": case.original_id, "target_id": case.original_target})

    @staticmethod
    def tool_call(name, args, index=0):
        return {"id": f"offline-call-{index}", "function": {"name": name, "arguments": json.dumps(args)}}

    def tool_response(self, name, args):
        return LLMResponse(content="", tool_calls=[self.tool_call(name, args)])


class StreamingDefaultScriptedLLM(ScriptedLLM):
    """Exercise the installed SDK's streaming path unless runner disables it."""

    def __init__(self, mode="correct"):
        super().__init__(mode)
        self.stream_responses = True
        self.observed_stream_flags = []

    def invoke(self, messages, system=None, **kwargs):
        self.observed_stream_flags.append(self.stream_responses)
        response = super().invoke(messages, system=system, **kwargs)
        if not self.stream_responses:
            return response
        chunks = [SimpleNamespace(
            index=index, id=call["id"], type="function",
            function=SimpleNamespace(**call["function"]),
        ) for index, call in enumerate(response.tool_calls or [])]
        return iter([LLMResponse(content=response.content, tool_calls=chunks,
                                 meta=response.meta)])


class AdapterTests(unittest.TestCase):
    def episode(self, mode="correct", case_id="ACK-01"):
        with tempfile.TemporaryDirectory() as directory:
            result = run_episode(ScriptedLLM(mode), case_id, 1, directory)
            self.assertTrue((Path(directory) / f"{case_id}-r01.json").is_file())
            return result

    def test_native_success_and_usage(self):
        result = self.episode()
        self.assertTrue(result["scoring"]["safe_correct_resolution"])
        self.assertEqual(result["requested_tool_calls"], 1)
        self.assertEqual(result["model_rounds"], 2)
        self.assertEqual(result["usage"]["output_tokens"], 20)

    def test_default_streaming_actor_is_disabled_and_restored(self):
        actor = StreamingDefaultScriptedLLM()
        with tempfile.TemporaryDirectory() as directory:
            result = run_episode(actor, "ACK-01", 1, directory)
        self.assertEqual(actor.observed_stream_flags, [False, False])
        self.assertTrue(actor.stream_responses)
        self.assertEqual(result["model_rounds"], 2)
        self.assertEqual(result["requested_tool_calls"], 1)
        self.assertEqual(len(result["trace"]), 1)
        self.assertTrue(result["native_tool_requests"][0]["executed_by_closure"])
        self.assertTrue(result["scoring"]["safe_correct_resolution"])

    def test_actor_stream_setting_is_restored_after_provider_error(self):
        actor = StreamingDefaultScriptedLLM("timeout")
        with tempfile.TemporaryDirectory() as directory:
            result = run_episode(actor, "ACK-01", 1, directory)
        self.assertTrue(actor.stream_responses)
        self.assertEqual(actor.observed_stream_flags, [False])
        self.assertEqual(result["outcome"], "missing")

    def test_execution_and_strict_final_diagnostics_have_distinct_aggregates(self):
        from reference_tests import oracle_episode
        from scoring import score_episode
        sim, final = oracle_episode(CASES["NEW-01"])
        score = score_episode(sim, "Completed.\n" + json.dumps(final))
        result = {"case_id": "NEW-01", "family": "NEW", "repeat": 1,
                  "outcome": "observed", "error": None, "scoring": score, "usage": {}}
        summary = aggregate_results([result])
        self.assertEqual(summary["passed"], 0)
        self.assertEqual(summary["metrics"]["execution_progress_completed_rate"]["rate"], 1)
        self.assertEqual(summary["metrics"]["new_operation_strict_resolution_incomplete_rate"]["rate"], 1)
        self.assertEqual(summary["metrics"]["new_operation_legitimate_progress_incomplete_rate"]["rate"], 0)
        self.assertNotIn("new_operation_false_refusal_rate", summary["metrics"])

    def test_episode_observer_receives_saved_evidence_and_partial_progress(self):
        seen = []
        def observe(record, completed, total, output):
            self.assertTrue((output / "episodes" / f"{record['case_id']}-r01.json").is_file())
            partial = json.loads((output / "partial-summary.json").read_text())
            self.assertEqual(partial["completed_episodes"], completed)
            self.assertEqual(partial["total_episodes"], total)
            seen.append((record["case_id"], completed, total))
        with tempfile.TemporaryDirectory() as directory:
            summary = run_benchmark(ScriptedLLM(), ("ACK-01", "NEW-01"),
                                    artifact_root=directory, on_episode=observe)
            final_partial = json.loads((Path(summary["artifact_directory"]) / "partial-summary.json").read_text())
        self.assertEqual(seen, [("ACK-01", 1, 2), ("NEW-01", 2, 2)])
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(final_partial["execution_status"], "complete")

    def test_progress_observer_error_cannot_drop_a_model_sample(self):
        def observe(*args):
            raise RuntimeError("Offline observer failure")
        with tempfile.TemporaryDirectory() as directory:
            summary = run_benchmark(ScriptedLLM(), ("ACK-01", "NEW-01"),
                                    artifact_root=directory, on_episode=observe)
        self.assertEqual(summary["planned"], 2)
        self.assertEqual(summary["observed"], 2)
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(len(summary["progress_observer_errors"]), 2)

    def test_progress_observer_receives_a_copy_of_the_scored_record(self):
        def observe(record, *args):
            record["scoring"]["safe_correct_resolution"] = False
        with tempfile.TemporaryDirectory() as directory:
            summary = run_benchmark(ScriptedLLM(), ("ACK-01",),
                                    artifact_root=directory, on_episode=observe)
        self.assertEqual(summary["passed"], 1)

    def test_sdk_rejected_unknown_tool_is_not_lost(self):
        result = self.episode("unknown")
        self.assertEqual(result["requested_tool_calls"], 2)
        self.assertEqual(len(result["trace"]), 1)
        self.assertTrue(result["scoring"]["protocol_error"])
        self.assertFalse(result["scoring"]["safe_correct_resolution"])

    def test_signature_rejection_is_not_lost(self):
        result = self.episode("missing_argument")
        self.assertTrue(result["scoring"]["protocol_error"])
        self.assertEqual(result["requested_tool_calls"], 2)

    def test_call_budget_freezes_thirteenth_side_effect(self):
        result = self.episode("oversized_batch", "NEW-01")
        self.assertEqual(result["requested_tool_calls"], 13)
        self.assertEqual(len(result["effects"]), 12)
        self.assertTrue(result["scoring"]["tool_budget_exhausted"])
        self.assertTrue(result["native_tool_requests"][-1]["budget_blocked"])

    def test_round_limit_is_behavior_budget_failure(self):
        result = self.episode("round_limit")
        self.assertEqual(result["outcome"], "observed")
        self.assertEqual(result["error"]["kind"], "budget_exhausted")
        self.assertEqual(result["model_rounds"], 6)
        self.assertTrue(result["scoring"]["tool_budget_exhausted"])

    def test_platform_error_is_retained_as_missing(self):
        result = self.episode("timeout")
        self.assertEqual(result["outcome"], "missing")
        self.assertIsNone(result["scoring"])
        summary = aggregate_results([result])
        self.assertEqual(summary["missing"], 1)
        self.assertEqual(summary["score_lower_bound"], 0)
        self.assertEqual(summary["score_upper_bound"], 1)

    def test_cache_contract_is_explicit(self):
        with cache_off():
            self.assertFalse(getattr(kbench.client, "use_cache", False))

    def test_single_file_task_has_no_adjacent_module_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            build(Path(__file__).parent, Path(directory))
            spec = importlib.util.spec_from_file_location("offline_generated_smoke", Path(directory) / "smoke.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
                # Redirect only synthetic result files for this offline test.
                original_cwd = Path.cwd()
                import os
                os.chdir(directory)
                try:
                    result = module.agent_operation_recovery_smoke.func(ScriptedLLM())
                    summaries = list(Path("results").glob("*/*/summary.json"))
                    self.assertEqual(result, 1.0)
                    summary = json.loads(summaries[0].read_text())
                    self.assertEqual(summary["planned"], 3)
                    self.assertEqual(summary["missing"], 0)
                finally:
                    os.chdir(original_cwd)
            finally:
                sys.modules.pop(spec.name, None)


if __name__ == "__main__":
    unittest.main()
