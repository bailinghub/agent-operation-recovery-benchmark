"""Frozen, bounded external-provider evaluation; no Kaggle publication.

Run each model in a separate process: the SDK's global event observers must
never see concurrent episodes in the same interpreter. Credentials are read
from a private file outside the repository and are excluded from all evidence.
"""

import argparse
import contextlib
import datetime
import hashlib
import io
import json
import logging
from pathlib import Path
import sys
from urllib.parse import urlparse

import kaggle_benchmarks as kbench
from kaggle_benchmarks.actors.llms import OpenAI as OpenAIActor
from openai import OpenAI

from cases import CASES
from kaggle_task import MAX_TOOL_ROUNDS, run_benchmark, write_json


class TokenHubEvaluationActor(OpenAIActor):
    def __init__(self, client, model, request_limit):
        super().__init__(client=client, model=model, name=model,
                         support_temperature=True, support_structured_outputs=False)
        self.stream_responses = False
        self.request_limit = request_limit
        self.request_count = 0
        self.request_errors = []
        self.unexpected_reasoning_responses = 0

    def invoke(self, messages, system=None, **kwargs):
        kwargs = dict(kwargs)
        kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}),
                                "thinking": {"type": "disabled"}}
        kwargs["max_tokens"] = 1024
        response = super().invoke(messages, system=system, **kwargs)
        if response.reasoning_traces:
            self.unexpected_reasoning_responses += 1
            raise RuntimeError("Provider did not honor the non-thinking condition")
        return response

    def _call_api(self, messages, **kwargs):
        if self.request_count >= self.request_limit:
            raise RuntimeError("Frozen evaluation request budget exhausted")
        self.request_count += 1
        try:
            return super()._call_api(messages, **kwargs)
        except Exception as exc:
            self.request_errors.append({"request_index": self.request_count,
                                        "error_type": type(exc).__name__,
                                        "http_status": getattr(exc, "status_code", None)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path,
                        default=Path.home() / ".config/agent-recovery-benchmark/provider.json")
    args = parser.parse_args()
    root = Path(__file__).parent
    manifest = json.loads(args.manifest.read_text())
    if args.model not in manifest["models"] or manifest["case_ids"] != list(CASES):
        raise SystemExit("Model or cases differ from frozen evaluation")
    if manifest["repeats"] != 2 or manifest["automatic_retries"] != 0:
        raise SystemExit("Unsupported frozen execution parameters")
    for name, expected in manifest["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise SystemExit("Source differs from frozen evaluation")
    configuration = json.loads(args.config.read_text())
    endpoint = urlparse(configuration.get("base_url", ""))
    key = configuration.get("api_key", "").strip()
    if (configuration.get("provider") != "tencent-tokenhub" or not key
            or endpoint.scheme != "https" or endpoint.hostname != "tokenhub.tencentmaas.com"):
        raise SystemExit("Invalid private provider configuration")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    kbench.client = kbench.clients.InMemoryClient()
    if kbench.config.ui_handler is not None:
        kbench.events.manager.unbind(kbench.config.ui_handler)
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with OpenAI(api_key=key, base_url=configuration["base_url"], max_retries=0,
                timeout=manifest["timeout_seconds"]) as client:
        actor = TokenHubEvaluationActor(client, args.model,
                                       len(CASES) * manifest["repeats"] * MAX_TOOL_ROUNDS)

        def progress(record, completed, total, output):
            print(json.dumps({"model": args.model, "completed": completed, "total": total,
                              "case_id": record["case_id"], "repeat": record["repeat"],
                              "outcome": record["outcome"],
                              "strict_pass": bool(record["scoring"] and record["scoring"]["safe_correct_resolution"]),
                              "requests": actor.request_count}), file=sys.__stdout__, flush=True)

        # Suppress optional SDK renderers and unsafe exception text. The callback
        # emits only synthetic IDs, counts and booleans through the saved stream.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            summary = run_benchmark(actor, tuple(CASES), repeats=manifest["repeats"],
                                    artifact_root=root / "results" / manifest["evaluation_id"],
                                    source_digests=manifest["source_sha256"], on_episode=progress)
        metadata = {"provider": "tencent-tokenhub", "endpoint_host": endpoint.hostname,
                    "model": args.model, "thinking_requested": "disabled", "stream": False,
                    "automatic_retries": 0, "timeout_seconds": manifest["timeout_seconds"],
                    "max_output_tokens_per_request": 1024, "model_request_count": actor.request_count,
                    "request_errors": actor.request_errors,
                    "unexpected_reasoning_responses": actor.unexpected_reasoning_responses,
                    "evaluation_id": manifest["evaluation_id"], "started_at": start,
                    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "execution": "local real provider calls; not a Kaggle leaderboard result"}
        write_json(Path(summary["artifact_directory"]) / "provider-run.json", metadata)
        write_json(root / "evidence" / f"{manifest['evaluation_id']}-{args.model}.json",
                   {"summary": summary, "provider": metadata})
    print(json.dumps({"model": args.model, "completed": True,
                      "artifact_directory": summary["artifact_directory"],
                      "observed": summary["observed"], "passed": summary["passed"],
                      "usage": summary["usage"]}), flush=True)


if __name__ == "__main__":
    main()
