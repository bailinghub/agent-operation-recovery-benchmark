# Agent Operation Recovery Benchmark

Can an agent recover an uncertain business operation without repeating its effects, while still completing a legitimate new request?

This package contains 24 fixed synthetic cases, deterministic native tools and a strict scorer, 49 offline calibration checks, and 96 recorded external-provider episodes. All business records and actions are synthetic and in memory. The included observations were collected on 9 October 2026 through Tencent TokenHub using the exact model identifiers `deepseek-v4.1-flash` and `deepseek-v4-pro`.

## What the results establish

Both model identifiers completed all 34 episodes per model that required business progress. Their strict final-contract pass counts were 15/48 and 0/48. The strict score also checks plain JSON, identity and target binding, permission rules, and the full trace. Business completion and strict resolution therefore measure different things. Read [RESULTS.md](RESULTS.md) for denominators, failure categories, and limits before comparing scores.

These are local external-provider observations from one model family and one provider, not Kaggle leaderboard entries. Fresh simulator and chat state were used for each repetition; repetitions are not claimed to be statistically independent samples. Logical observations simulate asynchronous progress rather than reproducing real network timing.

## Files

- `source/`: five frozen evaluation modules, two calibration test modules, and a task builder.
- `evaluation-manifest.json`: case IDs, exact model IDs, run settings, and frozen source digests.
- `results/`: all 96 episodes, their recorded configurations and summaries, and a compact overview.
- `requirements.txt`: the reference environment for installing the direct dependencies.
- `requirements-reference-lock.txt` and `reference-environment.json`: the full installed package snapshot used for the release's offline validation.
- `audit/validation.json`: release integrity, calibration, and public-content scan results.
- `SHA256SUMS`: every package file's SHA-256, excluding the checksum list itself.
- `LICENSE` and `NOTICE`: Apache-2.0 licensing for the benchmark code and accompanying authored materials.

## Offline reproduction

Use Python 3.12.13. Release calibration ran on macOS arm64. The original run configurations record `kaggle-benchmarks==0.6.1`, `kaggle==2.2.4`, and `kagglesdk==0.1.37`, plus SHA-256 digests of the installed SDK's native-tool and LLM modules. The reference environment retains those versions and matching SDK digests. Other dependency versions were captured on 10 October 2026; they are a reference for reproduction, not a claim that the original runs recorded a complete dependency lock.

From this package's root:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-reference-lock.txt
python -m unittest discover -s source -p '*tests.py'
```

The 49 tests use scripted responses and deterministic records. They make no model requests, need no credentials, and do not reproduce the 96 model responses. They check the simulator, scorer, native-tool adapter, budgets, missing-observation handling, and task source construction.

Verify the delivered bytes on Linux with `sha256sum -c SHA256SUMS`, or on macOS with `shasum -a 256 -c SHA256SUMS`. Files added by your virtual environment are not part of the delivered package. To inspect or re-aggregate the recorded observations, use the episode JSON files and the frozen functions in `source/scoring.py` and `source/kaggle_task.py`; do not repair malformed final responses before applying the strict scorer.

To build self-contained task source without inference:

```bash
python source/build_submission.py --output generated
```

The generated notebook includes a model-running cell. Building that notebook is offline; executing that cell is a separate model call and may require platform access and incur usage.

## Optional new provider evaluation

A new evaluation makes paid API calls and creates new evidence. It does not replay the included outputs or reproduce identical model behavior. The frozen runner supports Tencent TokenHub and checks the manifest's models, cases, source hashes, and request settings before inference. Keep each model in a separate process, run episodes sequentially, and leave automatic retries disabled.

Create your own private JSON configuration outside this package with the keys `provider`, `base_url`, and `api_key`. The provider must be `tencent-tokenhub`; use the HTTPS API base URL from your provider account with host `tokenhub.tencentmaas.com`. Obtain a key from your own account. The runner's default configuration location is `~/.config/agent-recovery-benchmark/provider.json`; a different private path can be supplied through `--config`. Never commit that file or the key.

With your private configuration ready, run one model at a time:

```bash
python source/provider_evaluation.py --manifest evaluation-manifest.json --model deepseek-v4.1-flash
python source/provider_evaluation.py --manifest evaluation-manifest.json --model deepseek-v4-pro
```

Each command permits at most 288 model requests: 24 cases × 2 repetitions × 6 model rounds. Actual usage in the recorded batch was 166 and 139 requests. Provider monetary costs were unavailable in those observations and remain `null`, not zero. New runs write under `source/results/` and `source/evidence/`, leaving the bundled `results/` intact. Provider model availability and prices can change; review your account before opting in.

## Licensing and dependencies

The benchmark code and accompanying authored materials are distributed under [Apache-2.0](LICENSE). The observed result files preserve the provider's actual outputs and model identifiers as research evidence. This license does not grant rights to provider trademarks, model weights, or services, and API access remains subject to your provider's terms.

The package imports third-party SDKs and libraries but does not include their implementation. They retain their own licenses; see [THIRD_PARTY.md](THIRD_PARTY.md). Installing them separately does not change the benchmark's license.

## Integrity and provenance

The five frozen evaluation modules and all 96 episode files are byte-identical to the reviewed evidence copy. The public evidence copy normalized only machine-specific summary `artifact_directory` fields to bundle-relative values before this release packaging. No result, final response, score, model identifier, or failed episode was rewritten for this release. Hashes, calibration results, and the scan report are included for independent inspection.
