"""Build self-contained Kaggle task functions and a private smoke notebook."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import textwrap


SOURCE_NAMES = ("cases.py", "simulator.py", "scoring.py", "kaggle_task.py")
LOCAL_MODULES = {"cases", "simulator", "scoring", "kaggle_task"}


def inline_source(path):
    source = path.read_text(encoding="utf-8")
    if path.name == "kaggle_task.py":
        source = source.split("# BUILD_TASKS_START:", 1)[0]
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and (node.module in LOCAL_MODULES or node.module == "__future__")
    )]
    return ast.unparse(tree)


def task_source(root, smoke):
    digests = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_NAMES}
    body = "\n\n".join(inline_source(root / name) for name in SOURCE_NAMES)
    name = "agent-operation-recovery-smoke" if smoke else "agent-operation-recovery"
    function = "agent_operation_recovery_smoke" if smoke else "agent_operation_recovery"
    arguments = "llm" if smoke else "llm, repeats: int = 1"
    description = "Three-case native-tool smoke: RUN-01, ACK-01, NEW-01." if smoke else "24 synthetic cases of recovering operations without duplicate effects."
    selection = "SMOKE_CASE_IDS" if smoke else "tuple(CASES)"
    repeat = "1" if smoke else "repeats"
    footer = (
        f"summary = run_benchmark(llm, {selection}, repeats={repeat}, source_digests={digests!r})\n"
        "return float(summary['score_lower_bound'])\n"
    )
    # Kaggle 0.6.1 serializes only task.func source, so each function embeds all
    # simulator, scorer and runner definitions rather than relying on globals.
    task = (
        f"@kbench.task(name={name!r}, description={description!r})\n"
        f"def {function}({arguments}) -> float:\n"
        + textwrap.indent(body + "\n\n" + footer, "    ")
    )
    return "import kaggle_benchmarks as kbench\n\n" + task, digests


def cell(source, kind="code"):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def build(root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    smoke_source, hashes = task_source(root, True)
    benchmark_source, _ = task_source(root, False)
    compile(smoke_source, "smoke.py", "exec")
    compile(benchmark_source, "benchmark.py", "exec")
    (destination / "smoke.py").write_text(smoke_source + "\n", encoding="utf-8")
    (destination / "benchmark.py").write_text(benchmark_source + "\n", encoding="utf-8")
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": [
            cell("# Agent Operation Recovery: three-case smoke\n\nSynthetic business operations, real native tool calls, and saved episode evidence.\n", "markdown"),
            cell(
                "import importlib.metadata\nimport platform\nimport kaggle_benchmarks as kbench\n"
                "print('Python:', platform.python_version())\n"
                "print('kaggle-benchmarks:', importlib.metadata.version('kaggle-benchmarks'))\n"
                "print('Default model:', kbench.llm.name)\n"
                "print('Available model names:', sorted(kbench.llms.keys()))\n"
            ),
            cell(smoke_source + "\n"),
            cell("smoke_run = agent_operation_recovery_smoke.run(llm=kbench.llm)\nsmoke_run\n"),
            cell(
                "from pathlib import Path\nimport json\n"
                "for summary_path in sorted(Path('results').glob('*/*/summary.json')):\n"
                "    summary = json.loads(summary_path.read_text())\n"
                "    print(summary_path, {key: summary[key] for key in ['planned', 'observed', 'missing', 'passed', 'coverage', 'score_lower_bound', 'score_upper_bound']})\n"
            ),
        ],
    }
    for index, entry in enumerate(notebook["cells"]):
        entry["id"] = f"recovery-smoke-{index:02d}"
    (destination / "smoke.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema": "agent-operation-recovery-build.v1", "source_sha256": hashes,
        "smoke_cases": ["RUN-01", "ACK-01", "NEW-01"], "default_repeats": 1,
        "artifacts": {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                      for name in ("smoke.py", "benchmark.py", "smoke.ipynb")},
        "execution": "build only; no model, upload, or publication",
    }
    (destination / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "dist")
    args = parser.parse_args()
    manifest = build(Path(__file__).parent, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "artifacts": list(manifest["artifacts"])}))


if __name__ == "__main__":
    main()
