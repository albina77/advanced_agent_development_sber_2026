#!/bin/sh
set -eu
agent_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$agent_dir/.." && pwd)
cd "$agent_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
python_bin="$agent_dir/.venv/bin/python"
if [ ! -x "$python_bin" ]; then
    "${PYTHON:-python3}" -m venv "$agent_dir/.venv"
fi
if ! "$python_bin" -c 'import importlib.util; names = ("langchain_gigachat", "langgraph", "dotenv", "pypdf", "bm25s", "pymorphy3", "qdrant_client", "sentence_transformers"); raise SystemExit(not all(importlib.util.find_spec(name) for name in names))' 2>/dev/null; then
    dependency_file="$agent_dir/requirements.txt"
    if [ -f "$agent_dir/requirements.lock.txt" ]; then
        dependency_file="$agent_dir/requirements.lock.txt"
    fi
    "$python_bin" -m pip install --disable-pip-version-check --retries 0 --timeout 15 -r "$dependency_file"
fi
case "${1:-}" in
    --test)
        shift
        exec "$python_bin" -m unittest discover -s "$agent_dir/tests" -t "$repo_dir" "$@"
        ;;
    --setup-rag)
        shift
        "$python_bin" -c 'from atm_agent.knowledge.bootstrap import prepare_sources; prepare_sources()'
        "$python_bin" -m atm_agent.knowledge.build_index --prepare-reranker "$@"
        exec "$python_bin" -m atm_agent.knowledge.build_index "$@"
        ;;
    --memory-worker) module=atm_agent.memory.worker ;;
    --memory-scenario) module=atm_agent.scenarios.memory ;;
    --synthetic-memory-scenarios) module=atm_agent.scenarios.synthetic_memory ;;
    --memory-lifecycle-scenario) module=atm_agent.scenarios.memory_lifecycle ;;
    --context-scenario) module=atm_agent.scenarios.context ;;
    --demo) module=atm_agent.scenarios.demo ;;
    *) exec "$python_bin" -m atm_agent "$@" ;;
esac
shift
exec "$python_bin" -m "$module" "$@"
