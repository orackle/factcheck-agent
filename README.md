# Fact-Check Agent

A multi-step LangGraph agent that verifies claims against live web sources. Plans queries → searches → reads pages → extracts evidence → reflects on confidence → synthesizes a cited verdict. **Every claim is tied to a URL and quote you can verify yourself.**

Backs the "Web Research & Fact-Checking Agent" card on [my portfolio](https://github.com/orackle/portfo).

## Why this works

Most fact-checking UIs are a single LLM call with a search API bolted on. The engineering insight here is that **a model shouldn't grade its own homework after one search.** This agent can decide it needs more evidence and loop back to search again (capped by `MAX_SEARCH_ITERATIONS`) before synthesizing a verdict. Result: 90% accuracy on a 10-claim test set with full citations.

## Try it

```bash
# Local Ollama (free, no key)
pip install -r requirements.txt
cp .env.example .env
uvicorn factcheck.api:app --reload
# → http://localhost:8000
```

Or from the CLI:
```bash
python -m factcheck.cli "Is the Eiffel Tower taller than the Statue of Liberty?"
python -m factcheck.cli --json --trace "..."
```

## What you get

- **Color-coded verdict** — supported / refuted / mixed / unverified
- **Citations** — every claim links to a source URL + exact quote
- **Agent trace** — collapsible view of every search, fetch, and reasoning step
- **Confidence %** — how sure the agent is (based on evidence overlap)
- **Contradictions** — if sources disagree, they're listed explicitly

## Key features

- ✅ **Pluggable LLM** — Ollama (local), OpenAI, or Anthropic via `.env`
- ✅ **Pluggable search** — DuckDuckGo (default) or Tavily
- ✅ **JSON-in-prompt + repair** — works with small 3B models, not just GPT-4
- ✅ **Loop-back edge** — reflect → search can fire multiple times (capped)
- ✅ **No hallucinated citations** — every source is fetched and quoted
- ✅ **20 mocked tests** — wiring verified without touching the network
- ✅ **Real eval results** — 9/10 on diverse claims (see [EVALUATION.md](EVALUATION.md))

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — graph design, node responsibilities, design decisions
- **[EVALUATION.md](EVALUATION.md)** — accuracy results, timing, per-claim breakdown, known miss
- **[SETUP.md](SETUP.md)** — detailed provider setup, tuning, troubleshooting

## Project structure

```
factcheck/
  agent/      state.py, nodes.py, graph.py
  tools/      search.py, fetch.py
  llm.py      provider-agnostic with JSON repair
  config.py   .env-driven settings
  schemas.py  Pydantic contracts
  cli.py      CLI entry point
  api.py      FastAPI + static frontend
static/
  index.html  Single-page UI (no build step)
eval/
  dataset.jsonl     10 labeled test claims
  run_eval.py       Runs agent, reports accuracy
tests/
  20 mocked tests   graph wiring, JSON parsing, search
```

## Running tests

```bash
pytest          # 20 tests, no network
python -m eval.run_eval          # real LLM + search (20–40 min)
python -m eval.run_eval --limit 3  # quick smoke check
```

## License

MIT
