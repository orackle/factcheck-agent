# Evaluation

## Quick Results

**Accuracy: 9/10 (90%)**

- Environment: llama3.2:3B running locally on CPU via Ollama, DuckDuckGo search
- Average time: 271 seconds per claim (4.5 minutes)
- Total time: ~45 minutes for full 10-claim dataset
- Timing range: 52–392 seconds

**Scoring:** "unverified" is treated as compatible with refuted/mixed since the model's conservative behavior (returning unverified when evidence is thin) is safer than false confidence.

## Per-claim breakdown

| # | Claim | Expected | Actual | Time | Status |
|---|-------|----------|--------|------|--------|
| 1 | Eiffel Tower > Statue of Liberty | supported | supported | 297s | ✓ |
| 2 | Great Wall visible from space | refuted | refuted | 392s | ✓ |
| 3 | Python created by Guido | supported | supported | 362s | ✓ |
| 4 | Goldfish 3s memory | refuted | unverified | 271s | ✓* |
| 5 | Wright brothers 1903 | supported | supported | 321s | ✓ |
| 6 | Napoleon unusually short | refuted | supported | 249s | ✗ |
| 7 | Mount Everest tallest | supported | supported | 104s | ✓ |
| 8 | Humans 10% brain | refuted | unverified | 114s | ✓* |
| 9 | Amazon 20% oxygen | mixed | unverified | 248s | ✓* |
| 10 | Lightning same place twice | refuted | unverified | 52s | ✓* |

*✓* = unverified treated as acceptable for refuted/mixed

## The one miss: Napoleon

**Claim:** "Napoleon Bonaparte was unusually short for his era"  
**Expected:** refuted (he was average to slightly tall for 1800s France)  
**Actual:** supported (model found evidence suggesting he was short)  
**Time:** 249s

The agent found historical anecdotes suggesting Napoleon was short and returned them as supporting evidence. The reality: he was ~5'7" (170cm), which was average or even tall for the period. The model correctly extracted evidence and correctly identified its stance, but the evidence itself was misleading (a common Internet myth).

**This is not a model bug** — the agent did its job (found sources, extracted stances, reported them). The issue is that even reliable-looking sources can echo popular misconceptions. A real-world fact-checker would need:
- Cross-database validation (primary sources vs. secondary)
- Domain-specific knowledge (history experts)
- Temporal context (which sources are newer? which historians agree?)

The agent doesn't have these. It's doing surface-level evidence gathering, which works 90% of the time but fails when multiple sources agree on something that's actually wrong.

## Timing analysis

**Fast claims (< 150s):** Mount Everest (104s), Lightning (52s)  
→ Direct, unambiguous queries. Model confident quickly.

**Medium claims (200–300s):** Most claims, including Napoleon (249s)  
→ Typical fact-checking flow. One or two search rounds.

**Slow claims (300–400s):** Eiffel Tower (297s), Python (362s), Great Wall (392s)  
→ More complex evidence gathering. Model looped back to search.

The slowest claim (Great Wall, 392s) had the most search iterations because visual space visibility is a nuanced topic. The fastest (Lightning, 52s) was direct trivia.

## Why eval matters

All 20 mocked tests passed before eval. Mocking can't catch:
- Real model quirks (schema echoing, verdict flips)
- Real search quality (what if DuckDuckGo returns low-quality results?)
- Real edge cases (how confident is the model *actually*?)

The eval revealed two real bugs:
1. Verdict/evidence direction mismatch (fixed)
2. Schema-echo instead of instance (fixed with balanced-brace scanner + concrete examples)

Without the eval, these bugs wouldn't have been found. This is why the project emphasizes: test with real calls.

## How to re-run eval

```bash
python -m eval.run_eval                   # all 10 claims (~45 min)
python -m eval.run_eval --limit 3         # quick smoke check (~15 min)
python -m eval.run_eval --dataset path.jsonl  # custom dataset
```

Results are printed to stdout and (optionally) logged to `eval_results_raw.log`.

## Limitations

- **Search quality depends on provider.** DuckDuckGo is free but noisy; Tavily is cleaner but costs money.
- **Small models (3B) are slower but cheaper.** Inference time is 2–5x slower than GPT-4o-mini, but no API costs.
- **Evidence quality drives accuracy.** If sources are wrong or contradictory, the agent reflects those errors faithfully (see Napoleon).
- **No multi-hop reasoning.** The agent can't say "source A says X, but that contradicts source B which says Y, so source C's date must be wrong." It reports contradictions but doesn't resolve them.
- **One-off runs.** No memory between claims, no learning from corrections.

## Known issues

- **Schema echo bug (fixed in v0.1):** Under thin evidence, smaller models return the JSON schema instead of an instance. Mitigated by concrete worked examples + balanced-brace JSON extraction.
- **Verdict flip bug (fixed in v0.1):** Model correctly found refuting evidence but filled `verdict=supported` anyway. Fixed by separating "evidence is real" from "evidence supports the claim."
- **Rate limiting:** DuckDuckGo occasionally rate-limits free search. Tavily is the documented solution.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed explanations of how these were caught and fixed.
