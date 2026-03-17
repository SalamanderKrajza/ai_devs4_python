# 01_01_grounding

Transforms markdown notes into interactive HTML with fact-checked, source-annotated concepts using structured outputs and web search.

## Run

```bash
npm run lesson1:grounding
```

Pass arguments after `--`:

```bash
npm run lesson1:grounding -- my-note.md
npm run lesson1:grounding -- --force
npm run lesson1:grounding -- my-note.md --force
```

## How it works

```
INPUT:  Markdown file (notes/ folder)
  ↓
1. EXTRACT  — 1 paragraph = 1 API call → concepts.json
  ↓
2. DEDUPE   — group synonyms under canonical labels → dedupe.json
  ↓
3. SEARCH   — 1 concept = 1 API call + web search → search_results.json
  ↓
4. GROUND   — 1 paragraph = 1 API call → grounded.html
```

Each stage runs in parallel batches and caches results. Use `--force` to rebuild from scratch.

## Output

All files written to `output/`:

| File | Content |
|------|---------|
| `concepts.json` | Extracted concepts per paragraph with scores |
| `dedupe.json` | Grouped canonical concepts with aliases |
| `search_results.json` | Web search summaries and sources |
| `grounded.html` | Final interactive HTML with tooltips |

## Configuration

Edit `src/config.js` to change models, timeouts, or retry settings.
