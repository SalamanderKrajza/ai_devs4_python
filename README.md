# What is AI DEVS 4
Let's quothe the autors =)
```
5 tygodni efektywnego wejścia w świat generatywnego AI – w projektowanie produkcyjnych rozwiązań, które na stałe staną się integralną częścią Twojego życia i pracy zespołów. AI_devs 4 to:

→ Przestrzeń na budowanie i wdrażanie produkcyjnych rozwiązań AI
→ Context engineering, budowanie zaawansowanych agentów
i workflows adresujące ograniczenia AI
→ Wdrożenie i utrzymywanie generatywnych aplikacji
```

# Other useful notes:
## Environment config
Install Poetry from https://python-poetry.org/docs/#installation, then run `poetry install` to set up dependencies. Use `poetry shell` (or `poetry run python <script>`) to run scripts inside the virtual environment. Copy `env.dist` to `.env` and fill in your keys.

# Tasks
| Task | Link | Tags | Description |
| --- | --- | --- | --- |
| S01E01 | [S01E01.py](tasks/S01E01/S01E01.py) | [gemini][structured-output][pandas] | Filter suspects and classify job categories with Gemini |
| S02E02 | [S02E02.py](tasks/S02E02/S02E02.py) | [agents][function-calling][geocoding][haversine][tqdm] | Find suspect near a power plant with agent tools and geocoding |

## Commands to update `original_repo` subtree:
```
git rm -r original_repo
git commit -m "tmp remove original_repo"

git fetch upstream
git subtree add --prefix=original_repo upstream main --squash

# Optional: squash the two commits into one
git reset --soft HEAD~2
git commit -m "Update original_repo subtree"
```

# Other resources:
### Models cooperation on token level
[🤝3mAI: Współpraca między modelami](https://youtu.be/DJI2XC71BlA?list=PL6gb3F2o2zOTJYxrcnWphmGLylXIlVCGJ) [ [ "RelayLLM: Efficient Reasoning via Collaborative Decoding"](https://arxiv.org/abs/2601.05167) ]

[🤝 3mAI: Przed wyruszeniem w drogę trzeba...](https://youtu.be/X2kWqlSnX1E?list=PL6gb3F2o2zOTJYxrcnWphmGLylXIlVCGJ) [ [Token-Level LLM Collaboration via FusionRoute](https://arxiv.org/abs/2601.05106) ]

## Other tools
→ [SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks](https://arxiv.org/abs/2602.12670)

---

<img src="https://mermaid.js.org/hero-chart-dark.svg" alt="Mermaid Hero Chart" style="max-height:200px; height:auto;" />

→ [Mermaid.js](https://mermaid.js.org/)

---

[Tokenizer online](https://tiktokenizer.vercel.app/)

---

