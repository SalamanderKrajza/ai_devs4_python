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

## Code to fetch new changes from orginal repo:
```
git fetch upstream
git subtree pull --prefix=original_repo upstream main --squash
```