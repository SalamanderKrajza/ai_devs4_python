<img src="assets/logo.svg" alt="AI_devs 4: Builders" width="200">

## Requirements

This project runs on [Node.js](https://nodejs.org/) (version 24 or later), a JavaScript runtime. Node.js ships with **npm**, a package manager used to install dependencies and run the examples.

### Installing Node.js

```bash
# macOS (Homebrew — https://brew.sh)
brew install node

# Windows (winget — https://learn.microsoft.com/en-us/windows/package-manager/winget)
winget install OpenJS.NodeJS

# Linux / Ubuntu / Debian (https://deb.nodesource.com)
curl -fsSL https://deb.nodesource.com/setup_24.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node -v
npm -v
```

Alternatively, download the installer directly from [nodejs.org/en/download](https://nodejs.org/en/download).

## Setup

Copy `env.example` to `.env`.

Set one Responses API key. You can choose between **OpenAI** and **OpenRouter**:

**[OpenRouter](https://openrouter.ai/settings/keys)** (recommended) — create an account and generate an API key. No additional verification required.

```bash
OPENROUTER_API_KEY=your_api_key_here
```

**[OpenAI](https://platform.openai.com/api-keys)** — create an account and generate an API key. Note that OpenAI requires [organization verification](https://help.openai.com/en/articles/10910291-api-organization-verification) before API access is granted, which may take additional time.

```bash
OPENAI_API_KEY=your_api_key_here
```

If both keys are present, provider defaults to OpenAI. Override with `AI_PROVIDER=openrouter`.

## Lesson 01

| Example | Run | Description |
|---------|-----|-------------|
| `01_01_interaction` | `npm run lesson1:interaction` | Multi-turn conversation via input history |
| `01_01_structured` | `npm run lesson1:structured` | Structured JSON output with schema validation |
| `01_01_grounding` | `npm run lesson1:grounding` | Fact-checked HTML from markdown notes |

Install dependencies:

```bash
npm run lesson1:install
```

## Lesson 02

| Example | Run | Description |
|---------|-----|-------------|
| `01_02_tool_use` | `npm run lesson2:tool_use` | Function calling with sandboxed filesystem tools |
| `01_02_tools` | `bun run lesson2:minimal` | Minimal Responses API function-calling demo with a single `get_weather` tool |

Install dependencies:

```bash
npm run lesson2:install
```

