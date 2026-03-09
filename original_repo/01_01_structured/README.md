# 01_01_structured

Structured outputs — the model returns guaranteed valid JSON matching a provided schema.

## Run

```bash
npm run lesson1:structured
```

## What it does

1. Defines a JSON schema for a "person" object (name, age, occupation, skills)
2. Sends text to the API with `text.format` set to the schema (`strict: true`)
3. Parses and displays the extracted data
