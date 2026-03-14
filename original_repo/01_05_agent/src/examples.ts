export function printExamples(base: string, token: string): void {
  const auth = `-H "Authorization: Bearer ${token}"`
  const ct = `-H "Content-Type: application/json"`

  const examples = [
    {
      label: 'Chat with alice agent',
      cmd: `curl -s ${base}/api/chat/completions \\\n    ${auth} \\\n    ${ct} \\\n    -d '{"agent":"alice","input":"What is 42 * 17?"}' | jq`,
    },
    {
      label: 'Multi-turn conversation (get sessionId, then follow up)',
      cmd: `SESSION=$(curl -s ${base}/api/chat/completions \\\n    ${auth} \\\n    ${ct} \\\n    -d '{"agent":"alice","input":"Remember: my name is Adam"}' | jq -r '.data.sessionId')\n\n  curl -N ${base}/api/chat/completions \\\n    ${auth} \\\n    ${ct} \\\n    -d "{\\"agent\\":\\"alice\\",\\"sessionId\\":\\"$SESSION\\",\\"input\\":\\"What is my name?\\",\\"stream\\":true}"`,
    },
    {
      label: 'Custom model (no agent template)',
      cmd: `curl -s ${base}/api/chat/completions \\\n    ${auth} \\\n    ${ct} \\\n    -d '{"model":"openai:gpt-4.1-mini","input":"Say hello"}' | jq`,
    },
    {
      label: 'Health check',
      cmd: `curl -s ${base}/health | jq`,
    },
    {
      label: 'List MCP servers',
      cmd: `curl -s ${base}/api/mcp/servers \\\n    ${auth} | jq`,
    },
  ]

  console.log('\n  Try it:\n')
  for (const { label, cmd } of examples) {
    console.log(`  # ${label}`)
    console.log(`  ${cmd}\n`)
  }
}
