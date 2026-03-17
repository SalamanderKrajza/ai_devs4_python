export const DEFAULT_CHECK_ASPECTS = [
  "prompt_adherence",
  "visual_artifacts",
  "anatomy",
  "text_rendering",
  "style_consistency",
  "composition"
];

export const buildAnalysisPrompt = (originalPrompt, aspects) => `Analyze this AI-generated image for quality issues. The original prompt was:
"${originalPrompt}"

Please evaluate the following aspects:

${aspects.includes("prompt_adherence") ? `1. PROMPT ADHERENCE: Does the image accurately represent what was requested? What elements match or are missing?` : ""}
${aspects.includes("visual_artifacts") ? `2. VISUAL ARTIFACTS: Are there any glitches, distortions, blur, noise, or unnatural patterns?` : ""}
${aspects.includes("anatomy") ? `3. ANATOMY: If there are people/animals, check for correct proportions, especially hands, fingers, faces, and limbs.` : ""}
${aspects.includes("text_rendering") ? `4. TEXT RENDERING: If text was requested, is it readable and correctly spelled?` : ""}
${aspects.includes("style_consistency") ? `5. STYLE CONSISTENCY: Is the visual style coherent throughout the image?` : ""}
${aspects.includes("composition") ? `6. COMPOSITION: Is the framing and layout balanced and appropriate?` : ""}

Use this exact output format:

VERDICT: ACCEPT or RETRY
SCORE: <1-10>
BLOCKING_ISSUES:
- <only issues that materially break the brief; use "none" if there are none>
MINOR_ISSUES:
- <optional polish notes that do not require another retry; use "none" if there are none>
NEXT_PROMPT_HINT:
- <targeted retry hint only if VERDICT is RETRY; otherwise use "none">

Decision rules:
- Use ACCEPT when the main subject, pose guidance, and style requirements are satisfied, even if minor polish notes remain.
- Use RETRY only when there are blocking issues such as wrong pose, broken composition, unreadable required text, severe artifacts, or clear style violations.
- Do NOT use RETRY for small polish improvements alone.`;
