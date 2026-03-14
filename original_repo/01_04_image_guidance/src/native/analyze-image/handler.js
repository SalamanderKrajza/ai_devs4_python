import { vision } from "./vision.js";
import { parseAnalysisReport } from "./report.js";
import { buildAnalysisPrompt, DEFAULT_CHECK_ASPECTS } from "./prompt.js";
import { readProjectImage } from "../shared/image-files.js";
import log from "../../helpers/logger.js";

export const analyzeImage = async ({ image_path, original_prompt, check_aspects }) => {
  try {
    const { imageBase64, mimeType } = await readProjectImage(image_path);
    const aspects = check_aspects || DEFAULT_CHECK_ASPECTS;

    log.vision(image_path, "Quality analysis");

    const analysis = await vision({
      imageBase64,
      mimeType,
      question: buildAnalysisPrompt(original_prompt, aspects)
    });

    log.visionResult(analysis.substring(0, 150) + "...");

    const report = parseAnalysisReport(analysis);

    return {
      success: true,
      image_path,
      original_prompt,
      aspects_checked: aspects,
      verdict: report.verdict,
      score: report.score,
      blocking_issues: report.blockingIssues,
      minor_issues: report.minorIssues,
      next_prompt_hints: report.nextPromptHints,
      analysis
    };
  } catch (error) {
    log.error("analyze_image", error.message);
    return { success: false, error: error.message };
  }
};
