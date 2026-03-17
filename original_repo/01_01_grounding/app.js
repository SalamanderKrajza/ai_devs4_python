import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { createInterface } from "node:readline/promises";

import { paths, cli } from "./src/config.js";
import { splitParagraphs } from "./src/utils/text.js";
import { resolveMarkdownPath } from "./src/utils/file.js";
import { extractConcepts } from "./src/pipeline/extract.js";
import { dedupeConcepts } from "./src/pipeline/dedupe.js";
import { searchConcepts } from "./src/pipeline/search.js";
import { generateAndApplyTemplate } from "./src/pipeline/ground.js";

const confirmRun = async () => {
  const rl = createInterface({ input: process.stdin, output: process.stdout });
  console.log("\n⚠️  UWAGA: Uruchomienie tego skryptu może zużyć znaczną ilość tokenów.");
  console.log("   Jeśli nie chcesz go uruchamiać, możesz sprawdzić gotowy wynik w pliku:");
  console.log("   01_01_grounding/output/grounded_demo.html\n");
  const answer = await rl.question("Czy chcesz kontynuować? (yes/y): ");
  rl.close();
  const normalized = answer.trim().toLowerCase();
  if (normalized !== "yes" && normalized !== "y") {
    console.log("Przerwano.");
    process.exit(0);
  }
};

const main = async () => {
  await confirmRun();
  const sourceFile = await resolveMarkdownPath(paths.notes, cli.inputFile);
  const markdown = await readFile(sourceFile, "utf8");
  const paragraphs = splitParagraphs(markdown);

  console.log(`\n📄 Source: ${sourceFile}`);
  console.log(`   Paragraphs: ${paragraphs.length}\n`);

  console.log("1. Extracting concepts...");
  const conceptsData = await extractConcepts(paragraphs, sourceFile);
  console.log(`   Total: ${conceptsData.conceptCount} concepts\n`);

  console.log("2. Deduplicating concepts...");
  const dedupeData = await dedupeConcepts(conceptsData);
  console.log(`   Groups: ${dedupeData.groups.length}\n`);

  console.log("3. Web search grounding...");
  const searchData = await searchConcepts(conceptsData, dedupeData);
  console.log(`   Results: ${Object.keys(searchData.resultsByCanonical).length}\n`);

  console.log("4. Generating HTML...");
  if (cli.force || !existsSync(paths.grounded)) {
    await generateAndApplyTemplate(markdown, conceptsData, dedupeData, searchData);
    console.log(`   Created: ${paths.grounded}\n`);
  } else {
    console.log(`   Skipped (exists, use --force to regenerate)\n`);
  }

  console.log("✅ Done! Output files:");
  console.log(`   ${paths.concepts}`);
  console.log(`   ${paths.dedupe}`);
  console.log(`   ${paths.search}`);
  console.log(`   ${paths.grounded}`);
};

main().catch((error) => {
  console.error(`Error: ${error.message}`);
  process.exit(1);
});
