import { processQuery } from "./src/executor.js";
import { api } from "./src/config.js";
import { tools, handlers } from "./src/tools/index.js";
import { initializeSandbox } from "./src/utils/sandbox.js";

const config = {
  model: api.model,
  tools,
  handlers,
  instructions: api.instructions
};

const queries = [
  // List files
  "What files are in the sandbox?",

  // Create a file
  "Create a file called hello.txt with content: 'Hello, World!'",

  // Read a file
  "Read the hello.txt file",

  // Get file metadata
  "Get info about hello.txt",

  // Create a directory
  "Create a directory called 'docs'",

  // Write file in subdirectory
  "Create a file docs/readme.txt with content: 'Documentation folder'",

  // List subdirectory
  "List files in the docs directory",

  // Delete a file
  "Delete the hello.txt file",

  // Security test - path traversal blocked
  "Try to read ../config.js"
];

const main = async () => {
  await initializeSandbox();
  console.log("Sandbox prepared: empty state\n");

  for (const query of queries) {
    await processQuery(query, config);
  }
};

main().catch(console.error);
