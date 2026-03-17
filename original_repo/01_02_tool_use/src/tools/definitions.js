export const tools = [
  {
    type: "function",
    name: "list_files",
    description: "List files and directories at a given path within the sandbox",
    parameters: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "Relative path within sandbox. Use '.' for root directory."
        }
      },
      required: ["path"],
      additionalProperties: false
    },
    strict: true
  },
  {
    type: "function",
    name: "read_file",
    description: "Read the contents of a file",
    parameters: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "Relative path to the file within sandbox"
        }
      },
      required: ["path"],
      additionalProperties: false
    },
    strict: true
  },
  {
    type: "function",
    name: "write_file",
    description: "Write content to a file (creates or overwrites)",
    parameters: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "Relative path to the file within sandbox"
        },
        content: {
          type: "string",
          description: "Content to write to the file"
        }
      },
      required: ["path", "content"],
      additionalProperties: false
    },
    strict: true
  },
  {
    type: "function",
    name: "delete_file",
    description: "Delete a file",
    parameters: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "Relative path to the file to delete"
        }
      },
      required: ["path"],
      additionalProperties: false
    },
    strict: true
  },
  {
    type: "function",
    name: "create_directory",
    description: "Create a directory (and parent directories if needed)",
    parameters: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "Relative path for the new directory"
        }
      },
      required: ["path"],
      additionalProperties: false
    },
    strict: true
  },
  {
    type: "function",
    name: "file_info",
    description: "Get metadata about a file or directory",
    parameters: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "Relative path to the file or directory"
        }
      },
      required: ["path"],
      additionalProperties: false
    },
    strict: true
  }
];
