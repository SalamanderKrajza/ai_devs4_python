# Filesystem Tools Reference

This document describes the tools available for the AI model to interact with the sandboxed filesystem.

---

## 1. list_files

**Purpose**: List files and directories at a given path within the sandbox.

### Input Schema

```json
{
  "path": "string (required)"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | ✓ | Relative path within sandbox. Use `"."` for root. |

### Handler

```
readdir(path) → returns directory entries with type info
```

### Output Schema

```json
[
  { "name": "file.txt", "type": "file" },
  { "name": "subdir", "type": "directory" }
]
```

### Example

```
Input:  { "path": "." }
Output: [{ "name": "welcome.txt", "type": "file" }, { "name": "docs", "type": "directory" }]
```

---

## 2. read_file

**Purpose**: Read the contents of a file.

### Input Schema

```json
{
  "path": "string (required)"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | ✓ | Relative path to the file within sandbox. |

### Handler

```
readFile(path, "utf-8") → returns file content as string
```

### Output Schema

```json
{
  "content": "string"
}
```

### Example

```
Input:  { "path": "welcome.txt" }
Output: { "content": "Welcome to the AI sandbox!\n\nThis is a safe space..." }
```

---

## 3. write_file

**Purpose**: Write content to a file (creates new or overwrites existing).

### Input Schema

```json
{
  "path": "string (required)",
  "content": "string (required)"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | ✓ | Relative path to the file within sandbox. |
| `content` | string | ✓ | Content to write to the file. |

### Handler

```
writeFile(path, content, "utf-8") → creates/overwrites file
```

### Output Schema

```json
{
  "success": true,
  "message": "File written: <path>"
}
```

### Example

```
Input:  { "path": "notes.txt", "content": "Hello, World!" }
Output: { "success": true, "message": "File written: notes.txt" }
```

---

## 4. delete_file

**Purpose**: Delete a file.

### Input Schema

```json
{
  "path": "string (required)"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | ✓ | Relative path to the file to delete. |

### Handler

```
unlink(path) → removes file from filesystem
```

### Output Schema

```json
{
  "success": true,
  "message": "File deleted: <path>"
}
```

### Example

```
Input:  { "path": "temp.txt" }
Output: { "success": true, "message": "File deleted: temp.txt" }
```

---

## 5. create_directory

**Purpose**: Create a directory (and parent directories if needed).

### Input Schema

```json
{
  "path": "string (required)"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | ✓ | Relative path for the new directory. |

### Handler

```
mkdir(path, { recursive: true }) → creates directory tree
```

### Output Schema

```json
{
  "success": true,
  "message": "Directory created: <path>"
}
```

### Example

```
Input:  { "path": "docs/api" }
Output: { "success": true, "message": "Directory created: docs/api" }
```

---

## 6. file_info

**Purpose**: Get metadata about a file or directory.

### Input Schema

```json
{
  "path": "string (required)"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | ✓ | Relative path to the file or directory. |

### Handler

```
stat(path) → returns filesystem metadata
```

### Output Schema

```json
{
  "size": 1234,
  "isDirectory": false,
  "created": "2025-01-21T10:30:00.000Z",
  "modified": "2025-01-21T11:45:00.000Z"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `size` | number | Size in bytes |
| `isDirectory` | boolean | True if path is a directory |
| `created` | string | ISO 8601 creation timestamp |
| `modified` | string | ISO 8601 last modified timestamp |

### Example

```
Input:  { "path": "welcome.txt" }
Output: { "size": 89, "isDirectory": false, "created": "2025-01-21T...", "modified": "2025-01-21T..." }
```

---

## Error Handling

All tools can return an error object when operations fail:

```json
{
  "error": "Error message describing what went wrong"
}
```

Common errors:
- `Access denied: path "..." is outside sandbox` — path traversal attempt blocked
- `ENOENT: no such file or directory` — file/directory doesn't exist
- `ENOTDIR: not a directory` — tried to list a file as directory
- `EISDIR: illegal operation on a directory` — tried to read/delete directory as file

---

## Security

All paths are validated through `resolveSandboxPath()` which:

1. Resolves the path relative to the sandbox root
2. Checks if the resolved path is still within the sandbox
3. Throws `Access denied` error if path escapes sandbox (e.g., `../config.js`)
