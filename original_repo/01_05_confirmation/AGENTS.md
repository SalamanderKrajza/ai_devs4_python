# JavaScript Engineering Guidelines

note: use npm, not bun. bun is banned

## Core Principles

1. **Simplicity over cleverness** — The best code is code you don't write. Delete before you add.

2. **Explicit over implicit** — Dependencies, errors, and side effects should be visible, not hidden.

3. **Data transformations over state mutations** — Functions take input, return output. Avoid changing things in place.

4. **Fail fast, recover at edges** — Validate and reject bad input early. Handle errors at system boundaries.

5. **Composition over inheritance** — Build from small, focused pieces. Combine them; don't extend them.

---

## Meta-Skills

**Judgment** — Every rule has exceptions. Know when to break them. A 50-line function can be clearer than 5 composed ones.

**Trade-offs** — There's no "best." Only "best for this context." Ask: what are we optimizing for?

**Simplicity** — The hardest skill. Resist adding. Question every abstraction. Complexity is easy; simplicity is work.

**Shipping** — Working code beats perfect code. Iterate. Learn. Improve. Delete.

**Reading** — Read more code than you write. Understand before you change.

**Knowing when to stop** — Good enough is often enough. Perfectionism kills projects.

---

## High-Signal Practices

### Functions & Data
- Pure functions by default — same input, same output, no side effects
- Functions over classes — classes only for stateful resources or framework requirements
- Early return over nested conditionals
- Return new data; never mutate inputs
- Destructure parameters with sensible defaults
- Max ~20 lines per function; extract when it does two things

### Error Handling  
- Errors are values, not surprises — Result types over try/catch chains
- Custom errors with context: what failed, why, what was the input
- Log errors once, at the boundary — not at every layer
- Never silently swallow: `catch (e) {}` is a bug

### Async
- async/await everywhere — .then() chains are hard to read
- Promise.all for parallel work; Promise.allSettled when partial failure is okay
- Timeouts on every external call — nothing should hang forever
- AbortController for cancellation

### Boundaries & Validation
- Validate at entry points — parse input into trusted types once
- Trust nothing from outside — user input, API responses, file contents
- Whitelist over blacklist

### Architecture
- Thin handlers → fat services → pure domain logic
- Dependency injection via function parameters or closures
- Feature folders over layer folders
- Add folders when you feel the pain — 3 files don't need structure
- One file, one concern; index.js for public API

### Naming
- Functions: verbs (create, get, validate, send)
- Data: nouns (user, config, result)  
- Booleans: is, has, can, should
- No abbreviations except universals (id, url, db, config)
- If you need a comment to explain the name, rename it

### Node.js
- ES modules, top-level await
- fs/promises, not callbacks
- Graceful shutdown on SIGTERM/SIGINT
- Streams for large data — don't buffer entire files
- crypto.randomUUID() for IDs

---

## When to Break the Rules

- **"Pure functions by default"** — Side effects are fine when that's the point (logging, I/O, persistence).
- **"Max 20 lines"** — A clear 40-line function beats 4 unclear 10-line functions.
- **"No mutation"** — Local mutation inside a function for performance is fine if nothing leaks.
- **"Validate at entry"** — Deep validation is okay when business rules live in domain layer.
- **"Functions over classes"** — Classes are fine when framework requires them or for genuinely stateful resources.

---

## Reference

### API Design
- RESTful resources, predictable URLs
- Consistent envelope: `{ data, error, meta }`
- HTTP status codes mean what they say
- Idempotent PUT/DELETE, safe GET
- Cursor-based pagination over offset

### Security
- Parameterized queries — never interpolate
- Secrets in env, never in code
- Constant-time comparison for sensitive data
- Rate limit auth endpoints

### Testing
- Pure functions are trivial to test
- Inject dependencies to enable mocking
- Test behavior, not implementation
- Test failures and edge cases, not just happy path

### Performance
- Measure before optimizing
- Batch database operations
- Lazy evaluation for large datasets (generators, streams)
- Connection pooling for databases

### Observability
- Structured logs (JSON)
- Request ID in every log entry
- Trace external calls with timing

### File I/O
- fs/promises over callbacks
- Stream large files — don't readFile() gigabytes
- Atomic writes: temp file → rename
- path.join() — never string concatenation

### Dependencies
- Prefer native APIs (fetch, crypto, node:test)
- Use framework features before writing custom abstractions
- Read library types/source before extending or wrapping
- Lock versions in lockfile
- Don't npm install for 10 lines of code
