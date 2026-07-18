# Repository Guidelines

## Project Structure

- `generation/` contains the OpenCodeEdit data-synthesis pipeline.
- `benchmark/CanItEdit/` contains evaluation scripts for the CanItEdit benchmark.
- `images/` contains documentation assets.
- Follow the component-specific instructions in `generation/README.md` and
  `benchmark/CanItEdit/README.md` when working in those directories.

## Development

- Keep changes focused and avoid committing generated datasets, API keys, or
  other secrets.
- Run the most relevant script-level checks for the files you change.
- Preserve the JSONL format and existing configuration schemas.

## Commit Messages

Use the Conventional Commits format:

```text
<type>(<optional scope>): <description>
```

Common types include `feat`, `fix`, `docs`, `refactor`, `test`, `chore`,
`build`, `ci`, `perf`, and `revert`.

Examples:

```text
feat(generation): add retry handling for API requests
fix(benchmark): correct pass@k aggregation
docs: clarify dataset generation steps
```

Use the imperative mood, keep the subject concise, and add `!` or a
`BREAKING CHANGE:` footer for breaking changes.
