# Songwriter's Toolkit GUI transplant audit

Target: `/home/theea/songwriter_s_toolkit/dev/gui`

This payload is an overlay for the fresh GMTec SvelteKit scaffold. It intentionally does **not** contain `node_modules` or replace the GMTec framework scaffold files.

## Apply

From the target GUI directory, extract this archive with overwrite enabled, then remove the obsolete scaffold e2e test:

```bash
cd /home/theea/songwriter_s_toolkit/dev/gui
unzip -o /path/to/SongwritersToolkit_GUI_transplant.zip
rm -f src/routes/page.svelte.e2e.ts
sudo docker compose -f compose.dev.yaml up
```

No dependency-file transplant is required. Keep the GMTec `package.json`, `package-lock.json`, `tsconfig.json`, `Dockerfile`, and `src/routes/+layout.svelte`.

## Portable project path

The GUI directory is `dev/gui`, so the toolkit-level `Projects` directory is two levels above it:

```text
../../Projects
```

`compose.dev.yaml` therefore mounts the host path relatively:

```yaml
- ../../Projects:/projects
```

The `/projects` side is only the container-internal mount point; it is not a host-machine hardcoded path.

Outside Docker, `src/lib/server/project.ts` falls back to resolving `../../Projects` from the GUI working directory. If `SONGWRITER_PROJECTS_ROOT` is set, that value overrides the fallback.

## Kept from the laptop application

- `data/user/*.json` — user vocabulary data
- active GUI components: `Generate`, `Lyrics`, `NegativeTextPrompt`, `PromptPair`, `SongMetadata`, `SongParts`, `TextPrompt`
- built-in vocabulary modules under `src/lib/data/`
- project/vocabulary server code under `src/lib/server/`
- canonical project types under `src/lib/types/`
- main `+page.svelte`
- project/vocabulary API routes
- the GUI focus styling added to `layout.css`
- the `data/user` watch-ignore addition in `vite.config.ts`

## Deliberately not transplanted

- `.svelte-kit/` — generated
- `node_modules/` — generated dependency tree
- laptop `.env` (`UID`/`GID`) — unused by the current compose file
- laptop `package.json` / `package-lock.json` / `tsconfig.json` — old SvelteKit 2 scaffold
- laptop `+layout.svelte` — old SvelteKit 2 version
- `AcapellaInput.svelte` — orphaned/unreferenced
- `GenrePrompt.svelte` — orphaned; superseded by `PromptPair`
- `ColorOptions.svelte` and `/customize` — isolated unfinished branch, not used by the main GUI
- `/demo` and its Playwright demo — scaffold/demo material
- laptop README/scaffold recreation metadata

## Compatibility/audit fixes included

- migrated application imports from `$lib` to SvelteKit 3 `#lib` imports with explicit extensions
- removed the machine-specific laptop bind mount and home-directory fallback
- consolidated project typing on `src/lib/types/project.ts`; removed the stale duplicate project schema from `project.ts`
- fixed project-save API validation so intended HTTP 400 errors are not converted into generic 500 responses
- uses `Response.json(...)` in API routes rather than the deprecated SvelteKit `json(...)` helper

## Verification performed

- no legacy `/home/...` paths, laptop/GMTec usernames, old toolkit-root spellings, `os.homedir()` fallback, or `$lib` imports remain
- all `#lib` imports in the cleaned tree resolve to actual files
- all TypeScript files and all Svelte `<script lang="ts">` blocks pass a TypeScript syntax parse
- GMTec `package.json`, `package-lock.json`, `tsconfig.json`, and SvelteKit-3 `+layout.svelte` remain unchanged

A full `svelte-check` was not run in the audit sandbox because its local Node/runtime dependency installation is not the same as the project's Node 24 Docker environment. The supplied Dockerfile uses Node 24, which is appropriate for the fresh SvelteKit 3 scaffold.
