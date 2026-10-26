# Chat and knowledge console

React / TypeScript / Ant Design frontend for the existing retrieval service.

The separately configured trial service also bundles a standalone `/studio/` page for parallel questions, observed execution stages, side-by-side originals and unchanged answers. It uses the same `/v1/try/` job API and needs no React build. See [trial setup and configuration](../INTERACTIVE_TRIAL.md). Its source is in `../src/llamaindex_retrieval/static/studio/`; no experiment questions or expected answers are bundled.

- `/admin/#/chat`: chat-focused page, conversation sidebar, multi-turn messages and responsive mobile layout. The empty hash opens chat.
- `/admin/#/dashboard`: knowledge console.
- `/admin/#/search`: existing Search Lab, including retrieve-only operations.

## Conversations

Chat sends `POST /v1/ask` with the exact question string, source settings and ordered `history: [{role, content}]`. It uses the same completed-answer history eligibility as Search Lab. Failed, interrupted, partial and invalid-root turns remain visible with an explicit notice but are **not sent as history**. For unified v3/v4, the original completed root JSON is carried, not a rewritten answer. This is a structural/completion policy, not a truth or factual-support check.

The UI stops at 32 eligible history pairs rather than silently dropping or summarizing old messages. There is one pending request per page; switching chats does not move its response to the newly selected chat. Enter sends, Shift+Enter inserts a newline, and IME composition Enter does not submit.

Each new chat defaults to **automatic retrieval**. With the optional binary retrieval gate configured, a positive decision queries both the knowledge base and the web; a negative decision answers without retrieval. Existing chats keep their saved scope. Source settings can be changed before the first request; start a new chat to change scope afterward. Auto/web modes can send model-generated queries to external search providers. The UI does not enable experimental backend protocols or change model configuration.

## Persistence and privacy

Complete response objects, raw model strings, source snapshots and conversation drafts are stored in the current browser's IndexedDB (`rwkvrag-chat-v1`). Drafts are saved when leaving the input/conversation, not on every keystroke. This is **not** server-side conversation sync or an account system. Large histories consume browser storage; no automatic deletion or raw truncation is performed.

Storage failures and stale-tab write conflicts show a warning. Changes then remain in memory: export the current chat before reloading. Exports include full raw records and may contain sensitive data. Do not leave sensitive conversations on shared devices. Browser deletion removes the local chat only, not backend request/trace records. Different browser origins have separate stores.

“Stop waiting” aborts the frontend fetch and excludes any late response; it does not certify cancellation of server/model work. Reloading or leaving the chat page interrupts waiting, does not automatically resend and cannot recover a server-side background job. The API is non-streaming; no token stream, intermediate answer or simulated progress is invented.

Answers retain source-drawer access, selected original spans, snapshot/coordinate metadata, raw output and full response inspection. Source binding and completion do not establish factual correctness. Generated text is not rendered as executable HTML.

## Development and checks

From this directory, with Node supporting TypeScript stripping:

```bash
npm ci
npm run dev
node --experimental-strip-types --test tests/*.test.mjs
npx tsc -b
```

`npm run build` writes the configured application static bundle; use a separate output directory when validating without publishing:

```bash
npx vite build --outDir /absolute/path/to/preview-build
```

`tests/chat.browser.py` provides isolated real-Edge checks from WSL using synthetic HTTP fixtures only (multi-turn payloads, persistence, cancellation, late responses, failure/raw/source handling, scope settings, mobile layout and storage conflicts). It needs an existing Edge executable plus `httpx` and `websockets` in the selected Python environment; it installs nothing. See its `--help` and use a fresh output directory and browser profile. Browser fixture success is not a native-model or answer-quality evaluation.

`node tests/studio.browser.mjs` checks the bundled Studio with synthetic HTTP fixtures: parallel jobs, source scopes, unchanged answer text, citation navigation, failed jobs, repeated batches, snapshot display without live events and mobile layout. It requires `playwright-core` and Chromium. Set `PLAYWRIGHT_MODULE_DIR` to an existing Node package directory containing `node_modules/playwright-core` and `PLAYWRIGHT_CHROMIUM_EXECUTABLE` to an installed browser if they are outside the default environment. The test does not install dependencies or call a model.
