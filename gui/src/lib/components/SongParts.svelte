<script lang="ts">
	type SongPartEntry = {
		id: number;
		value: string;
	};

	interface Props {
		vocabulary: string[];
		entries?: SongPartEntry[];
		onDone?: (entries: SongPartEntry[]) => void;
	}

	let {
		vocabulary = $bindable([]),
		entries = $bindable([]),
		onDone
	}: Props = $props();

	let input = $state('');
	let focused = $state(false);
	let editing = $state(true);

	let inputElement = $state<HTMLInputElement | null>(null);


	let draggedId = $state<number | null>(null);
	let pendingUnknown = $state<string | null>(null);

	function normalize(value: string): string {
		return value
			.trim()
			.toLowerCase()
			.replace(/[-_]+/g, ' ')
			.replace(/\s+/g, ' ');
	}

	function findCanonical(
		value: string,
		list: readonly string[]
	): string | null {
		const normalized = normalize(value);

		if (!normalized) {
			return null;
		}

		return (
			list.find(
				(item) => normalize(item) === normalized
			) ?? null
		);
	}

	function suggestions(): string[] {
		const value = normalize(input);

		if (value.length < 2) {
			return [];
		}

		return vocabulary.filter((part) =>
			normalize(part).includes(value)
		);
	}

	function parsePart(value: string): {
		base: string;
		suffix: string;
	} | null {
		const trimmed = value.trim();

		if (!trimmed) {
			return null;
		}

		const match = trimmed.match(
			/^(.+?)(?:\s+(\d+))?$/
		);

		if (!match) {
			return null;
		}

		return {
			base: match[1].trim(),
			suffix: match[2] ? ` ${match[2]}` : ''
		};
	}

	function canonicalizePart(
		value: string
	): string | null {
		const parsed = parsePart(value);

		if (!parsed) {
			return null;
		}

		const canonicalBase = findCanonical(
			parsed.base,
			vocabulary
		);

		if (!canonicalBase) {
			return null;
		}

		return `${canonicalBase}${parsed.suffix}`;
	}

	function refocusInput() {
		queueMicrotask(() => {
			inputElement?.focus();
		});
	}

	function selectSuggestion(value: string) {
		input = value;
		focused = true;
		refocusInput();
	}

	function addPart() {
		const raw = input.trim();

		if (!raw) {
			return;
		}

		const canonical = canonicalizePart(raw);

		if (!canonical) {
			pendingUnknown = raw;
			return;
		}

		const nextId =
			entries.reduce((max, entry) => Math.max(max, entry.id), 0) + 1;

		entries = [
			...entries,
			{
				id: nextId,
				value: canonical
			}
		];

		input = '';
		focused = true;

		refocusInput();
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter') {
			event.preventDefault();
			addPart();
		}
	}

	function removePart(id: number) {
		entries = entries.filter(
			(entry) => entry.id !== id
		);
	}

	function updatePart(
		id: number,
		value: string
	) {
		entries = entries.map((entry) =>
			entry.id === id
				? { ...entry, value }
				: entry
		);
	}

	function startDrag(
		event: DragEvent,
		id: number
	) {
		if (!editing) {
			event.preventDefault();
			return;
		}

		draggedId = id;

		if (event.dataTransfer) {
			event.dataTransfer.effectAllowed = 'move';
			event.dataTransfer.setData(
				'text/plain',
				String(id)
			);
		}
	}

	function dragOver(event: DragEvent) {
		if (!editing || draggedId === null) {
			return;
		}

		event.preventDefault();

		if (event.dataTransfer) {
			event.dataTransfer.dropEffect = 'move';
		}
	}

	function dropOn(
		event: DragEvent,
		targetId: number
	) {
		event.preventDefault();

		if (
			!editing ||
			draggedId === null ||
			draggedId === targetId
		) {
			draggedId = null;
			return;
		}

		const fromIndex = entries.findIndex(
			(entry) => entry.id === draggedId
		);

		const toIndex = entries.findIndex(
			(entry) => entry.id === targetId
		);

		if (fromIndex === -1 || toIndex === -1) {
			draggedId = null;
			return;
		}

		const reordered = [...entries];

		const [moved] = reordered.splice(
			fromIndex,
			1
		);

		reordered.splice(toIndex, 0, moved);

		entries = reordered;
		draggedId = null;
	}

	function finishDrag() {
		draggedId = null;
	}

	function done() {
		const corrected: SongPartEntry[] = [];

		for (const entry of entries) {
			const canonical = canonicalizePart(
				entry.value
			);

			if (!canonical) {
				pendingUnknown = entry.value.trim();
				return;
			}

			corrected.push({
				...entry,
				value: canonical
			});
		}

		entries = corrected;
		editing = false;

		onDone?.(corrected);
	}

	function edit() {
		editing = true;
	}

	async function addCustomPart() {
		if (!pendingUnknown) {
			return;
		}

		const value = pendingUnknown;

		const parsed = parsePart(value);

		const vocabularyValue =
			parsed?.base ?? value;

		const response = await fetch(
			'/api/vocabulary/songparts',
			{
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					value: vocabularyValue
				})
			}
		);

		if (!response.ok) {
			return;
		}

		vocabulary = [
			...vocabulary,
			vocabularyValue
		];

		pendingUnknown = null;

		/*
		 * If this was an edited existing entry,
		 * replace that entry with its newly accepted value.
		 *
		 * If this was a newly typed part, add it.
		 */
		const existing = entries.find(
			(entry) => entry.value === value
		);

		if (existing) {
			const canonical =
				canonicalizePart(value);

			if (canonical) {
				updatePart(
					existing.id,
					canonical
				);
			}
		} else {
			const canonical =
				canonicalizePart(value);

			if (canonical) {
				const nextId =
					entries.reduce((max, entry) => Math.max(max, entry.id), 0) + 1;

				entries = [
					...entries,
					{
						id: nextId,
						value: canonical
					}
				];

				input = '';
				focused = true;

				refocusInput();
			}
		}
	}

	function rejectCustomPart() {
		pendingUnknown = null;

		focused = true;
		refocusInput();
	}
</script>

<div class="song-parts">
<h3 style="margin:.2em" class="font-medium">Structure list</h3>
    {#if editing}
        <div class="field">
            <input
                bind:this={inputElement}
                type="text"
                aria-label="Song part"
                placeholder="Add parts..."
                bind:value={input}
                onfocus={() => (focused = true)}
                onblur={() => (focused = false)}
                onkeydown={handleKeydown}
            />

            {#if focused && suggestions().length > 0}
                <ul class="suggestions">
                    {#each suggestions() as suggestion (suggestion)}
                        <li>
                            <button
                                type="button"
                                onmousedown={(event) =>
                                    event.preventDefault()
                                }
                                onclick={() =>
                                    selectSuggestion(
                                        suggestion
                                    )
                                }
                            >
                                {suggestion}
                            </button>
                        </li>
                    {/each}
                </ul>
            {/if}
        </div>
    {/if}

    <div class="structure-list" role="list">
        {#each entries as entry, index (entry.id)}
            <div
                role="listitem"
                class:dragging={draggedId === entry.id}
                class="bg-green-200 border-green-800 structure-entry"
                draggable={editing}
                ondragstart={(event) =>
                    startDrag(event, entry.id)
                }
                ondragover={dragOver}
                ondrop={(event) =>
                    dropOn(event, entry.id)
                }
                ondragend={finishDrag}
            >
                {#if editing}
                    <span
                        class="drag-handle"
                        aria-hidden="true"
                    >
                        ☷
                    </span>
                {/if}

                <span class="structure-number">
                    {index + 1}.
                </span>

                {#if editing}
                    <input
                        class="part-input"
                        type="text"
                        aria-label={`Song part ${index + 1}`}
                        value={entry.value}
                        oninput={(event) =>
                            updatePart(
                                entry.id,
                                (
                                    event.currentTarget as HTMLInputElement
                                ).value
                            )
                        }
                    />
                {:else}
                    <span class="part-value">
                        {entry.value}
                    </span>
                {/if}

                {#if editing}
                    <button
                        type="button"
                        class="remove-button"
                        aria-label={`Remove ${entry.value}`}
                        onclick={() =>
                            removePart(entry.id)
                        }
                    >
                        ×
                    </button>
                {/if}
            </div>
        {/each}
    </div>

    {#if entries.length > 0}
        <div class="structure-actions">
            {#if editing}
                <button
                    type="button"
					class="bg-green-50"
                    onclick={done}
                >
                    Done
                </button>
            {:else}
                <button
                    type="button"
					class="bg-green-50"
                    onclick={edit}
                >
                    Edit
                </button>
            {/if}
        </div>
    {/if}

    {#if pendingUnknown}
        <div class="dialog-backdrop">
            <div
                class="dialog"
                role="dialog"
                aria-modal="true"
                aria-labelledby="unknown-part-title"
            >
                <h2 id="unknown-part-title">
                    Unknown song part
                </h2>

                <p>
                    <strong>{pendingUnknown}</strong>
                    is not in the Song Parts list.
                </p>

                <p>Add it for future use?</p>

                <div class="dialog-actions">
                    <button
                        type="button"
                        onclick={addCustomPart}
                    >
                        Yes
                    </button>

                    <button
                        type="button"
                        onclick={rejectCustomPart}
                    >
                        No
                    </button>
                </div>
            </div>
        </div>
    {/if}
</div>

<style>
	input {
		width: 100%;
		box-sizing: border-box;
		background: white;
		margin: .2em;
	}
	.song-parts {
		position: relative;
	}

	.field {
		position: relative;
        margin-left: 1rem;
        margin-right: 1rem;
	}

	.field input {
		width: 100%;
		box-sizing: border-box;
	}

	.suggestions {
		position: absolute;
		z-index: 20;
		top: 100%;
		left: 0;
		right: 0;
		margin: 0;
		padding: 0;
		list-style: none;
		background: white;
		border: 1px solid #999;
		max-height: 220px;
		overflow-y: auto;
	}

	.suggestions button {
		display: block;
		width: 100%;
		padding: 5px 8px;
		border: 0;
		background: white;
		text-align: left;
		cursor: pointer;
	}

	.suggestions button:hover {
		background: #eee;
	}

	.structure-list {
		margin-top: 6px;
	}

	.structure-entry {
		display: flex;
		align-items: center;
		gap: 4px;
		min-height: 27px;
        margin-left: 1rem;
        margin-right: 1rem;
		margin-top: .1rem;
	}

	.structure-entry.dragging {
		opacity: 0.4;
	}

	.drag-handle {
		width: 14px;
		cursor: grab;
		user-select: none;
		color: #777;
	}

	.structure-number {
		width: 22px;
		text-align: right;
		flex-shrink: 0;
	}

	.part-input {
		flex: 1;
		min-width: 0;
		border: 1px solid transparent;
		padding: 2px 4px;
		background: transparent;
	}

	.part-input:focus {
		border-color: #999;
		outline: none;
		background: white;
	}

	.part-value {
		flex: 1;
	}

	.remove-button {
		width: 20px;
		padding: 0;
		border: 0;
		background: transparent;
		cursor: pointer;
		font-weight: bold;
	}

	.remove-button:hover {
		color: red;
	}

	.structure-actions {
		display: flex;
		gap: 6px;
		margin-top: 8px;
	}

	.structure-actions button {
		padding: 3px 10px;
	}

	.dialog-backdrop {
		position: fixed;
		inset: 0;
		z-index: 100;
		display: grid;
		place-items: center;
		background: rgb(0 0 0 / 35%);
	}

	.dialog {
		width: min(420px, calc(100vw - 40px));
		padding: 20px;
		background: white;
		border: 1px solid #555;
	}

	.dialog-actions {
		display: flex;
		justify-content: flex-end;
		gap: 8px;
		margin-top: 20px;
	}
</style>