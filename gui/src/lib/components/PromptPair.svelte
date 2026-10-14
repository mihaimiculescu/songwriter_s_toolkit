<script lang="ts">
	import { onMount } from 'svelte';

	type Entry = {
		id: number;
		attribute: string;
		value: string;
	};

	type PendingUnknown = {
		field: 'attribute' | 'value';
		value: string;
	};

	interface Props {
		rightLabel: string;
		rightVocabulary: string[];
		attributeVocabulary: string[];
		attributeKind: 'attributes';
		rightKind: 'genres' | 'songtypes' | 'themes' | 'vocals' | 'instruments';
		entries?: Entry[];
	}

	let {
		rightLabel,
		rightVocabulary,
		attributeVocabulary,
		attributeKind,
		rightKind,
		entries = $bindable([])
	}: Props = $props();

	let root = $state<HTMLDivElement>();

	let attributeInput = $state('');
	let rightInput = $state('');

	let attributeFocused = $state(false);
	let rightFocused = $state(false);

	let nextId = 1;

	let pendingUnknowns = $state<PendingUnknown[]>([]);
	let currentUnknown = $state<PendingUnknown | null>(null);

	let missingRightAttribute = $state<string | null>(null);

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

	function suggestions(
		value: string,
		list: readonly string[]
	): string[] {
		const normalized = normalize(value);

		if (normalized.length < 3) {
			return [];
		}

		return list.filter((item) =>
			normalize(item).includes(normalized)
		);
	}

	function attributeSuggestions(): string[] {
		return suggestions(
			attributeInput,
			attributeVocabulary
		);
	}

	function rightSuggestions(): string[] {
		return suggestions(
			rightInput,
			rightVocabulary
		);
	}

	function selectAttribute(value: string) {
		attributeInput = value;
		attributeFocused = false;
	}

	function selectRight(value: string) {
		rightInput = value;
		rightFocused = false;
	}

	function isKnown(
		value: string,
		list: readonly string[]
	): boolean {
		return !!findCanonical(value, list);
	}

	function requestUnknown(
		field: 'attribute' | 'value',
		value: string
	) {
		pendingUnknowns = [
			...pendingUnknowns,
			{ field, value }
		];

		showNextUnknown();
	}

	function showNextUnknown() {
		if (currentUnknown || pendingUnknowns.length === 0) {
			return;
		}

		const [next, ...remaining] = pendingUnknowns;

		pendingUnknowns = remaining;
		currentUnknown = next;
	}

	async function addCustomValue(
		kind: 'attributes' | 'genres' | 'songtypes' | 'themes' | 'vocals' | 'instruments',
		value: string
	): Promise<boolean> {
		const response = await fetch(
			`/api/vocabulary/${kind}`,
			{
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ value })
			}
		);

		if (!response.ok) {
			return false;
		}

		return true;
	}

	async function acceptUnknown() {
		const unknown = currentUnknown;

		if (!unknown) {
			return;
		}

		const kind =
			unknown.field === 'attribute'
				? attributeKind
				: rightKind;

		const success = await addCustomValue(
			kind,
			unknown.value
		);

		if (!success) {
			return;
		}

		if (unknown.field === 'attribute') {
			attributeVocabulary = [
				...attributeVocabulary,
				unknown.value
			];
		} else {
			rightVocabulary = [
				...rightVocabulary,
				unknown.value
			];
		}

		// Close this dialog first.
		currentUnknown = null;

		// If another unknown value is waiting, show it.
		if (pendingUnknowns.length > 0) {
			showNextUnknown();
			return;
		}

		// Otherwise finish the current entry.
		commitAfterVocabularyUpdate();
	}

	function rejectUnknown() {
		const unknown = currentUnknown;

		if (!unknown) {
			return;
		}

		if (unknown.field === 'attribute') {
			attributeInput = '';
		} else {
			rightInput = '';
		}

		currentUnknown = null;

		if (pendingUnknowns.length > 0) {
			showNextUnknown();
			return;
		}

		resetInputs();
	}

	function resetInputs() {
		attributeInput = '';
		rightInput = '';
		attributeFocused = false;
		rightFocused = false;
	}

	function commitAfterVocabularyUpdate() {
		if (pendingUnknowns.length > 0 || currentUnknown) {
			return;
		}

		const attribute =
			findCanonical(
				attributeInput,
				attributeVocabulary
			) ?? attributeInput.trim();

		const value =
			findCanonical(
				rightInput,
				rightVocabulary
			) ?? rightInput.trim();

		if (!value) {
			return;
		}

		entries = [
			...entries,
			{
				id: nextId++,
				attribute,
				value
			}
		];

		resetInputs();
	}

	function commit() {
		const rawAttribute = attributeInput.trim();
		const rawValue = rightInput.trim();

		/*
		 * Nothing entered.
		 */
		if (!rawAttribute && !rawValue) {
			resetInputs();
			return;
		}

		/*
		 * Attribute without its corresponding value is invalid.
		 */
		if (rawAttribute && !rawValue) {
			missingRightAttribute = rawAttribute;
			return;
		}

		/*
		 * Canonicalize known values.
		 */
		const canonicalAttribute =
			findCanonical(
				rawAttribute,
				attributeVocabulary
			);

		const canonicalValue =
			findCanonical(
				rawValue,
				rightVocabulary
			);

		attributeInput =
			canonicalAttribute ?? rawAttribute;

		rightInput =
			canonicalValue ?? rawValue;

		/*
		 * Ask about unknown values.
		 */
		if (
			rawAttribute &&
			!isKnown(
				rawAttribute,
				attributeVocabulary
			)
		) {
			requestUnknown(
				'attribute',
				rawAttribute
			);
		}

		if (
			rawValue &&
			!isKnown(
				rawValue,
				rightVocabulary
			)
		) {
			requestUnknown(
				'value',
				rawValue
			);
		}

		if (
			pendingUnknowns.length > 0 ||
			currentUnknown
		) {
			return;
		}

		commitAfterVocabularyUpdate();
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter') {
			event.preventDefault();
			commit();
		}
	}

	function handleOutsideClick(event: MouseEvent) {
		if (!root) {
			return;
		}

		if (
			event.target instanceof Node &&
			!root.contains(event.target)
		) {
			commit();
		}
	}

	function removeEntry(id: number) {
		entries = entries.filter(
			(entry) => entry.id !== id
		);
	}

	function dismissMissingRight() {
		missingRightAttribute = null;
	}

	onMount(() => {
		document.addEventListener(
			'mousedown',
			handleOutsideClick
		);

		return () => {
			document.removeEventListener(
				'mousedown',
				handleOutsideClick
			);
		};
	});
</script>

<div
	bind:this={root}
	class="prompt-pair"
>
<h3 style="margin:.2em;" class="mb-2 font-medium">{rightLabel}</h3>
	<div class="prompt-pair-inputs">
		<div class="field">
			<input
				class= " m-1 rounded-md"
				type="text"
				aria-label="Attribute"
				placeholder="Attribute"
				bind:value={attributeInput}
				onfocus={() => (attributeFocused = true)}
				onkeydown={handleKeydown}
			/>

			{#if attributeFocused && attributeSuggestions().length > 0}
				<ul class="suggestions">
					{#each attributeSuggestions() as suggestion (suggestion)}
						<li>
							<button
								type="button"
								onmousedown={(event) =>
									event.preventDefault()
								}
								onclick={() =>
									selectAttribute(suggestion)
								}
							>
								{suggestion}
							</button>
						</li>
					{/each}
				</ul>
			{/if}
		</div>

		<div class="field">
			<input 
				class= " m-1 rounded-md"
				type="text"
				aria-label={rightLabel}
				placeholder={rightLabel}
				bind:value={rightInput}
				onfocus={() => (rightFocused = true)}
				onkeydown={handleKeydown}
			/>

			{#if rightFocused && rightSuggestions().length > 0}
				<ul class="suggestions">
					{#each rightSuggestions() as suggestion (suggestion)}
						<li>
							<button
								type="button"
								onmousedown={(event) =>
									event.preventDefault()
								}
								onclick={() =>
									selectRight(suggestion)
								}
							>
								{suggestion}
							</button>
						</li>
					{/each}
				</ul>
			{/if}
		</div>
	</div>

	{#if entries.length > 0}
		<div class="tags">
			{#each entries as entry (entry.id)}
				<div class="tag">
					{#if entry.attribute}
						<span>{entry.attribute}</span>
						<span class="separator"></span>
					{/if}

					<span>{entry.value}</span>

					<button
						type="button"
						aria-label={`Remove ${entry.value}`}
						onclick={() => removeEntry(entry.id)}
					>
						×
					</button>
				</div>
			{/each}
		</div>
	{/if}
</div>

{#if currentUnknown}
	<div class="dialog-backdrop">
		<div
			class="dialog"
			role="dialog"
			aria-modal="true"
			aria-labelledby="unknown-vocabulary-title"
		>
			<h2 id="unknown-vocabulary-title">
				Unknown {currentUnknown.field === 'attribute'
					? 'Attribute'
					: rightLabel}
			</h2>

			<p>
				<strong>{currentUnknown.value}</strong>
				is not in the
				<strong>
					{currentUnknown.field === 'attribute'
						? 'Attribute'
						: rightLabel}
				</strong>
				list.
			</p>

			<p>Add it for future use?</p>

			<div class="dialog-actions">
				<button
					type="button"
					onclick={acceptUnknown}
				>
					Yes
				</button>

				<button
					type="button"
					onclick={rejectUnknown}
				>
					No
				</button>
			</div>
		</div>
	</div>
{/if}

{#if missingRightAttribute}
	<div class="dialog-backdrop">
		<div
			class="dialog"
			role="dialog"
			aria-modal="true"
			aria-labelledby="missing-value-title"
		>
			<h2 id="missing-value-title">
				{rightLabel} required
			</h2>

			<p>
				You need to pick a {rightLabel} for the
				Attribute
				<strong>{missingRightAttribute}</strong>.
			</p>

			<div class="dialog-actions">
				<button
					type="button"
					onclick={dismissMissingRight}
				>
					OK
				</button>
			</div>
		</div>
	</div>
{/if}

<style>
	.prompt-pair {
		position: relative;
		margin-right: .4em;
	}

	.prompt-pair-inputs {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 4px;
	}

	.field {
		position: relative;
	}

	input {
		width: 100%;
		box-sizing: border-box;
		background: white;
		margin: .2em;
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

	.tags {
		display: flex;
		flex-wrap: wrap;
		gap: .2rem;
		margin: .2rem;
	}

	.tag {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		padding: 2px 7px;
		border-radius: 10px;
		background: #eee;
	}

	.tag button {
		border: 0;
		background: transparent;
		cursor: pointer;
		font-weight: bold;
	}

	.separator {
		opacity: 0.5;
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