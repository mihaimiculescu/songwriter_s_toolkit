<script lang="ts">
	type SongPartEntry = {
		id: number;
		value: string;
	};

	interface Props {
		selectedFile: File | null;
		textPrompt: string;
		entries: SongPartEntry[];
		output: Record<string, unknown>;
		settings: Record<string, unknown>;
	}

	let {
		selectedFile,
		textPrompt,
		entries,
		output,
		settings
	}: Props = $props();

	let generating = $state(false);
	let errorMessage = $state('');
	let successMessage = $state('');

	function validate(): string | null {
		if (!selectedFile) {
			return 'Please load an acapella file first.';
		}

		if (entries.length === 0) {
			return 'Please build the Structure list first.';
		}

		if (!textPrompt.trim()) {
			return 'Please enter a text prompt.';
		}

		return null;
	}

	async function generate() {
		errorMessage = '';
		successMessage = '';

		const validationError = validate();

		if (validationError) {
			errorMessage = validationError;
			return;
		}

		generating = true;

		try {
			const formData = new FormData();

			formData.append(
				'audioFile',
				selectedFile!,
				selectedFile!.name
			);

			formData.append(
				'output',
				JSON.stringify(output)
			);

			formData.append(
				'settings',
				JSON.stringify(settings)
			);

			const response = await fetch(
				'/api/projects',
				{
					method: 'POST',
					body: formData
				}
			);

			const result = await response.json();

			if (!response.ok) {
				throw new Error(
					result?.error ??
						'Failed to save project.'
				);
			}

			successMessage =
				`Project ${result.displayName} saved`;
		} catch (error) {
			errorMessage =
				error instanceof Error
					? error.message
					: 'Failed to save project.';
		} finally {
			generating = false;
		}
	}
</script>

<div class="text-center text-2xl font-extrabold">
	<button
		type="button"
		class=" generate-button"
		disabled={generating}
		onclick={generate}
	>
		{#if generating}
			Saving...
		{:else}
			Generate
		{/if}
	</button>

	{#if errorMessage}
		<div class="error-message">
			{errorMessage}
		</div>
	{/if}

	{#if successMessage}
		<div class="success-message">
			{successMessage}
		</div>
	{/if}
</div>

<style>
/* 	.generate-area {
		display: flex;
		align-items: center;
		gap: 1rem;
		margin-top: 1rem;
	} */

	.generate-button {
		/* padding: 0.6rem 1.5rem; */
		cursor: pointer;
        min-width: 100%;
	}

	.generate-button:disabled {
		opacity: 0.6;
		cursor: default;
	}

	.error-message {
		color: #b91c1c;
	}

	.success-message {
		color: #376348;
	}
</style>