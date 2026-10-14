<script lang="ts">
	type SongPartEntry = {
		id: number;
		value: string;
	};

	interface Props {
		selectedFile: File | null;
		sourceAudioName: string;
		activeProjectId: string | null;
		textPrompt: string;
		entries: SongPartEntry[];
		output: Record<string, unknown>;
		settings: Record<string, unknown>;
		onProjectSaved?: (projectId: string, audioName: string) => void;
	}

	let {
		selectedFile,
		sourceAudioName,
		activeProjectId,
		textPrompt,
		entries,
		output,
		settings,
		onProjectSaved
	}: Props = $props();

	let generating = $state(false);
	let errorMessage = $state('');
	let successMessage = $state('');
	let showSaveChoice = $state(false);

	function validate(): string | null {
		if (!sourceAudioName) {
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

	function generate() {
		errorMessage = '';
		successMessage = '';

		const validationError = validate();
		if (validationError) {
			errorMessage = validationError;
			return;
		}

		if (activeProjectId) {
			showSaveChoice = true;
			return;
		}

		void saveNewFromUpload();
	}

	async function readResult(response: Response, fallback: string) {
		const result = await response.json();
		if (!response.ok) {
			throw new Error(result?.error ?? fallback);
		}
		return result;
	}

	async function generateSavedProject(projectId: string) {
		return await readResult(
			await fetch(
				`/api/projects/${encodeURIComponent(projectId)}/generate`,
				{ method: 'POST' }
			),
			'Project was saved, but generation failed.'
		);
	}

	async function saveNewFromUpload() {
		if (!selectedFile) {
			errorMessage = 'Please load an acapella file first.';
			return;
		}

		generating = true;
		let savedProjectId: string | null = null;

		try {
			const formData = new FormData();
			formData.append('audioFile', selectedFile, selectedFile.name);
			formData.append('output', JSON.stringify(output));
			formData.append('settings', JSON.stringify(settings));

			const result = await readResult(
				await fetch('/api/projects', {
					method: 'POST',
					body: formData
				}),
				'Failed to save project.'
			);

			savedProjectId = result.projectId;
			onProjectSaved?.(result.projectId, result.audioName);
			await generateSavedProject(result.projectId);
			successMessage = `Project ${result.displayName} generated`;
		} catch (error) {
			const message =
				error instanceof Error ? error.message : 'Generation failed.';
			errorMessage = savedProjectId
				? `Project ${savedProjectId} was saved, but generation failed: ${message}`
				: message;
		} finally {
			generating = false;
		}
	}

	async function saveExisting(mode: 'overwrite' | 'clone') {
		if (!activeProjectId) return;

		showSaveChoice = false;
		generating = true;
		errorMessage = '';
		successMessage = '';
		let savedProjectId: string | null = null;

		try {
			const encodedId = encodeURIComponent(activeProjectId);
			const endpoint =
				mode === 'overwrite'
					? `/api/projects/${encodedId}`
					: `/api/projects/${encodedId}/clone`;

			const result = await readResult(
				await fetch(endpoint, {
					method: mode === 'overwrite' ? 'PUT' : 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ output, settings })
				}),
				mode === 'overwrite'
					? 'Failed to overwrite project.'
					: 'Failed to save project as new.'
			);

			savedProjectId = result.projectId;
			onProjectSaved?.(result.projectId, result.audioName);
			await generateSavedProject(result.projectId);

			successMessage =
				mode === 'overwrite'
					? `Project ${result.projectId} overwritten and generated`
					: `Project ${result.displayName} saved as new and generated`;
		} catch (error) {
			const message =
				error instanceof Error ? error.message : 'Generation failed.';
			errorMessage = savedProjectId
				? `Project ${savedProjectId} was saved, but generation failed: ${message}`
				: message;
		} finally {
			generating = false;
		}
	}
</script>

<div class="text-center text-2xl font-extrabold">
	<button
		type="button"
		class="generate-button"
		disabled={generating}
		onclick={generate}
	>
		{#if generating}
			Generating...
		{:else}
			Generate
		{/if}
	</button>

	{#if showSaveChoice}
		<div class="save-choice">
			<div>This project already exists. Save changes how?</div>
			<div class="save-choice-buttons">
				<button type="button" onclick={() => saveExisting('overwrite')}>
					Overwrite
				</button>
				<button type="button" onclick={() => saveExisting('clone')}>
					Save as new project
				</button>
				<button type="button" onclick={() => (showSaveChoice = false)}>
					Cancel
				</button>
			</div>
		</div>
	{/if}

	{#if errorMessage}
		<div class="error-message">{errorMessage}</div>
	{/if}

	{#if successMessage}
		<div class="success-message">{successMessage}</div>
	{/if}
</div>

<style>
	.generate-button {
		cursor: pointer;
		min-width: 100%;
	}

	.generate-button:disabled {
		opacity: 0.6;
		cursor: default;
	}

	.save-choice {
		margin-top: 0.4rem;
		font-size: 1rem;
		font-weight: 600;
	}

	.save-choice-buttons {
		display: flex;
		justify-content: center;
		gap: 0.5rem;
		margin-top: 0.3rem;
	}

	.save-choice-buttons button {
		padding: 0.2rem 0.6rem;
		border: 1px solid currentColor;
		border-radius: 0.35rem;
		background: white;
		cursor: pointer;
	}

	.error-message {
		color: #b91c1c;
	}

	.success-message {
		color: #376348;
	}
</style>
