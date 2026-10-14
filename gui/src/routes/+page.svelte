<script lang="ts">
	import { onMount } from 'svelte';

	import Generate from '#lib/components/Generate.svelte';
	import Lyrics from '#lib/components/Lyrics.svelte';
	import NegativeTextPrompt from '#lib/components/NegativeTextPrompt.svelte';
	import PromptPair from '#lib/components/PromptPair.svelte';
	import SongMetadata from '#lib/components/SongMetadata.svelte';
	import SongParts from '#lib/components/SongParts.svelte';
	import TextPrompt from '#lib/components/TextPrompt.svelte';

	import type {
		LoadedProject,
		ProjectOutput,
		ProjectSettings,
		ProjectSummary,
		PromptPairValue
	} from '#lib/types/project.ts';

	type PromptPairEntry = PromptPairValue & {
		id: number;
	};

	let fileInput: HTMLInputElement;
	let selectedFile = $state<File | null>(null);
	let activeProjectId = $state<string | null>(null);
	let loadedAudioName = $state('');

	let showLoadPanel = $state(false);
	let projectList = $state<ProjectSummary[]>([]);
	let selectedProjectToLoad = $state('');
	let loadError = $state('');
	let loadingProject = $state(false);

	let attributeVocabulary = $state<string[]>([]);
	let genreVocabulary = $state<string[]>([]);
	let songTypeVocabulary = $state<string[]>([]);
	let themeVocabulary = $state<string[]>([]);
	let vocalVocabulary = $state<string[]>([]);
	let instrumentVocabulary = $state<string[]>([]);

	let genreEntries = $state<PromptPairEntry[]>([]);
	let songTypeEntries = $state<PromptPairEntry[]>([]);
	let themeEntries = $state<PromptPairEntry[]>([]);
	let vocalEntries = $state<PromptPairEntry[]>([]);
	let instrumentEntries = $state<PromptPairEntry[]>([]);

	let textPrompt = $state('');
	let negativetextPrompt = $state('NO USER INPUT');

	let songPartVocabulary = $state<string[]>([]);
	let songPartEntries = $state<{ id: number; value: string }[]>([]);
	let committedSongPartEntries = $state<{ id: number; value: string }[]>([]);
	let lyricsByPart = $state<Record<string, string>>({});

	let keyVocabulary = $state<string[]>([]);
	let modeVocabulary = $state<string[]>([]);
	let languageVocabulary = $state<string[]>([]);
	let timeSignatureVocabulary = $state<string[]>([]);

	let tempo = $state<number | null>(null);
	let key = $state('');
	let mode = $state('');
	let timeSignature = $state('');
	let duration = $state<number | null>(null);
	let language = $state('');

	let autoTempo = $state(true);
	let autoKey = $state(true);
	let autoTimeSignature = $state(true);
	let autoDuration = $state(true);
	let autoLanguage = $state(true);

	let sourceAudioName = $derived(
		selectedFile?.name ?? loadedAudioName
	);

	function promptPairValue(entry: PromptPairValue): string {
		return [entry.attribute, entry.value]
			.filter(Boolean)
			.join(' ');
	}

	function withoutIds(entries: PromptPairEntry[]): PromptPairValue[] {
		return entries.map(({ attribute, value }) => ({
			attribute,
			value
		}));
	}

	function withIds(entries: PromptPairValue[]): PromptPairEntry[] {
		return entries.map((entry, index) => ({
			id: index + 1,
			attribute: entry.attribute,
			value: entry.value
		}));
	}

	let projectOutput = $derived<ProjectOutput>({
		version: 1,

		source: {
			audio: sourceAudioName
		},

		prompt: {
			text: textPrompt,
			negative: negativetextPrompt
		},

		structure: committedSongPartEntries.map((entry) => ({
			part: entry.value,
			lyrics: lyricsByPart[entry.value] ?? ''
		})),

		metadata: {
			tempo: autoTempo ? null : tempo,
			key: autoKey ? null : key || null,
			mode: autoKey ? null : mode || null,
			timeSignature: autoTimeSignature
				? null
				: timeSignature || null,
			duration: autoDuration ? null : duration,
			language: autoLanguage ? null : language || null
		},

		genres: genreEntries.map(promptPairValue),
		songTypes: songTypeEntries.map(promptPairValue),
		themes: themeEntries.map(promptPairValue),
		vocals: vocalEntries.map(promptPairValue),
		instruments: instrumentEntries.map(promptPairValue)
	});

	let projectSettings = $derived<ProjectSettings>({
		version: 1,

		metadataAuto: {
			tempo: autoTempo,
			key: autoKey,
			mode: autoKey,
			timeSignature: autoTimeSignature,
			duration: autoDuration,
			language: autoLanguage
		},

		promptPairs: {
			genres: withoutIds(genreEntries),
			songTypes: withoutIds(songTypeEntries),
			themes: withoutIds(themeEntries),
			vocals: withoutIds(vocalEntries),
			instruments: withoutIds(instrumentEntries)
		}
	});

	function handleFileChange(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		selectedFile = input.files?.[0] ?? null;

		if (selectedFile) {
			activeProjectId = null;
			loadedAudioName = '';
		}
	}

	function browse() {
		fileInput.click();
	}

	async function openProjectLoader() {
		loadError = '';
		selectedProjectToLoad = '';
		loadingProject = true;

		try {
			const response = await fetch('/api/projects');
			const result = await response.json();

			if (!response.ok) {
				throw new Error(
					result?.error ?? 'Failed to list projects.'
				);
			}

			projectList = result as ProjectSummary[];
			selectedProjectToLoad = projectList[0]?.projectId ?? '';
			showLoadPanel = true;
		} catch (error) {
			loadError =
				error instanceof Error
					? error.message
					: 'Failed to list projects.';
		} finally {
			loadingProject = false;
		}
	}

	function applyLoadedProject(project: LoadedProject) {
		const { output, settings } = project;

		selectedFile = null;
		if (fileInput) {
			fileInput.value = '';
		}

		activeProjectId = project.projectId;
		loadedAudioName = output.source.audio;

		textPrompt = output.prompt.text;
		negativetextPrompt = output.prompt.negative;

		genreEntries = withIds(settings.promptPairs.genres);
		songTypeEntries = withIds(settings.promptPairs.songTypes);
		themeEntries = withIds(settings.promptPairs.themes);
		vocalEntries = withIds(settings.promptPairs.vocals);
		instrumentEntries = withIds(settings.promptPairs.instruments);

		tempo = output.metadata.tempo;
		key = output.metadata.key ?? '';
		mode = output.metadata.mode ?? '';
		timeSignature = output.metadata.timeSignature ?? '';
		duration = output.metadata.duration;
		language = output.metadata.language ?? '';

		autoTempo = settings.metadataAuto.tempo;
		autoKey = settings.metadataAuto.key;
		autoTimeSignature = settings.metadataAuto.timeSignature;
		autoDuration = settings.metadataAuto.duration;
		autoLanguage = settings.metadataAuto.language;

		const structure = output.structure.map((entry, index) => ({
			id: index + 1,
			value: entry.part
		}));

		songPartEntries = structure.map((entry) => ({ ...entry }));
		committedSongPartEntries = structure.map((entry) => ({ ...entry }));
		lyricsByPart = Object.fromEntries(
			output.structure.map((entry) => [entry.part, entry.lyrics])
		);
	}

	async function loadSelectedProject() {
		if (!selectedProjectToLoad) {
			return;
		}

		loadingProject = true;
		loadError = '';

		try {
			const response = await fetch(
				`/api/projects/${encodeURIComponent(selectedProjectToLoad)}`
			);
			const result = await response.json();

			if (!response.ok) {
				throw new Error(
					result?.error ?? 'Failed to load project.'
				);
			}

			applyLoadedProject(result as LoadedProject);
			showLoadPanel = false;
		} catch (error) {
			loadError =
				error instanceof Error
					? error.message
					: 'Failed to load project.';
		} finally {
			loadingProject = false;
		}
	}

	function handleProjectSaved(projectId: string, audioName: string) {
		activeProjectId = projectId;
		loadedAudioName = audioName;
	}

	async function loadVocabulary() {
		const [
			attributes,
			genres,
			songtypes,
			themes,
			vocals,
			songparts,
			keys,
			modes,
			languages,
			instruments,
			timesignatures
		] = await Promise.all([
			fetch('/api/vocabulary/attributes').then((r) => r.json()),
			fetch('/api/vocabulary/genres').then((r) => r.json()),
			fetch('/api/vocabulary/songtypes').then((r) => r.json()),
			fetch('/api/vocabulary/themes').then((r) => r.json()),
			fetch('/api/vocabulary/vocals').then((r) => r.json()),
			fetch('/api/vocabulary/songparts').then((r) => r.json()),
			fetch('/api/vocabulary/keys').then((r) => r.json()),
			fetch('/api/vocabulary/modes').then((r) => r.json()),
			fetch('/api/vocabulary/languages').then((r) => r.json()),
			fetch('/api/vocabulary/instruments').then((r) => r.json()),
			fetch('/api/vocabulary/timesignatures').then((r) => r.json())
		]);

		attributeVocabulary = attributes;
		genreVocabulary = genres;
		songTypeVocabulary = songtypes;
		themeVocabulary = themes;
		vocalVocabulary = vocals;
		songPartVocabulary = songparts;
		instrumentVocabulary = instruments;
		keyVocabulary = keys;
		modeVocabulary = modes;
		languageVocabulary = languages;
		timeSignatureVocabulary = timesignatures;
	}

	onMount(loadVocabulary);
</script>

<div class="bg-olive-100 pl-2 pr-2 pb-1 pt-0.5">
	<div class="rounded-lg mt-0.5 p-1.5 bg-red-200 flex items-center gap-1">
		<label
			for="acapella-file"
			class="bg-red-300 border border-red-800 rounded-l-lg pr-1 pl-1 font-medium"
		>
			Acapella file*
		</label>

		<button
			type="button"
			class="bg-red-100 border border-red-800 text-left rounded-r-lg pr-1 pl-1"
			onclick={browse}
		>
			{sourceAudioName || 'Click to browse'}
		</button>

		<div class="grid grid-cols-5">
			<div class="col-span-1"></div>
			{#if activeProjectId}
				<span class="col-span-2 col-start-2">
					Project: {activeProjectId}
				</span>
			{/if}
			<button
				type="button"
				class="col-span-1 col-start-5 bg-red-100 border border-red-800 rounded-lg pr-2 pl-2 font-medium"
				disabled={loadingProject}
				onclick={openProjectLoader}
			>
				{loadingProject ? 'Loading...' : 'Load Project'}
			</button>
		</div>

		<input
			id="acapella-file"
			bind:this={fileInput}
			type="file"
			accept="audio/*"
			class="hidden"
			onchange={handleFileChange}
		/>
	</div>

	{#if showLoadPanel}
		<div class="load-panel">
			{#if projectList.length === 0}
				<span>No saved projects found.</span>
			{:else}
				<select bind:value={selectedProjectToLoad}>
					{#each projectList as project (project.projectId)}
						<option value={project.projectId}>
							{project.projectId}
						</option>
					{/each}
				</select>

				<button type="button" onclick={loadSelectedProject} disabled={loadingProject}>
					Load selected project
				</button>
			{/if}

			<button type="button" onclick={() => (showLoadPanel = false)}>
				Cancel
			</button>
		</div>
	{/if}

	{#if loadError}
		<div class="load-error">{loadError}</div>
	{/if}

	<div class="grid grid-cols-3 gap-0.5">
		<div class="rounded-b-sm rounded-t-lg mt-0.5 bg-fuchsia-300">
			<PromptPair
				rightLabel="Genre"
				rightVocabulary={genreVocabulary}
				attributeVocabulary={attributeVocabulary}
				attributeKind="attributes"
				rightKind="genres"
				bind:entries={genreEntries}
			/>
		</div>

		<div class="rounded-b-sm rounded-t-lg mt-0.5 bg-purple-200">
			<PromptPair
				rightLabel="Song Type"
				rightVocabulary={songTypeVocabulary}
				attributeVocabulary={attributeVocabulary}
				attributeKind="attributes"
				rightKind="songtypes"
				bind:entries={songTypeEntries}
			/>
		</div>

		<div class="rounded-b-sm rounded-t-lg mt-0.5 bg-rose-300">
			<PromptPair
				rightLabel="Topic"
				rightVocabulary={themeVocabulary}
				attributeVocabulary={attributeVocabulary}
				attributeKind="attributes"
				rightKind="themes"
				bind:entries={themeEntries}
			/>
		</div>
	</div>

	<div class="grid grid-cols-[2fr_4fr] gap-0.5">
		<div class="rounded-b-sm rounded-t-lg mt-0.5 bg-indigo-100">
			<PromptPair
				rightLabel="Lead Vocal"
				rightVocabulary={vocalVocabulary}
				attributeVocabulary={attributeVocabulary}
				attributeKind="attributes"
				rightKind="vocals"
				bind:entries={vocalEntries}
			/>
		</div>

		<div class="bg-amber-200 rounded-b-sm rounded-t-lg mt-0.5">
			<PromptPair
				rightLabel="Instruments"
				rightVocabulary={instrumentVocabulary}
				attributeVocabulary={attributeVocabulary}
				attributeKind="attributes"
				rightKind="instruments"
				bind:entries={instrumentEntries}
			/>
		</div>
	</div>

	<div>
		<SongMetadata
			keyVocabulary={keyVocabulary}
			modeVocabulary={modeVocabulary}
			timeSignatureVocabulary={timeSignatureVocabulary}
			languageVocabulary={languageVocabulary}
			bind:tempo
			bind:autoTempo
			bind:key
			bind:mode
			bind:autoKey
			bind:timeSignature
			bind:autoTimeSignature
			bind:duration
			bind:autoDuration
			bind:language
			bind:autoLanguage
		/>
	</div>

	<div style="margin-top:.1em" class="grid grid-cols-[4fr_3fr] gap-0.5">
		<div
			style="padding-left:.2rem; padding-right: .2em;"
			class="rounded-b-sm rounded-t-lg mt-0.5 bg-orange-100"
		>
			<TextPrompt bind:value={textPrompt} />
		</div>

		<div
			style="padding-left:.2rem; padding-right: .2em;"
			class="rounded-b-sm rounded-t-lg mt-0.5 bg-stone-300"
		>
			<NegativeTextPrompt bind:value={negativetextPrompt} />
		</div>
	</div>

	<div class="grid grid-cols-[1fr_4fr] gap-0.5">
		<div
			style="padding-left:.2rem; padding-right: .2em;"
			class="rounded-b-sm rounded-t-lg mt-0.5 bg-green-100"
		>
			<SongParts
				vocabulary={songPartVocabulary}
				bind:entries={songPartEntries}
				onDone={(entries) => {
					committedSongPartEntries = entries;
				}}
			/>
		</div>

		<div
			style="padding-left:.2rem; padding-right: .2em;"
			class="rounded-b-sm rounded-t-lg mt-0.5 bg-indigo-200"
		>
			<Lyrics
				entries={committedSongPartEntries}
				bind:lyrics={lyricsByPart}
			/>
		</div>
	</div>
</div>

<div class="rounded-lg m-2.5 mt-0.5 text-center border-2 text-violet-900 border-violet-900 bg-violet-100 shadow-md/50">
	<Generate
		selectedFile={selectedFile}
		sourceAudioName={sourceAudioName}
		activeProjectId={activeProjectId}
		textPrompt={textPrompt}
		entries={committedSongPartEntries}
		output={projectOutput}
		settings={projectSettings}
		onProjectSaved={handleProjectSaved}
	/>
</div>

<style>
	.load-panel {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin: 0.3rem 0;
		padding: 0.4rem;
		border: 1px solid #991b1b;
		border-radius: 0.4rem;
		background: #fee2e2;
	}

	.load-panel select {
		min-width: 24rem;
		background: white;
		border: 1px solid #991b1b;
		border-radius: 0.3rem;
		padding: 0.2rem;
	}

	.load-panel button {
		background: white;
		border: 1px solid #991b1b;
		border-radius: 0.3rem;
		padding: 0.2rem 0.5rem;
		cursor: pointer;
	}

	.load-error {
		color: #b91c1c;
		font-weight: 600;
	}
</style>
