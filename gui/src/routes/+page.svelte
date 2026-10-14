<script lang="ts">
	import { onMount } from 'svelte';

	import PromptPair from '#lib/components/PromptPair.svelte';

	let fileInput: HTMLInputElement;
	let selectedFile = $state<File | null>(null);

	let attributeVocabulary = $state<string[]>([]);
	let genreVocabulary = $state<string[]>([]);
	let songTypeVocabulary = $state<string[]>([]);
	let themeVocabulary = $state<string[]>([]);
	let vocalVocabulary = $state<string[]>([]);
	let instrumentVocabulary = $state<string[]>([]);

	type PromptPairEntry = {
		id: number;
		attribute: string;
		value: string;
	};

	let genreEntries = $state<PromptPairEntry[]>([]);
	let songTypeEntries = $state<PromptPairEntry[]>([]);
	let themeEntries = $state<PromptPairEntry[]>([]);
	let vocalEntries = $state<PromptPairEntry[]>([]);
	let instrumentEntries = $state<PromptPairEntry[]>([]);

/* 	let genreEntries = $state([]);
	let songTypeEntries = $state([]);
	let themeEntries = $state([]);
	let vocalEntries = $state([]);
	let instrumentEntries = $state([]);
 */
	import TextPrompt from '#lib/components/TextPrompt.svelte';
	let textPrompt = $state('');

	import NegativeTextPrompt from '#lib/components/NegativeTextPrompt.svelte';
	let negativetextPrompt = $state('NO USER INPUT');


	import SongParts from '#lib/components/SongParts.svelte';
	let songPartVocabulary = $state<string[]>([]);
	let songPartEntries = $state<{ id: number; value: string }[]>([]);
	let committedSongPartEntries = $state<{ id: number; value: string }[]>([]);

	import Lyrics from '#lib/components/Lyrics.svelte';
	let lyricsByPart = $state<Record<string, string>>({});

	import SongMetadata from '#lib/components/SongMetadata.svelte';

	
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

	import Generate from '#lib/components/Generate.svelte'

	import type {
		ProjectOutput,
		ProjectSettings
	} from '#lib/types/project.ts';	

	function promptPairValue(entry: {
		attribute: string;
		value: string;
	}): string {
		return [entry.attribute, entry.value]
			.filter(Boolean)
			.join(' ');
	}

	let projectOutput = $derived<ProjectOutput>({
		version: 1,

		source: {
			audio: selectedFile?.name ?? ''
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
			language: autoLanguage
				? null
				: language || null
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

		generation: {
			keepAcapella: false
		}
	});

	function handleFileChange(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		selectedFile = input.files?.[0] ?? null;
	}

	function browse() {
		fileInput.click();
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
	<div class="rounded-lg mt-0.5 p-1.5  bg-red-200 flex items-center">
		<label for="acapella-file" class="bg-red-300 border border-red-800 rounded-l-lg pr-1 pl-1 font-medium">
			Acapella file*
		</label>

		<button
			type="button"
			class="bg-red-100  border border-red-800 text-left rounded-r-lg pr-1 pl-1"
			onclick={browse}
		>
			{selectedFile ? selectedFile.name : 'Click to browse'}
		</button>

		<input
			id="acapella-file"
			bind:this={fileInput}
			type="file"
			accept="audio/*"
			class="hidden"
			onchange={handleFileChange}
		/>
	</div>

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


	<div >
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
		<div  style="padding-left:.2rem; padding-right: .2em;" class="rounded-b-sm rounded-t-lg mt-0.5 bg-orange-100">
		<TextPrompt bind:value={textPrompt} />
		</div>

		<div  style="padding-left:.2rem; padding-right: .2em;" class="rounded-b-sm rounded-t-lg mt-0.5 bg-stone-300">
		<NegativeTextPrompt bind:value={negativetextPrompt} />
		</div>

	</div>
	<div class="grid grid-cols-[1fr_4fr] gap-0.5">
		<div  style="padding-left:.2rem; padding-right: .2em;" class="rounded-b-sm rounded-t-lg mt-0.5 bg-green-100">
		<SongParts
		vocabulary={songPartVocabulary}
		bind:entries={songPartEntries}
		onDone={(entries) => {
			committedSongPartEntries = entries;
		}}
		/>
		</div>

		<div  style="padding-left:.2rem; padding-right: .2em;" class="rounded-b-sm rounded-t-lg mt-0.5 bg-indigo-200" >
		<Lyrics
			entries={committedSongPartEntries}
			bind:lyrics={lyricsByPart}
		/>
		</div>
	</div>	
</div>
<div class="rounded-lg m-2.5 mt-0.5 text-center border-4 text-yellow-50 border-violet-900 bg-violet-600 shadow-md/50">
	<Generate
		selectedFile={selectedFile}
		textPrompt={textPrompt}
		entries={committedSongPartEntries}
		output={projectOutput}
		settings={projectSettings}
	/>
</div>