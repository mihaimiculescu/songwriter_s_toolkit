<script lang="ts">
	interface Props {
		keyVocabulary: string[];
		modeVocabulary: string[];
		timeSignatureVocabulary: string[];
		languageVocabulary: string[];

		tempo?: number | null;
		key?: string;
        mode?: string;
		timeSignature?: string;
		duration?: number | null;
		language?: string;

		autoTempo?: boolean;
		autoKey?: boolean;
		autoTimeSignature?: boolean;
		autoDuration?: boolean;
		autoLanguage?: boolean;
	}

	let {
		keyVocabulary = [],
		modeVocabulary = [],
		timeSignatureVocabulary = [],
		languageVocabulary = [],

		tempo = $bindable(null),
		key = $bindable(''),
        mode = $bindable(''),
		timeSignature = $bindable(''),
		duration = $bindable(null),
		language = $bindable(''),

		autoTempo = $bindable(true),
		autoKey = $bindable(true),
		autoTimeSignature = $bindable(true),
		autoDuration = $bindable(true),
		autoLanguage = $bindable(true)
	}: Props = $props();

</script>

<div class="bg-slate-400 rounded-b-sm rounded-t-lg mt-0.5 pr-4 song-metadata">
	<!-- Tempo -->
	<div class="metadata-field">
		<div class="metadata-label">
			<label for="tempo" class="m-1">Tempo</label>

			<label class="auto-option">
				<input
					type="checkbox"
					bind:checked={autoTempo}
				/>
				Auto
			</label>
		</div>

		<input
			class="bg-gray-50 m-1 rounded-md"
			id="tempo"
			type="number"
			min="1"
			step="1"
			bind:value={tempo}
			disabled={autoTempo}
			placeholder="BPM"
		/>
	</div>

    <!-- Key -->
    <div class="metadata-field">
        <div class="metadata-label">
            <label for="key" class="m-1">Key</label>

            <label class="auto-option">
                <input
                    type="checkbox"
                    bind:checked={autoKey}
                />
                Auto
            </label>
        </div>

        <div class="key-selects">
            <select
				class="bg-gray-50 m-1 rounded-md"
                id="key"
                bind:value={key}
                disabled={autoKey}
            >
                <option value="" disabled>
                    Key
                </option>

                {#each keyVocabulary as option (option)}
                    <option value={option}>
                        {option}
                    </option>
                {/each}
            </select>

            <select
				class="bg-gray-50 m-1 rounded-md"
                id="mode"
                bind:value={mode}
                disabled={autoKey}
            >
                <option value="" disabled>
                    Mode
                </option>

                {#each modeVocabulary as option (option)}
                    <option value={option}>
                        {option}
                    </option>
                {/each}
            </select>
        </div>
    </div>

	<!-- Time signature -->
	<div class="metadata-field">
		<div class="metadata-label">
			<label for="time-signature" class="m-1">
				Time signature
			</label>

			<label class="auto-option">
				<input
					type="checkbox"
					bind:checked={autoTimeSignature}
				/>
				Auto
			</label>
		</div>

		<select
			class="bg-gray-50 m-1 rounded-md"
			id="time-signature"
			bind:value={timeSignature}
			disabled={autoTimeSignature}
		>
			<option value="" disabled>
				Select time signature
			</option>

			{#each timeSignatureVocabulary as option (option)}
				<option value={option}>
					{option}
				</option>
			{/each}
		</select>
	</div>

	<!-- Duration -->
	<div class="metadata-field">
		<div class="metadata-label">
			<label for="duration" class="m-1">Duration</label>

			<label class="auto-option">
				<input
					type="checkbox"
					bind:checked={autoDuration}
				/>
				Auto
			</label>
		</div>

		<input
			class="bg-gray-50 m-1 rounded-md"
			id="duration"
			type="number"
			min="1"
			step="1"
			bind:value={duration}
			disabled={autoDuration}
			placeholder="Seconds"
		/>
	</div>

	<!-- Language -->
	<div class="metadata-field">
		<div class="metadata-label">
			<label for="language" class="m-1">Language</label>

			<label class="auto-option">
				<input
					type="checkbox"
					bind:checked={autoLanguage}
				/>
				Auto
			</label>
		</div>

		<select
			class="bg-gray-50 m-1 rounded-md"
			id="language"
			bind:value={language}
			disabled={autoLanguage}
		>
			<option value="" disabled>
				Select language
			</option>

			{#each languageVocabulary as option (option)}
				<option value={option}>
					{option}
				</option>
			{/each}
		</select>
	</div>
</div>
<style>
    .key-selects {
        display: flex;
        gap: 0.5rem;    
    }

    .key-selects select {
        flex: 1;
        min-width: 0;
    }

	.song-metadata {
		display: flex;
		align-items: flex-start;
		gap: 0.75rem;
		width: 100%;
	}

	.metadata-field {
		flex: 1 1 0;
		min-width: 0;
	}

	.metadata-label {
		display: flex;
		align-items: center;
		justify-content: flex-start;
		gap: 2rem;
		margin-bottom: 0.5rem;
	}

	.metadata-field input,
	.metadata-field select {
		width: 100%;
		box-sizing: border-box;
	}

	.auto-option {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		font-size: 0.875rem;
		font-weight: normal;
		white-space: nowrap;
	}

	.key-selects {
		display: flex;
		gap: 0.5rem;
		width: 100%;
	}

	.key-selects select {
		flex: 1 1 0;
		min-width: 0;
	}

	.metadata-field input:disabled,
	.metadata-field select:disabled {
		opacity: 0.55;
		cursor: not-allowed;
	}
</style>