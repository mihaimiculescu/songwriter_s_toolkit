<script lang="ts">
	type SongPartEntry = {
		id: number;
		value: string;
	};

	interface Props {
		entries: SongPartEntry[];
		lyrics: Record<string, string>;
	}

	let {
		entries,
		lyrics = $bindable({})
	}: Props = $props();

	function updateLyrics(
		part: string,
		value: string
	) {
		lyrics = {
			...lyrics,
			[part]: value
		};
	}
</script>

<div class="mb-1">
    <h3 style="margin:.2em" class="font-medium">Lyrics</h3>
    {#if entries.length === 0}
        <div class="bg-neutral-100 empty-message">
            Build the Structure list first.
        </div>
    {:else}
        <div class="lyrics-list">
            {#each entries as entry (entry.id)}
                <div class="bg-indigo-100 min-w-1/4 lyrics-part">
                    <label for={`lyrics-${entry.id}`}>
                        {entry.value}
                    </label>

					<textarea
						style="background-color:white;"
						id={`lyrics-${entry.id}`}
						placeholder="Add lyrics ..."
						value={lyrics[entry.value] ?? ''}
						oninput={(event) =>
							updateLyrics(
								entry.value,
								(event.currentTarget as HTMLTextAreaElement)
									.value
                            )
                        }
                        rows="4"
                    ></textarea>
                </div>
            {/each}
        </div>
    {/if}
</div>

<style>
    .lyrics-part {
        display: flex;
        flex-direction: column;
        padding-top: 0.5rem;
        margin-top: 0.2rem;
    }
    .empty-message {
        min-height: 6.5rem;
    }
    .lyrics-part label {
        display: block;
    }
</style>