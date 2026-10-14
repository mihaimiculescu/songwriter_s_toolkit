<script lang="ts">
	let {
		value = $bindable('NO USER INPUT')
	}: {
		value?: string;
	} = $props();
</script>
<div class="text-prompt">
	<h3 style="margin:.2em" class="font-medium">Așa, bun, și de ce să mă feresc?</h3>
		<textarea class="text-area"
			bind:value
			placeholder="Add negative text ..."
			spellcheck="true"
			aria-label="Negative text prompt"
			rows="4"
		>
		</textarea>
</div>

<style>
	.text-prompt {
		position: relative;
	}
	.text-area {
		margin-top: .5rem;
		width: 100%;
		box-sizing: border-box;
	    background-color: white;

	}
</style>