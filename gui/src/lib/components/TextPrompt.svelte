<script lang="ts">
	let {
		value = $bindable('')
	}: {
		value?: string;
	} = $props();
</script>
<div class="text-prompt">
	<h3 style="margin:.2em" class="font-medium">Ia zi-i, tati, ce vrei de la sufletul meu?</h3>
		<textarea class="text-area"
			bind:value
			placeholder="Add text ..."
			spellcheck="true"
			aria-label="Text prompt"
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