export type SongPartEntry = {
	id: number;
	value: string;
};

export type PromptPairValue = {
	attribute: string;
	value: string;
};

/*
 * output.json is the downstream musical/project description.
 *
 * It describes WHAT should be generated and deliberately avoids
 * browser/UI-only state.
 */
export type ProjectOutput = {
	version: 1;

	source: {
		audio: string;
	};

	prompt: {
		text: string;
		negative: string;
	};

	structure: Array<{
		part: string;
		lyrics: string;
	}>;

	metadata: {
		tempo: number | null;
		key: string | null;
		mode: string | null;
		timeSignature: string | null;
		duration: number | null;
		language: string | null;
	};

	genres: string[];
	songTypes: string[];
	themes: string[];
	vocals: string[];
	instruments: string[];
};

/*
 * settings.json is the GUI reload state.
 *
 * output.json intentionally flattens prompt pairs and uses null for
 * metadata fields that should be detected automatically. Those choices
 * are lossy from an editor point of view, so settings.json stores exactly
 * the additional state required to reconstruct the GUI.
 */
export type ProjectSettings = {
	version: 1;

	metadataAuto: {
		tempo: boolean;
		key: boolean;
		mode: boolean;
		timeSignature: boolean;
		duration: boolean;
		language: boolean;
	};

	promptPairs: {
		genres: PromptPairValue[];
		songTypes: PromptPairValue[];
		themes: PromptPairValue[];
		vocals: PromptPairValue[];
		instruments: PromptPairValue[];
	};
};

export type ProjectSummary = {
	projectId: string;
	audioName: string;
	modifiedAt: string;
};

export type LoadedProject = {
	projectId: string;
	output: ProjectOutput;
	settings: ProjectSettings;
};
