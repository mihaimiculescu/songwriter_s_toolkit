export type SongPartEntry = {
	id: number;
	value: string;
};

/*
 * This is the musical/project description.
 *
 * output.json represents WHAT the project contains.
 * It is deliberately platform-agnostic and does not
 * contain ACE-Step-specific generation settings.
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
 * This is the generator configuration.
 *
 * settings.json represents HOW the project was configured.
 *
 * It is intentionally separate from output.json so that
 * the musical output remains platform-agnostic.
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

	generation: {
		keepAcapella: boolean;
	};
};