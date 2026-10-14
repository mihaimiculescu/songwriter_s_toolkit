import { genattributes } from '#lib/data/genattributes.ts';
import { genres } from '#lib/data/genres.ts';
import { songparts } from '#lib/data/songparts.ts';
import { songtypes } from '#lib/data/songtypes.ts';
import { themes } from '#lib/data/themes.ts';
import { keys } from '#lib/data/keys.ts';
import { modes } from '#lib/data/modes.ts';
import { languages } from '#lib/data/languages.ts';
import { timesignatures } from '#lib/data/timesignatures.ts';
import { vocals } from '#lib/data/vocals.ts';
import { instruments } from '#lib/data/instruments.ts';

export const vocabularyConfig = {
	attributes: {
		builtin: [...genattributes],
		userFile: 'user-genattributes.json'
	},
	genres: {
		builtin: [...genres],
		userFile: 'user-genres.json'
	},
	songtypes: {
		builtin: [...songtypes],
		userFile: 'user-songtypes.json'
	},
	themes: {
		builtin: [...themes],
		userFile: 'user-themes.json'
	},
	vocals: {
		builtin: [...vocals],
		userFile: 'user-vocals.json'
	},
	songparts: {
		builtin: [...songparts],
		userFile: 'user-songparts.json'
	},
	keys: {
		builtin: [...keys],
		userFile: 'user-keys.json'
	},
	modes: {
		builtin: [...modes],
		userFile: 'user-modes.json'
	},
	languages: {
		builtin: [...languages],
		userFile: 'user-languages.json'
	},
	timesignatures: {
		builtin: [...timesignatures],
		userFile: 'user-timesignatures.json'
	},
	instruments: {
		builtin: [...instruments],
		userFile: 'user-instruments.json'
	}
} as const;

export type VocabularyType = keyof typeof vocabularyConfig;