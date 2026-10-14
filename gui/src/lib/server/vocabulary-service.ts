import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

import {
	vocabularyConfig,
	type VocabularyType
} from './vocabulary';

function normalize(value: string): string {
	return value
		.trim()
		.toLowerCase()
		.replace(/[-_]+/g, ' ')
		.replace(/\s+/g, ' ');
}

export async function getVocabulary(
	type: VocabularyType
): Promise<string[]> {
	const config = vocabularyConfig[type];

	const filePath = path.join(
		process.cwd(),
		'data',
		'user',
		config.userFile
	);

	let userValues: string[] = [];

	try {
		const content = await readFile(filePath, 'utf8');
		const parsed: unknown = JSON.parse(content);

		if (Array.isArray(parsed)) {
			userValues = parsed.filter(
				(item): item is string =>
					typeof item === 'string'
			);
		}
	} catch {
		userValues = [];
	}

	const merged = [
		...config.builtin,
		...userValues
	];

	const seen = new Set<string>();

	return merged.filter((item) => {
		const key = normalize(item);

		if (seen.has(key)) {
			return false;
		}

		seen.add(key);
		return true;
	});
}

export async function addVocabularyValue(
	type: VocabularyType,
	value: string
): Promise<void> {
	const config = vocabularyConfig[type];

	const filePath = path.join(
		process.cwd(),
		'data',
		'user',
		config.userFile
	);

	let values: string[] = [];

	try {
		const content = await readFile(filePath, 'utf8');
		const parsed: unknown = JSON.parse(content);

		if (Array.isArray(parsed)) {
			values = parsed.filter(
				(item): item is string =>
					typeof item === 'string'
			);
		}
	} catch {
		values = [];
	}

	const normalizedValue = normalize(value);

	const exists = values.some(
		(item) => normalize(item) === normalizedValue
	);

	if (!exists) {
		values.push(value);

		await writeFile(
			filePath,
			JSON.stringify(values, null, 2) + '\n'
		);
	}
}