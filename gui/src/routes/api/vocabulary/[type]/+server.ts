import type { RequestHandler } from './$types';

import {
	getVocabulary,
	addVocabularyValue
} from '#lib/server/vocabulary-service.ts';

import {
	vocabularyConfig,
	type VocabularyType
} from '#lib/server/vocabulary.ts';

function isVocabularyType(
	value: string
): value is VocabularyType {
	return value in vocabularyConfig;
}

export const GET: RequestHandler = async ({ params }) => {
	const type = params.type;

	if (!type || !isVocabularyType(type)) {
		return Response.json(
			{ error: 'Invalid vocabulary type' },
			{ status: 400 }
		);
	}

	const values = await getVocabulary(type);

	return Response.json(values);
};

export const POST: RequestHandler = async ({
	params,
	request
}) => {
	const type = params.type;

	if (!type || !isVocabularyType(type)) {
		return Response.json(
			{ error: 'Invalid vocabulary type' },
			{ status: 400 }
		);
	}

	const body = await request.json();

	if (
		typeof body.value !== 'string' ||
		!body.value.trim()
	) {
		return Response.json(
			{ error: 'Invalid value' },
			{ status: 400 }
		);
	}

	await addVocabularyValue(
		type,
		body.value.trim()
	);

	return Response.json({ ok: true });
};
