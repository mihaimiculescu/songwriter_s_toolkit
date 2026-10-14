import { error, isHttpError } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { writeFile, unlink } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { saveProject } from '#lib/server/project.ts';
import type {
	ProjectOutput,
	ProjectSettings
} from '#lib/types/project.ts';

export const POST: RequestHandler = async ({ request }) => {
	let temporaryAudioPath: string | null = null;

	try {
		const formData = await request.formData();

		const audioFile = formData.get('audioFile');
		const outputValue = formData.get('output');
		const settingsValue = formData.get('settings');

		if (!(audioFile instanceof File)) {
			error(400, 'No acapella file was uploaded.');
		}

		if (typeof outputValue !== 'string') {
			error(400, 'Missing project output.');
		}

		if (typeof settingsValue !== 'string') {
			error(400, 'Missing project settings.');
		}

		let output: ProjectOutput;
		let settings: ProjectSettings;

		try {
			output = JSON.parse(outputValue);
		} catch {
			error(400, 'Invalid project output JSON.');
		}

		try {
			settings = JSON.parse(settingsValue);
		} catch {
			error(400, 'Invalid project settings JSON.');
		}

		/*
		 * The uploaded File exists in the request, but
		 * saveProject() expects a filesystem path.
		 *
		 * Write it to a temporary file first.
		 */
		const tempName =
			`songwriter-toolkit-${Date.now()}-${audioFile.name}`;

		temporaryAudioPath = path.join(
			os.tmpdir(),
			tempName
		);

		const audioBuffer =
			Buffer.from(await audioFile.arrayBuffer());

		await writeFile(
			temporaryAudioPath,
			audioBuffer
		);

		/*
		 * Now saveProject() can copy the temporary file
		 * into the final project directory.
		 */
		const result = await saveProject(
			temporaryAudioPath,
			audioFile.name,
			output,
			settings
		);

		return Response.json({
			success: true,
			projectId: result.projectId,
			displayName: result.displayName
		});
	} catch (err) {
		console.error('Project save failed:', err);

		if (isHttpError(err)) {
			throw err;
		}

		return Response.json(
			{
				error:
					err instanceof Error
						? err.message
						: 'Failed to save project.'
			},
			{
				status: 500
			}
		);
	} finally {
		/*
		 * The temporary upload is no longer needed after
		 * saveProject() has copied it into the project folder.
		 */
		if (temporaryAudioPath) {
			try {
				await unlink(temporaryAudioPath);
			} catch {
				/*
				 * Nothing to do if the temporary file was
				 * already removed or never created.
				 */
			}
		}
	}
};
