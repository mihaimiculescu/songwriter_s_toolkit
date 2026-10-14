import { error, isHttpError } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { cloneProject } from '#lib/server/project.ts';
import type {
	ProjectOutput,
	ProjectSettings
} from '#lib/types/project.ts';

export const POST: RequestHandler = async ({ params, request }) => {
	try {
		const body = await request.json();
		const output = body?.output as ProjectOutput | undefined;
		const settings = body?.settings as ProjectSettings | undefined;

		if (!output || !settings) {
			error(400, 'Missing project output or settings.');
		}

		const result = await cloneProject(
			params.projectId,
			output,
			settings
		);

		return Response.json({
			success: true,
			projectId: result.projectId,
			audioName: result.audioName,
			displayName: result.displayName
		});
	} catch (err) {
		if (isHttpError(err)) {
			throw err;
		}

		return Response.json(
			{
				error:
					err instanceof Error
						? err.message
						: 'Failed to save project as new.'
			},
			{ status: 500 }
		);
	}
};
