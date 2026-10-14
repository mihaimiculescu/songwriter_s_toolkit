import type { RequestHandler } from './$types';
import { runOrchestrator } from '#lib/server/orchestrator.ts';

export const POST: RequestHandler = async ({ params }) => {
	try {
		const result = await runOrchestrator(params.projectId);
		return Response.json(result);
	} catch (error) {
		console.error('Project generation failed:', error);
		return Response.json(
			{
				ok: false,
				error:
					error instanceof Error
						? error.message
						: 'Project generation failed.'
			},
			{ status: 500 }
		);
	}
};
