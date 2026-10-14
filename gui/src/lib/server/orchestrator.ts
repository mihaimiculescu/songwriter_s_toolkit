import net from 'node:net';

const DEFAULT_SOCKET = '/orchestrator-runtime/orchestrator.sock';
const MAX_RESPONSE_BYTES = 64 * 1024;

type OrchestratorResponse = {
	ok: boolean;
	project?: string;
	midi?: string;
	error?: string;
};

export async function runOrchestrator(projectId: string): Promise<OrchestratorResponse> {
	const socketPath = process.env.ORCHESTRATOR_SOCKET || DEFAULT_SOCKET;

	return await new Promise<OrchestratorResponse>((resolve, reject) => {
		const socket = net.createConnection({ path: socketPath });
		let settled = false;
		let response = '';

		function fail(error: Error) {
			if (settled) return;
			settled = true;
			socket.destroy();
			reject(error);
		}

		socket.setEncoding('utf8');

		socket.on('connect', () => {
			socket.write(`${JSON.stringify({ project: projectId })}\n`);
		});

		socket.on('data', (chunk: string) => {
			response += chunk;
			if (Buffer.byteLength(response, 'utf8') > MAX_RESPONSE_BYTES) {
				fail(new Error('Orchestrator response is too large.'));
				return;
			}

			const newline = response.indexOf('\n');
			if (newline === -1) return;

			const line = response.slice(0, newline);
			let parsed: OrchestratorResponse;
			try {
				parsed = JSON.parse(line) as OrchestratorResponse;
			} catch {
				fail(new Error('Host orchestrator returned invalid JSON.'));
				return;
			}

			if (settled) return;
			settled = true;
			socket.end();

			if (!parsed.ok) {
				reject(new Error(parsed.error || 'Host orchestration failed.'));
				return;
			}

			resolve(parsed);
		});

		socket.on('error', (error: NodeJS.ErrnoException) => {
			if (settled) return;
			const detail =
				error.code === 'ENOENT' || error.code === 'ECONNREFUSED'
					? 'Host orchestrator service is not running or its Unix socket is unavailable.'
					: error.message;
			fail(new Error(detail));
		});

		socket.on('end', () => {
			if (!settled) {
				fail(new Error('Host orchestrator closed the connection without a result.'));
			}
		});
	});
}
