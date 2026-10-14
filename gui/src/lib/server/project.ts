import { mkdir, writeFile, copyFile } from 'node:fs/promises';
import path from 'node:path';
import type { ProjectOutput, ProjectSettings } from '#lib/types/project.ts';

function pad(value: number): string {
	return String(value).padStart(2, '0');
}

export function makeProjectId(
	date = new Date()
): string {
	return [
		date.getDate(),
		date.getMonth() + 1,
		date.getFullYear()
	]
		.map(pad)
		.join('') +
		'_' +
		pad(date.getHours()) +
		pad(date.getMinutes());
}

export function makeProjectDisplayName(
	audioName: string,
	date = new Date()
): string {
	const name = path.parse(audioName).name;

	const months = [
		'Jan',
		'Feb',
		'Mar',
		'Apr',
		'May',
		'Jun',
		'Jul',
		'Aug',
		'Sep',
		'Oct',
		'Nov',
		'Dec'
	];

	return `${name} - ${date.getDate()} ${
		months[date.getMonth()]
	} ${date.getFullYear()} at ${pad(
		date.getHours()
	)}:${pad(date.getMinutes())}`;
}

export async function saveProject(
	audioFilePath: string,
	audioFileName: string,
	output: ProjectOutput,
	settings: ProjectSettings
) {
	const now = new Date();

	const projectId = makeProjectId(now);
	const projectName = path.parse(audioFileName).name;
	const configuredProjectsRoot =
		process.env.SONGWRITER_PROJECTS_ROOT;

	const projectsRoot = configuredProjectsRoot
		? path.resolve(process.cwd(), configuredProjectsRoot)
		: path.resolve(process.cwd(), '..', '..', 'Projects');

	const projectDir = path.join(
		projectsRoot,
		`${projectName}_${projectId}`
	);

	await mkdir(projectDir, {
		recursive: true
	});

	await writeFile(
		path.join(projectDir, 'output.json'),
		JSON.stringify(output, null, 2),
		'utf8'
	);

	await writeFile(
		path.join(projectDir, 'settings.json'),
		JSON.stringify(settings, null, 2),
		'utf8'
	);

	await copyFile(
		audioFilePath,
		path.join(projectDir, audioFileName)
	);

	return {
		projectId,
		projectDir,
		displayName: makeProjectDisplayName(
			audioFileName,
			now
		)
	};
}
