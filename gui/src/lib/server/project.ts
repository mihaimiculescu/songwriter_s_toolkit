import {
	access,
	copyFile,
	mkdir,
	readdir,
	readFile,
	stat,
	writeFile
} from 'node:fs/promises';
import path from 'node:path';
import type {
	LoadedProject,
	ProjectOutput,
	ProjectSettings,
	ProjectSummary
} from '#lib/types/project.ts';

function pad(value: number): string {
	return String(value).padStart(2, '0');
}

export function makeProjectId(date = new Date()): string {
	return (
		[date.getDate(), date.getMonth() + 1, date.getFullYear()]
			.map(pad)
			.join('') +
		'_' +
		pad(date.getHours()) +
		pad(date.getMinutes())
	);
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

	return `${name} - ${date.getDate()} ${months[date.getMonth()]} ${date.getFullYear()} at ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function projectsRoot(): string {
	const configured = process.env.SONGWRITER_PROJECTS_ROOT;

	if (!configured) {
		throw new Error(
			'SONGWRITER_PROJECTS_ROOT is not configured.'
		);
	}

	return path.resolve(configured);
}

function projectDirectory(projectId: string): string {
	if (
		!projectId ||
		projectId === '.' ||
		projectId === '..' ||
		path.basename(projectId) !== projectId
	) {
		throw new Error('Invalid project id.');
	}

	return path.join(projectsRoot(), projectId);
}

async function writeProjectFiles(
	projectDir: string,
	output: ProjectOutput,
	settings: ProjectSettings
): Promise<void> {
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
}

async function readProjectFiles(
	projectId: string
): Promise<LoadedProject> {
	const projectDir = projectDirectory(projectId);

	const [outputText, settingsText] = await Promise.all([
		readFile(path.join(projectDir, 'output.json'), 'utf8'),
		readFile(path.join(projectDir, 'settings.json'), 'utf8')
	]);

	const output = JSON.parse(outputText) as ProjectOutput;
	const settings = JSON.parse(settingsText) as ProjectSettings;

	if (!output?.source?.audio) {
		throw new Error('Project has no source audio recorded.');
	}

	await access(path.join(projectDir, output.source.audio));

	return {
		projectId,
		output,
		settings
	};
}

export async function saveProject(
	audioFilePath: string,
	audioFileName: string,
	output: ProjectOutput,
	settings: ProjectSettings
) {
	const now = new Date();
	const projectId = `${path.parse(audioFileName).name}_${makeProjectId(now)}`;
	const projectDir = projectDirectory(projectId);

	await mkdir(projectDir, { recursive: false });

	const normalizedOutput: ProjectOutput = {
		...output,
		source: {
			audio: audioFileName
		}
	};

	await writeProjectFiles(projectDir, normalizedOutput, settings);
	await copyFile(audioFilePath, path.join(projectDir, audioFileName));

	return {
		projectId,
		projectDir,
		audioName: audioFileName,
		displayName: makeProjectDisplayName(audioFileName, now)
	};
}

export async function listProjects(): Promise<ProjectSummary[]> {
	const root = projectsRoot();
	await mkdir(root, { recursive: true });

	const entries = await readdir(root, { withFileTypes: true });
	const projects: ProjectSummary[] = [];

	for (const entry of entries) {
		if (!entry.isDirectory()) {
			continue;
		}

		try {
			const loaded = await readProjectFiles(entry.name);
			const info = await stat(projectDirectory(entry.name));

			projects.push({
				projectId: entry.name,
				audioName: loaded.output.source.audio,
				modifiedAt: info.mtime.toISOString()
			});
		} catch {
			// Ignore unrelated/incomplete directories under Projects.
		}
	}

	return projects.sort(
		(a, b) =>
			new Date(b.modifiedAt).getTime() -
			new Date(a.modifiedAt).getTime()
	);
}

export async function loadProject(
	projectId: string
): Promise<LoadedProject> {
	return readProjectFiles(projectId);
}

export async function overwriteProject(
	projectId: string,
	output: ProjectOutput,
	settings: ProjectSettings
) {
	const loaded = await readProjectFiles(projectId);
	const projectDir = projectDirectory(projectId);
	const audioName = loaded.output.source.audio;

	const normalizedOutput: ProjectOutput = {
		...output,
		source: {
			audio: audioName
		}
	};

	await writeProjectFiles(projectDir, normalizedOutput, settings);

	return {
		projectId,
		audioName,
		displayName: projectId
	};
}

export async function cloneProject(
	projectId: string,
	output: ProjectOutput,
	settings: ProjectSettings
) {
	const loaded = await readProjectFiles(projectId);
	const sourceDir = projectDirectory(projectId);
	const audioName = loaded.output.source.audio;
	const now = new Date();
	const newProjectId = `${path.parse(audioName).name}_${makeProjectId(now)}`;
	const newProjectDir = projectDirectory(newProjectId);

	await mkdir(newProjectDir, { recursive: false });

	const normalizedOutput: ProjectOutput = {
		...output,
		source: {
			audio: audioName
		}
	};

	await writeProjectFiles(
		newProjectDir,
		normalizedOutput,
		settings
	);

	await copyFile(
		path.join(sourceDir, audioName),
		path.join(newProjectDir, audioName)
	);

	return {
		projectId: newProjectId,
		projectDir: newProjectDir,
		audioName,
		displayName: makeProjectDisplayName(audioName, now)
	};
}
