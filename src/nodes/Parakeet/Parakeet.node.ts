import {
	NodeConnectionTypes,
	NodeOperationError,
} from 'n8n-workflow';
import type {
	IExecuteFunctions,
	IDataObject,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
} from 'n8n-workflow';

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { execFile, execFileSync, type ExecException } from 'child_process';
import { promisify } from 'util';
import * as http from 'http';
import * as https from 'https';

const execFileAsync = promisify(execFile);

const MODEL_FILENAME = 'ggml-parakeet-tdt-0.6b-v3-q8_0.bin';
const DEFAULT_MODEL_URL =
	'https://huggingface.co/ggml-org/parakeet-GGUF/resolve/main/' + MODEL_FILENAME;

/**
 * n8n's standard data volume (PVC in k8s, named volume in Docker).
 * Both the model and the binary cache here so they survive restarts.
 */
const CACHE_DIR = path.join(os.homedir(), '.n8n', 'parakeet');

function defaultModelPath(): string {
	return path.join(CACHE_DIR, 'models', MODEL_FILENAME);
}

/** Follow redirects and stream `url` to `dest`. */
async function downloadFile(url: string, dest: string): Promise<void> {
	const finalUrl = await resolveFinalUrl(url);
	await new Promise<void>((resolve, reject) => {
		const file = fs.createWriteStream(dest);
		const reqMod = finalUrl.startsWith('https:') ? https : http;
		const req = reqMod.get(finalUrl, (res) => {
			if ((res.statusCode ?? 0) !== 200) {
				res.resume();
				file.close(() => fs.unlink(dest, () => {}));
				reject(new Error(`HTTP ${res.statusCode} downloading ${url}`));
				return;
			}
			res.pipe(file);
			file.on('finish', () => file.close(() => resolve()));
		});
		req.on('error', (e) => {
			file.close(() => fs.unlink(dest, () => {}));
			reject(e);
		});
	});
}

/** Extract a .tar.gz into `destDir`. */
function extractTarGz(archive: string, destDir: string): void {
	execFileSync('tar', ['-xzf', archive, '-C', destDir], { stdio: 'pipe' });
}
/**
 * Ensure the parakeet-cli binary is available. Checks the local cache first;
 * if missing, downloads the platform tarball from GitHub Releases and extracts it.
 */
async function ensureBinary(): Promise<string> {
	const override = process.env.PARAKEET_CLI_PATH;
	if (override && fs.existsSync(override)) return override;

	const platform = `${process.platform}-${process.arch}`;
	const binDir = path.join(CACHE_DIR, 'bin', platform);
	const exeName = process.platform === 'win32' ? 'parakeet-cli.exe' : 'parakeet-cli';
	const binPath = path.join(binDir, exeName);
	if (fs.existsSync(binPath)) {
		ensureSymlinks(binDir);
		return binPath;
	}

	const version = require(path.join(__dirname, '..', '..', '..', 'package.json')).version as string;
	const url = `https://github.com/suneetk92/n8n-nodes-parakeet/releases/download/v${version}/parakeet-cli-${platform}.tar.gz`;
	fs.mkdirSync(binDir, { recursive: true });
	const tmp = path.join(CACHE_DIR, `bin-${platform}.tar.gz`);

	await downloadFile(url, tmp);
	// The tarball's top-level entry is the platform folder (e.g. linux-x64/), so extract
	// into its parent and the files land directly in binDir.
	extractTarGz(tmp, path.dirname(binDir));
	fs.unlinkSync(tmp);

	try { fs.chmodSync(binPath, 0o755); } catch { /* non-fatal */ }
	ensureSymlinks(binDir);
	return binPath;
}

/**
 * npm pack drops symlinks. Scan the bin dir for versioned shared libraries
 * (lib*.so.<major>[.<minor>...]) and recreate soname → file links so the
 * dynamic loader resolves DT_NEEDED entries (e.g. libparakeet.so.1 → libparakeet.so.1.9.4).
 * No hardcoded filenames — works for any future whisper.cpp release.
 */
function ensureSymlinks(binDir: string): void {
	for (const file of fs.readdirSync(binDir)) {
		const m = file.match(/^lib(.+)\.so\.(\d+)(\..+)?$/);
		if (!m) continue;
		const soname = `lib${m[1]}.so.${m[2]}`;
		if (soname === file) continue;
		const linkPath = path.join(binDir, soname);
		try {
			fs.rmSync(linkPath, { force: true });
			fs.symlinkSync(file, linkPath);
		} catch { /* non-fatal */ }
	}
}
/**
 * Resolve a URL through redirects (no body) to its final location.
 */
function resolveFinalUrl(url: string, attempt = 0): Promise<string> {
	return new Promise((resolve, reject) => {
		const reqMod = url.startsWith('https:') ? https : http;
		const req = reqMod.get(url, (res) => {
			const status = res.statusCode ?? 0;
			if (status >= 300 && status < 400 && res.headers.location) {
				res.resume();
				if (attempt > 5) return reject(new Error('Too many redirects'));
				resolveFinalUrl(res.headers.location, attempt + 1).then(resolve, reject);
				return;
			}
			if (status !== 200) {
				res.resume();
				reject(new Error(`HTTP ${status}`));
				return;
			}
			resolve(url);
		});
		req.on('error', reject);
	});
}

/**
 * Download `url` to `dest` if it doesn't already exist. Follows redirects
 * (HuggingFace resolve -> CDN). Returns the final path.
 */
async function ensureModel(dest: string): Promise<string> {
	if (fs.existsSync(dest)) return dest;
	fs.mkdirSync(path.dirname(dest), { recursive: true });

	const url = process.env.PARAKEET_MODEL_URL || DEFAULT_MODEL_URL;
	const tmp = dest + '.downloading';

	// Resolve redirects first so the body download is a single clean stream.
	const finalUrl = await resolveFinalUrl(url);

	await new Promise<void>((resolve, reject) => {
		const file = fs.createWriteStream(tmp);
		const reqMod = finalUrl.startsWith('https:') ? https : http;
		const req = reqMod.get(finalUrl, (res) => {
			if ((res.statusCode ?? 0) !== 200) {
				res.resume();
				file.close(() => fs.unlink(tmp, () => {}));
				reject(new Error(`HTTP ${res.statusCode} downloading model`));
				return;
			}
			res.pipe(file);
			file.on('finish', () => file.close(() => resolve()));
		});
		req.on('error', (e) => {
			file.close(() => fs.unlink(tmp, () => {}));
			reject(e);
		});
	});

	fs.renameSync(tmp, dest);
	return dest;
}

export class Parakeet implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Parakeet (Speech to Text)',
		name: 'parakeet',
		icon: 'file:../../icons/parakeet.svg',
		group: ['transform'],
		version: 1,
		description:
			'Transcribe audio to text using parakeet.cpp (whisper.cpp) with the parakeet-tdt-0.6b-v3 model. Runs fully offline.',
		defaults: {
			name: 'Parakeet STT',
		},
		usableAsTool: true,
		inputs: [NodeConnectionTypes.Main],
		outputs: [NodeConnectionTypes.Main],
		properties: [
			{
				displayName: 'Input',
				name: 'inputType',
				type: 'options',
				options: [
					{ name: 'Binary (from previous node)', value: 'binary' },
					{ name: 'URL', value: 'url' },
				],
				default: 'binary',
				description: 'Where the audio comes from.',
			},
			{
				displayName: 'Audio URL',
				name: 'audioUrl',
				type: 'string',
				displayOptions: { show: { inputType: ['url'] } },
				default: '',
				required: true,
				description: 'URL of the audio file (wav, mp3, flac, ogg).',
			},
			{
				displayName: 'Binary Property',
				name: 'binaryProperty',
				type: 'string',
				displayOptions: { show: { inputType: ['binary'] } },
				default: 'data',
				description: 'Name of the binary property on the input item that holds the audio.',
			},
			{
				displayName: 'Model Path',
				name: 'modelPath',
				type: 'string',
				default: '',
				placeholder: 'auto (cached q8_0 model)',
				description:
					'Path to a .bin model. Leave empty to use the auto-downloaded ggml-parakeet-tdt-0.6b-v3-q8_0.bin.',
			},
			{
				displayName: 'Options',
				name: 'options',
				type: 'collection',
				placeholder: 'Add option',
				default: {},
				options: [
					{
						displayName: 'Threads',
						name: 'threads',
						type: 'number',
						default: 4,
						description: 'Number of CPU threads to use during computation.',
					},
					{
						displayName: 'Use GPU',
						name: 'useGpu',
						type: 'boolean',
						default: false,
						description: 'Enable GPU (requires a CUDA/Metal build). Default off for CPU-only binaries.',
					},
					{
						displayName: 'GPU Device',
						name: 'device',
						type: 'number',
						default: 0,
						description: 'GPU device index to use.',
					},
					{
						displayName: 'Print Segments',
						name: 'printSegments',
						type: 'boolean',
						default: false,
						description: 'Include per-segment and per-token detail in the output.',
					},
					{
						displayName: 'Write Text File',
						name: 'outputTxt',
						type: 'boolean',
						default: false,
						description: 'Also write the transcript to a .txt file (returned as a binary property).',
					},
					{
						displayName: 'Output File Base Path',
						name: 'outputFile',
						type: 'string',
						default: '',
						description: 'Base path (without extension) for the .txt output. Requires "Write Text File".',
					},
					{
						displayName: 'Quiet (no logs)',
						name: 'noPrints',
						type: 'boolean',
						default: false,
						description: 'Suppress all logging from parakeet-cli; only the transcript is returned.',
					},
				],
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const node = this.getNode();

		let binaryPath: string;
		try {
			binaryPath = await ensureBinary();
		} catch (e) {
			throw new NodeOperationError(
				node,
				`Failed to obtain parakeet-cli binary: ${(e as Error).message}. Set PARAKEET_CLI_PATH to a local binary.`,
			);
		}
		ensureSymlinks(path.dirname(binaryPath));
		const modelParam = (this.getNodeParameter('modelPath', 0, '') as string) || '';
		const modelPath = modelParam || defaultModelPath();
		try {
			await ensureModel(modelPath);
		} catch (e) {
			throw new NodeOperationError(
				node,
				`Failed to download model to "${modelPath}": ${(e as Error).message}. Set PARAKEET_MODEL_PATH to a local .bin file.`,
			);
		}

		const options = (this.getNodeParameter('options', 0, {}) as IDataObject) || {};
		const inputType = (this.getNodeParameter('inputType', 0, 'binary') as string) || 'binary';
		const binaryProperty = (this.getNodeParameter('binaryProperty', 0, 'data') as string) || 'data';

		const results: INodeExecutionData[] = [];

		for (let i = 0; i < items.length; i++) {
			const item = items[i];

			// Resolve the audio to a local temp file.
			let tmpAudio: string;
			if (inputType === 'url') {
				const url = (this.getNodeParameter('audioUrl', i, '') as string) || '';
				const buf = Buffer.from(await this.helpers.httpRequest({ url, encoding: 'arraybuffer' }));
				tmpAudio = path.join(os.tmpdir(), `parakeet-in-${Date.now()}-${i}.wav`);
				fs.writeFileSync(tmpAudio, buf);
			} else {
				const binary = item.binary;
				if (!binary || !binary[binaryProperty]) {
					throw new NodeOperationError(
						node,
						`No binary data found under property "${binaryProperty}" on the input item. Provide audio via a File node or set "Input" to URL.`,
					);
				}
				const binData = binary[binaryProperty];
				const buf = Buffer.from(binData.data, 'base64');
				const ext = binData.fileExtension || 'wav';
				tmpAudio = path.join(os.tmpdir(), `parakeet-in-${Date.now()}-${i}.${ext}`);
				fs.writeFileSync(tmpAudio, buf);
			}

			// Build the parakeet-cli argument list (one-to-one with the CLI flags).
			const args: string[] = ['-m', modelPath];
			if (options.threads != null) args.push('-t', String(options.threads));
			if (!options.useGpu) {
				args.push('-ng');
			}
			if (options.device != null && options.useGpu) args.push('-dev', String(options.device));
			if (options.printSegments) args.push('-ps');
			if (options.outputTxt) args.push('-otxt');
			if (options.outputFile) args.push('-of', String(options.outputFile));
			if (options.noPrints) args.push('-np');
			args.push(tmpAudio);

			let stdout = '';
			try {
				const res = await execFileAsync(binaryPath, args, {
					maxBuffer: 256 * 1024 * 1024,
					timeout: 30 * 60 * 1000,
				});
				stdout = res.stdout || '';
			} catch (e) {
				const err = e as ExecException & { code?: number | string };
				throw new NodeOperationError(
					node,
					`parakeet-cli failed (exit ${err.code ?? 'unknown'}): ${(err.stderr || err.message).toString().slice(-2000)}`,
				);
			} finally {
				try { fs.unlinkSync(tmpAudio); } catch { /* ignore */ }
			}

			const transcript = stdout.trim();

			const outItem: INodeExecutionData = {
				json: {
					text: transcript,
					model: path.basename(modelPath),
					input: inputType === 'url' ? (this.getNodeParameter('audioUrl', i, '') as string) : binaryProperty,
				},
			};

			// If -otxt was used, surface the written .txt as a binary property.
			if (options.outputTxt) {
				const base = options.outputFile ? String(options.outputFile) : tmpAudio.replace(/\.[^.]+$/, '');
				const txtPath = base + '.txt';
				if (fs.existsSync(txtPath)) {
					const txtBuf = fs.readFileSync(txtPath);
					const binData = await this.helpers.prepareBinaryData(txtBuf, path.basename(txtPath), 'text/plain');
					outItem.binary = { transcript: binData };
				}
			}

			results.push(outItem);
		}

		return [results];
	}
}
