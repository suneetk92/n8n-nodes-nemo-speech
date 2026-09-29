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
import { execFile, type ExecException } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);

const MODEL_FILENAME = 'ggml-parakeet-tdt-0.6b-v3-q8_0.bin';

function resolveBinaryPath(): string {
	const override = process.env.PARAKEET_CLI_PATH;
	if (override && fs.existsSync(override)) return override;
	const platformDir = path.join(__dirname, '..', '..', '..', 'bin', `${process.platform}-${process.arch}`);
	const exeName = process.platform === 'win32' ? 'parakeet-cli.exe' : 'parakeet-cli';
	return path.join(platformDir, exeName);
}

function resolveModelPath(): string {
	const override = process.env.PARAKEET_MODEL_PATH;
	if (override) return override;
	return path.join(os.homedir(), '.cache', 'n8n-nodes-parakeet', 'models', MODEL_FILENAME);
}

export class Parakeet implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Parakeet (Speech to Text)',
		name: 'parakeet',
		icon: 'fa:feather-alt',
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

		const binaryPath = resolveBinaryPath();
		if (!fs.existsSync(binaryPath)) {
			throw new NodeOperationError(
				node,
				`parakeet-cli binary not found at "${binaryPath}". Run "npm install" (postinstall) or set PARAKEET_CLI_PATH.`,
			);
		}

		const modelParam = this.getNodeParameter('modelPath', 0) as string;
		const modelPath = modelParam || resolveModelPath();
		if (!fs.existsSync(modelPath)) {
			throw new NodeOperationError(
				node,
				`Model not found at "${modelPath}". It is normally downloaded on install. Set PARAKEET_MODEL_PATH or re-run the install.`,
			);
		}

		const options = (this.getNodeParameter('options', 0) as IDataObject) || {};
		const inputType = (this.getNodeParameter('inputType', 0) as string) || 'binary';
		const binaryProperty = (this.getNodeParameter('binaryProperty', 0) as string) || 'data';

		const results: INodeExecutionData[] = [];

		for (let i = 0; i < items.length; i++) {
			const item = items[i];

			// Resolve the audio to a local temp file.
			let tmpAudio: string;
			if (inputType === 'url') {
				const url = this.getNodeParameter('audioUrl', i) as string;
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
					input: inputType === 'url' ? (this.getNodeParameter('audioUrl', i) as string) : binaryProperty,
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
