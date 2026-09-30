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

/**
 * n8n's standard data volume (PVC in k8s, named volume in Docker).
 * Both the binary and model cache here so they survive restarts.
 */
const CACHE_DIR = path.join(os.homedir(), '.n8n', 'nemo-speech');

// Default diarization model (NVIDIA Nemotron-3-Diarization, Q8_0 GGUF).
const DIAR_MODEL_FILENAME = 'Nemotron-3-Diarization.q8_0.gguf';
const DEFAULT_DIAR_MODEL_URL =
	'https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/main/' + DIAR_MODEL_FILENAME;

// Companion models converted from official sources and hosted for this node
// (see suneetk/nemo-speech-companions on Hugging Face):
//   - VAD:  Silero VAD 6.2.3 (official snakers4/silero-vad pip package)
//   - PnC:  NVIDIA punctuation_en_bert.nemo (NGC), converted to GGUF
//   - NMT:  NVIDIA Riva-Translate-4B-Instruct-v2, converted to Q8_0 GGUF
//   - ITN:  NeMo-Speech.cpp's official itn_configs release asset (Sparrowhawk grammars)
const COMPANIONS_BASE_URL = 'https://huggingface.co/suneetk/nemo-speech-companions/resolve/main/';
const VAD_MODEL_FILENAME = 'silero-v6.2.3.gguf';
const DEFAULT_VAD_MODEL_URL = COMPANIONS_BASE_URL + VAD_MODEL_FILENAME;
const PNC_MODEL_FILENAME = 'pnc-bert-base-en.q8_0.gguf';
const DEFAULT_PNC_MODEL_URL = COMPANIONS_BASE_URL + PNC_MODEL_FILENAME;
const NMT_MODEL_FILENAME = 'riva-translate-4b-instruct-v2.q8_0.gguf';
const DEFAULT_NMT_MODEL_URL = COMPANIONS_BASE_URL + NMT_MODEL_FILENAME;
const ITN_CONFIGS_FILENAME = 'itn_configs.tar.bz2';
const DEFAULT_ITN_CONFIGS_URL = COMPANIONS_BASE_URL + ITN_CONFIGS_FILENAME;

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

/** Resolve a URL through redirects (no body) to its final location. */
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

// ---------------------------------------------------------------------------
// Binary management
// ---------------------------------------------------------------------------

/** Extract a .tar.gz into `destDir`. */
function extractTarGz(archive: string, destDir: string): void {
	execFileSync('tar', ['-xzf', archive, '-C', destDir], { stdio: 'pipe' });
}

/**
 * Ensure the nemo-speech + ffmpeg binaries are available. Checks the local
 * cache first; if missing, downloads the platform tarball from GitHub
 * Releases and extracts it.
 */
async function ensureBinaries(): Promise<{ nemoSpeech: string; ffmpeg: string }> {
	const platform = `${process.platform}-${process.arch}`;
	const binDir = path.join(CACHE_DIR, 'bin', platform);
	const nemoName = process.platform === 'win32' ? 'nemo-speech.exe' : 'nemo-speech';
	const ffmpegName = process.platform === 'win32' ? 'ffmpeg.exe' : 'ffmpeg';
	const nemoPath = path.join(binDir, nemoName);
	const ffmpegPath = path.join(binDir, ffmpegName);
	const versionMarker = path.join(binDir, '.version');
	const pkgVersion = require(path.join(__dirname, '..', '..', '..', 'package.json')).version as string;

	// Env var overrides (for local dev / custom binary locations).
	const nemoOverride = process.env.NEMO_SPEECH_PATH;
	const ffmpegOverride = process.env.FFMPEG_PATH;
	if (nemoOverride && fs.existsSync(nemoOverride) && ffmpegOverride && fs.existsSync(ffmpegOverride)) {
		return { nemoSpeech: nemoOverride, ffmpeg: ffmpegOverride };
	}
	// Fast path: both binaries present and version matches.
	if (
		fs.existsSync(nemoPath) &&
		fs.existsSync(ffmpegPath) &&
		fs.existsSync(versionMarker)
	) {
		const cachedVersion = fs.readFileSync(versionMarker, 'utf-8').trim();
		if (cachedVersion === pkgVersion) {
			return { nemoSpeech: nemoPath, ffmpeg: ffmpegPath };
		}
		// Stale cache — wipe and re-download.
		fs.rmSync(binDir, { recursive: true, force: true });
	}

	const url = `https://github.com/suneetk92/n8n-nodes-nemo-speech/releases/download/v${pkgVersion}/nemo-speech-${platform}.tar.gz`;
	fs.mkdirSync(binDir, { recursive: true });
	const tmp = path.join(CACHE_DIR, `bin-${platform}.tar.gz`);

	await downloadFile(url, tmp);
	// The tarball's top-level entry is the platform folder (e.g. linux-x64/).
	extractTarGz(tmp, path.dirname(binDir));
	fs.unlinkSync(tmp);

	try { fs.chmodSync(nemoPath, 0o755); } catch { /* non-fatal */ }
	try { fs.chmodSync(ffmpegPath, 0o755); } catch { /* non-fatal */ }
	fs.writeFileSync(versionMarker, pkgVersion);

	return { nemoSpeech: nemoPath, ffmpeg: ffmpegPath };
}

// ---------------------------------------------------------------------------
// Audio conversion
// ---------------------------------------------------------------------------

/**
 * Convert an audio file to 16 kHz mono PCM16 WAV using ffmpeg.
 */
async function convertToWav(ffmpeg: string, inputPath: string, outputPath: string): Promise<void> {
	const args = [
		'-y',
		'-i', inputPath,
		'-ac', '1',
		'-ar', '16000',
		'-sample_fmt', 's16',
		outputPath,
	];
	try {
		await execFileAsync(ffmpeg, args, { maxBuffer: 256 * 1024 * 1024, timeout: 10 * 60 * 1000 });
	} catch (e) {
		const err = e as ExecException;
		throw new Error(`ffmpeg conversion failed (exit ${err.code ?? 'unknown'}): ${(err.stderr || err.message).toString().slice(-1000)}`);
	}
}

/**
 * Ensure a companion model file is available locally, downloading it on
 * first use. Returns the local path. Shared by diarization, VAD, PnC, and NMT.
 */
async function ensureCompanionModel(modelDir: string, filename: string, url: string): Promise<string> {
	const dest = path.join(modelDir, filename);
	if (fs.existsSync(dest)) return dest;
	fs.mkdirSync(modelDir, { recursive: true });
	await downloadFile(url, dest + '.downloading');
	fs.renameSync(dest + '.downloading', dest);
	return dest;
}

/**
 * Ensure the official ITN grammar bundle (Sparrowhawk .far files, one dir per
 * language) is extracted locally. Returns the parent directory path expected
 * by --itn-model-dir.
 */
async function ensureItnConfigs(modelDir: string): Promise<string> {
	const destDir = path.join(modelDir, 'itn_configs');
	if (fs.existsSync(destDir)) return destDir;
	fs.mkdirSync(modelDir, { recursive: true });
	const tmp = path.join(modelDir, 'itn_configs.tar.bz2.downloading');
	await downloadFile(DEFAULT_ITN_CONFIGS_URL, tmp);
	execFileSync('tar', ['-xjf', tmp, '-C', modelDir], { stdio: 'pipe' });
	fs.unlinkSync(tmp);
	return destDir;
}

/**
 * Resolve the --model argument to a local .gguf path, downloading it via
 * Node's own HTTPS client (through ensureCompanionModel) instead of the
 * nemo-speech binary's own internal downloader, which shells out to `curl`
 * — unavailable on minimal images (e.g. this project's target Alpine/musl
 * n8n container). Local paths pass through unchanged. Indexed names/aliases
 * (e.g. the default "nvidia/parakeet-tdt-0.6b-v3", or short aliases like
 * "parakeet-ctc") are looked up in the bundled model-index.json to find the
 * HF repo/revision/filename, then downloaded to modelDir just like the
 * companion models.
 */
async function resolveAsrModel(
	modelParam: string,
	modelDir: string,
	modelIndexPath: string,
): Promise<string> {
	if (fs.existsSync(modelParam)) return modelParam;
	if (!fs.existsSync(modelIndexPath)) return modelParam; // no index bundled; let the CLI try (will fail without curl)

	const index = JSON.parse(fs.readFileSync(modelIndexPath, 'utf-8')) as {
		models: Array<{
			repo: string;
			aliases?: string[];
			revision: string;
			artifacts: Array<{ role: string; filename: string }>;
		}>;
	};
	const needle = modelParam.toLowerCase();
	const shortRepo = needle.includes('/') ? needle.slice(needle.lastIndexOf('/') + 1) : needle;
	const entry = index.models.find((m) => {
		if (m.repo.toLowerCase() === needle) return true;
		if (m.repo.toLowerCase().endsWith('/' + shortRepo)) return true;
		return (m.aliases || []).some((a) => a.toLowerCase() === needle);
	});
	if (!entry) return modelParam; // unknown name; let the CLI try (will fail without curl)

	const artifact = entry.artifacts.find((a) => a.role === 'asr') || entry.artifacts[0];
	if (!artifact) return modelParam;

	const url = `https://huggingface.co/${entry.repo}/resolve/${entry.revision}/${artifact.filename}`;
	return ensureCompanionModel(modelDir, artifact.filename, url);
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

export class NemoSpeech implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'NeMo Speech (Speech to Text)',
		name: 'nemoSpeech',
		icon: 'file:../../icons/nemo-speech.svg',
		group: ['transform'],
		version: 1,
		description:
			'Transcribe audio to text using NeMo-Speech.cpp with the parakeet-tdt-0.6b-v3 model (Q8_0 GGUF). Runs fully offline on CPU.',
		defaults: {
			name: 'NeMo Speech STT',
		},
		usableAsTool: true,
		inputs: [NodeConnectionTypes.Main],
		outputs: [NodeConnectionTypes.Main],
		properties: [
			// --- Input source ---
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
				description: 'URL of the audio file (wav, mp3, flac, ogg, m4a).',
			},
			{
				displayName: 'Binary Property',
				name: 'binaryProperty',
				type: 'string',
				displayOptions: { show: { inputType: ['binary'] } },
				default: 'data',
				description: 'Name of the binary property on the input item that holds the audio.',
			},
			// --- Top-level toggles ---
			{
				displayName: 'Output Format',
				name: 'format',
				type: 'options',
				displayOptions: { hide: { diarize: [true] } },
				options: [
					{ name: 'Plain text', value: 'text' },
					{ name: 'JSON (with timestamps)', value: 'json' },
					{ name: 'SRT subtitles', value: 'srt' },
					{ name: 'WebVTT subtitles', value: 'vtt' },
				],
				default: 'text',
				description: 'Output format. JSON/SRT/VTT request word timestamps automatically. Forced to JSON when Diarize is on (text/srt/vtt never surface per-word speaker tags), so this field is hidden in that case.',
			},
			{
				displayName: 'VAD Masking',
				name: 'vadMasking',
				type: 'boolean',
				default: false,
				description: 'Enable VAD feature masking (auto-downloads the official Silero 6.2.3 model if no path is set in Options).',
			},
			{
				displayName: 'Diarize',
				name: 'diarize',
				type: 'boolean',
				default: false,
				description: 'Enable speaker diarization (auto-downloads the default Nemotron-3-Diarization model).',
			},
			{
				displayName: 'Endpointing',
				name: 'endpointing',
				type: 'boolean',
				default: false,
				description: 'Enable mid-stream end-of-utterance detection (multiple finals).',
			},
			{
				displayName: 'Enable ITN',
				name: 'itn',
				type: 'boolean',
				default: false,
				description: 'Enable inverse text normalization ("twenty twenty four" → "2024"). Opt-in — auto-downloads the official multi-language grammars (en, es, de, ...) on first use. Self-punctuating ASR models (e.g. parakeet-tdt) already normalize most cardinals themselves, so this mainly helps ordinals/symbols or plain-text models like parakeet-ctc.',
			},
			{
				displayName: 'Enable PnC',
				name: 'pnc',
				type: 'boolean',
				default: false,
				description: 'Enable automatic punctuation and capitalization. Opt-in — auto-downloads the official PnC BERT model on first use. No-ops with a warning on self-punctuating models (e.g. parakeet-tdt); restores punctuation/casing for plain-text models (e.g. parakeet-ctc).',
			},
			{
				displayName: 'Translate To',
				name: 'translateTo',
				type: 'string',
				default: '',
				placeholder: 'e.g. es, de, fr, zh',
				description: 'Target language code. Enables translation via the official Riva-Translate-4B-Instruct-v2 model (auto-downloaded, 4.2 GB). Leave empty to disable.',
			},
			{
				displayName: 'Write Text File',
				name: 'outputTxt',
				type: 'boolean',
				default: false,
				description: 'Also write the transcript to a .txt file (returned as a binary property).',
			},
			// --- Options ---
			{
				displayName: 'Options',
				name: 'options',
				type: 'collection',
				placeholder: 'Add option',
				default: {},
				options: [
					// Model
					{
						displayName: 'Model Path',
						name: 'modelPath',
						type: 'string',
						default: '',
						placeholder: 'auto (default parakeet-tdt model)',
						description:
							'Path to a .gguf model file, or an indexed model name (e.g. "parakeet-tdt"). Leave empty to use the default indexed model (parakeet-tdt-0.6b-v3), auto-downloaded on first run.',
					},
					// Core
					{
						displayName: 'Threads',
						name: 'threads',
						type: 'number',
						default: 4,
						description: 'Number of CPU threads to use during computation.',
					},
					{
						displayName: 'Device',
						name: 'device',
						type: 'options',
						options: [
							{ name: 'CPU', value: 'cpu' },
							{ name: 'CUDA (GPU)', value: 'cuda:0' },
							{ name: 'Metal', value: 'metal' },
							{ name: 'Vulkan', value: 'vulkan:0' },
						],
						default: 'cpu',
						description: 'Compute device. CPU for the static binary; GPU requires a CUDA/Metal build.',
					},
					{
						displayName: 'Stream Mode',
						name: 'stream',
						type: 'boolean',
						default: false,
						description: 'Feed audio through the streaming recognizer in 160 ms chunks. Offline-only models (e.g. Parakeet TDT) reject this.',
					},
					{
						displayName: 'Word Timestamps',
						name: 'wordTimes',
						type: 'boolean',
						default: false,
						description: 'Include per-word timestamps in the output (JSON format).',
					},
					{
						displayName: 'Verbose Logs',
						name: 'verbose',
						type: 'boolean',
						default: false,
						description: 'Enable verbose diagnostics from nemo-speech.',
					},
					{
						displayName: 'Quiet (no logs)',
						name: 'noPrints',
						type: 'boolean',
						default: false,
						description: 'Suppress all logging from nemo-speech; only the transcript is returned.',
					},
					// Decoding / CTC
					{
						displayName: 'LM Path (KenLM)',
						name: 'lmPath',
						type: 'string',
						default: '',
						description: 'Path to a KenLM .bin or .arpa language model. Enables Flashlight beam search (requires a flashlight build).',
					},
					{
						displayName: 'Lexicon Path',
						name: 'lexicon',
						type: 'string',
						default: '',
						description: 'Path to a Flashlight lexicon TSV file. Required with LM path.',
					},
					{
						displayName: 'Tokenizer Path',
						name: 'tokenizer',
						type: 'string',
						default: '',
						description: 'Path to a SentencePiece tokenizer for OOV boosting (CTC).',
					},
					{
						displayName: 'Beam Size',
						name: 'beamSize',
						type: 'number',
						default: 32,
						description: 'Flashlight beam width. Default 32.',
					},
					{
						displayName: 'Beam Threshold',
						name: 'beamThreshold',
						type: 'number',
						default: 20,
						description: 'Beam pruning threshold. Default 20.0.',
					},
					{
						displayName: 'LM Weight',
						name: 'lmWeight',
						type: 'number',
						default: 0.8,
						description: 'Language model rescoring weight. Default 0.8.',
					},
					{
						displayName: 'Word Score',
						name: 'wordScore',
						type: 'number',
						default: 1.0,
						description: 'Word insertion bonus. Default 1.0.',
					},
					{
						displayName: 'Max Boost (CTC)',
						name: 'maxBoost',
						type: 'number',
						default: 10,
						description: 'CTC: max per-word boost magnitude. Default 10.0.',
					},
					// Streaming / CTC padding
					{
						displayName: 'Chunk Size (sec)',
						name: 'chunkSec',
						type: 'number',
						default: 0.16,
						description: 'CTC buffered window in seconds. Default 0.16.',
					},
					{
						displayName: 'Left Padding (sec)',
						name: 'leftPadSec',
						type: 'number',
						default: 1.92,
						description: 'CTC left context in seconds. Default 1.92.',
					},
					{
						displayName: 'Right Padding (sec)',
						name: 'rightPadSec',
						type: 'number',
						default: 1.92,
						description: 'CTC right context in seconds. Default 1.92.',
					},
					// VAD
					{
						displayName: 'VAD Model Path',
						name: 'vadModel',
						type: 'string',
						default: '',
						description: 'Path to a Silero VAD GGUF model. Leave empty to auto-download the official Silero 6.2.3 model when VAD masking or VAD-based endpointing is enabled.',
					},
					{
						displayName: 'VAD-Based Endpointing',
						name: 'vadBasedEou',
						type: 'boolean',
						default: false,
						description: 'Use the VAD timeline (instead of token-silence) for endpointing. Requires --endpointing; auto-downloads the VAD model if no path is set.',
					},
					{
						displayName: 'VAD Onset Threshold',
						name: 'vadOnset',
						type: 'number',
						default: 0.5,
						description: 'Probability threshold to enter speech. Default 0.5.',
					},
					{
						displayName: 'VAD Offset Threshold',
						name: 'vadOffset',
						type: 'number',
						default: 0.3,
						description: 'Probability threshold to leave speech. Default 0.3.',
					},
					{
						displayName: 'VAD Pad (ms)',
						name: 'vadPadMs',
						type: 'number',
						default: 200,
						description: 'Extend both segment edges by this many ms. Default 200.',
					},
					// Diarization
					{
						displayName: 'Diar Model Path',
						name: 'diarModel',
						type: 'string',
						default: '',
						description: 'Path to a diarizer GGUF. Leave empty to use the default Nemotron-3-Diarization model (auto-downloaded).',
					},
					// Postprocessing
					{
						displayName: 'Profanity List Path',
						name: 'profanityList',
						type: 'string',
						default: '',
						description: 'Path to a profanity filter list file (one word per line).',
					},
					{
						displayName: 'ITN Model Dir',
						name: 'itnModelDir',
						type: 'string',
						default: '',
						description: 'Path to a Sparrowhawk grammar directory. Only used when Enable ITN is on — leave empty to auto-download the official grammars.',
					},
					{
						displayName: 'PnC Model Path',
						name: 'pncModel',
						type: 'string',
						default: '',
						description: 'Path to a PnC BERT GGUF. Only used when Enable PnC is on — leave empty to auto-download the official model.',
					},
					// Translation (NMT)
					{
						displayName: 'NMT Model Path',
						name: 'nmtModel',
						type: 'string',
						default: '',
						description: 'Path to a Riva-Translate GGUF. Leave empty to auto-download the official Riva-Translate-4B-Instruct-v2 model.',
					},
					// Boosting
					{
						displayName: 'Speech Context (boosted words)',
						name: 'speechContext',
						type: 'string',
						default: '',
						description: 'Comma-separated list of words/phrases to boost in the transcript (e.g. "NVIDIA,Parakeet").',
					},
					// Output file
					{
						displayName: 'Output File Base Path',
						name: 'outputFile',
						type: 'string',
						default: '',
						description: 'Base path (without extension) for the .txt output. Requires "Write Text File".',
					},
				],
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const node = this.getNode();

		// Resolve binaries.
		let nemoSpeech: string;
		let ffmpeg: string;
		try {
			({ nemoSpeech, ffmpeg } = await ensureBinaries());
		} catch (e) {
			throw new NodeOperationError(
				node,
				`Failed to obtain nemo-speech/ffmpeg binaries: ${(e as Error).message}. Set NEMO_SPEECH_PATH and FFMPEG_PATH to local binaries.`,
			);
		}

		const options = (this.getNodeParameter('options', 0, {}) as IDataObject) || {};
		const modelParam = (options.modelPath as string) || '';
		const inputType = (this.getNodeParameter('inputType', 0, 'binary') as string) || 'binary';
		const binaryProperty = (this.getNodeParameter('binaryProperty', 0, 'data') as string) || 'data';
		const format = (this.getNodeParameter('format', 0, 'text') as string) || 'text';
		const vadMasking = this.getNodeParameter('vadMasking', 0, false) as boolean;
		const diarize = this.getNodeParameter('diarize', 0, false) as boolean;
		// Diarization only surfaces per-word speaker tags in JSON output —
		// text/srt/vtt never read the speaker field — so force JSON whenever
		// Diarize is on, regardless of the (hidden, in that case) Format field.
		const effectiveFormat = diarize ? 'json' : format;
		const endpointing = this.getNodeParameter('endpointing', 0, false) as boolean;
		const itnEnabled = this.getNodeParameter('itn', 0, false) as boolean;
		const pncEnabled = this.getNodeParameter('pnc', 0, false) as boolean;
		const translateTo = (this.getNodeParameter('translateTo', 0, '') as string) || '';
		const outputTxt = this.getNodeParameter('outputTxt', 0, false) as boolean;

		// Model dir for auto-download of the default indexed model.
		const modelDir = path.join(CACHE_DIR, 'models');
		fs.mkdirSync(modelDir, { recursive: true });

		// Resolve the ASR model to a local path ourselves (see resolveAsrModel):
		// the binary's own auto-downloader needs `curl`, which isn't guaranteed
		// to exist on minimal target images.
		const modelIndexPath = path.join(path.dirname(nemoSpeech), 'model-index.json');
		let resolvedModelPath: string;
		try {
			resolvedModelPath = await resolveAsrModel(
				modelParam || 'nvidia/parakeet-tdt-0.6b-v3',
				modelDir,
				modelIndexPath,
			);
		} catch (e) {
			throw new NodeOperationError(node, `Failed to download ASR model: ${(e as Error).message}`);
		}

		const results: INodeExecutionData[] = [];

		for (let i = 0; i < items.length; i++) {
			const item = items[i];

			// Resolve the audio to a local temp file.
			let tmpAudio: string;
			if (inputType === 'url') {
				const url = (this.getNodeParameter('audioUrl', i, '') as string) || '';
				const buf = Buffer.from(await this.helpers.httpRequest({ url, encoding: 'arraybuffer' }));
				tmpAudio = path.join(os.tmpdir(), `nemo-in-${Date.now()}-${i}.bin`);
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
				tmpAudio = path.join(os.tmpdir(), `nemo-in-${Date.now()}-${i}.bin`);
				fs.writeFileSync(tmpAudio, buf);
			}

			// Convert to 16 kHz mono PCM16 WAV (nemo-speech requires WAV input).
			const tmpWav = tmpAudio + '.wav';
			try {
				await convertToWav(ffmpeg, tmpAudio, tmpWav);
			} catch (e) {
				throw new NodeOperationError(node, (e as Error).message);
			} finally {
				try { fs.unlinkSync(tmpAudio); } catch { /* ignore */ }
			}

			// Build the nemo-speech argument list.
			const args: string[] = ['transcribe', tmpWav];

			// Model: resolved to a local path already (see resolveAsrModel above).
			args.push('--model', resolvedModelPath);

			// Device
			const device = (options.device as string) || 'cpu';
			args.push('--device', device);

			// Output format
			if (effectiveFormat !== 'text') args.push('--format', effectiveFormat);

			// Stream mode
			if (options.stream) args.push('--stream');

			// Word timestamps
			if (options.wordTimes) args.push('--word-times');

			// Verbose / Quiet
			if (options.verbose) args.push('--verbose');
			if (options.noPrints) args.push('--quiet');

			// Decoding / CTC
			if (options.lmPath) args.push('--lm-path', String(options.lmPath));
			if (options.lexicon) args.push('--lexicon', String(options.lexicon));
			if (options.tokenizer) args.push('--tokenizer', String(options.tokenizer));
			if (options.beamSize != null) args.push('--beam-size', String(options.beamSize));
			if (options.beamThreshold != null) args.push('--beam-threshold', String(options.beamThreshold));
			if (options.lmWeight != null) args.push('--lm-weight', String(options.lmWeight));
			if (options.wordScore != null) args.push('--word-score', String(options.wordScore));
			if (options.maxBoost != null) args.push('--max-boost', String(options.maxBoost));

			// Streaming / CTC padding
			if (options.chunkSec != null) args.push('--chunk-sec', String(options.chunkSec));
			if (options.leftPadSec != null) args.push('--left-pad-sec', String(options.leftPadSec));
			if (options.rightPadSec != null) args.push('--right-pad-sec', String(options.rightPadSec));

			// VAD (auto-downloads the official Silero 6.2.3 GGUF when a mask/EOU
			// mode is requested and no explicit path is given).
			const wantsVad = vadMasking || Boolean(options.vadBasedEou);
			if (options.vadModel) {
				args.push('--vad-model', String(options.vadModel));
			} else if (wantsVad) {
				args.push('--vad-model', await ensureCompanionModel(modelDir, VAD_MODEL_FILENAME, DEFAULT_VAD_MODEL_URL));
			}
			if (vadMasking) args.push('--vad-masking');
			if (options.vadOnset != null) args.push('--vad-onset', String(options.vadOnset));
			if (options.vadOffset != null) args.push('--vad-offset', String(options.vadOffset));
			if (options.vadPadMs != null) args.push('--vad-pad-ms', String(options.vadPadMs));

			// Diarization (auto-downloads the default Nemotron-3-Diarization GGUF).
			let diarPath: string | undefined;
			if (diarize) {
				diarPath = options.diarModel
					? String(options.diarModel)
					: await ensureCompanionModel(modelDir, DIAR_MODEL_FILENAME, DEFAULT_DIAR_MODEL_URL);
				args.push('--diar-model', diarPath);
			}

			// Endpointing
			if (endpointing) args.push('--endpointing');
			if (options.vadBasedEou) args.push('--vad-based-eou');

			// Postprocessing. ITN and PnC are opt-in: auto-downloaded from official
			// converted sources on first use. Off by default means simply omitting
			// both flags — NOT passing --verbatim/--no-punctuation, since those
			// force lowercase/unpunctuated rendering and strip the casing and
			// punctuation a self-punctuating model (e.g. parakeet-tdt) already
			// bakes into its own output. Enabling adds the separate BERT-based
			// PnC pass / Sparrowhawk ITN grammars on top of whatever the ASR
			// head already produced.
			if (options.profanityList) args.push('--profanity-list', String(options.profanityList));
			if (itnEnabled) {
				const itnDir = options.itnModelDir
					? String(options.itnModelDir)
					: await ensureItnConfigs(modelDir);
				args.push('--itn-model-dir', itnDir);
			}
			if (pncEnabled) {
				const pncPath = options.pncModel
					? String(options.pncModel)
					: await ensureCompanionModel(modelDir, PNC_MODEL_FILENAME, DEFAULT_PNC_MODEL_URL);
				args.push('--pnc-model', pncPath);
			}

			// NMT (opt-in: translation needs an explicit target language; the
			// official Riva-Translate-4B GGUF auto-downloads on first use).
			if (translateTo) {
				const nmtPath = options.nmtModel
					? String(options.nmtModel)
					: await ensureCompanionModel(modelDir, NMT_MODEL_FILENAME, DEFAULT_NMT_MODEL_URL);
				args.push('--nmt-model', nmtPath, '--translate-to', translateTo);
			}

			// Boosting
			if (options.speechContext) {
				const words = String(options.speechContext).split(',').map(w => w.trim()).filter(Boolean);
				for (const word of words) {
					args.push('--speech-context', word);
				}
			}

			// Env: model cache dir + thread count. Every model arg (ASR, diar,
			// VAD, PnC, ITN, NMT) is already a resolved local path by this point,
			// so the binary never needs to consult NEMO_SPEECH_MODEL_INDEX or
			// auto-download anything itself.
			const env: NodeJS.ProcessEnv = { ...process.env };
			env.NEMO_SPEECH_MODEL_DIR = modelDir;
			if (options.threads != null) {
				env.OMP_NUM_THREADS = String(options.threads);
				env.OPENBLAS_NUM_THREADS = String(options.threads);
			}

			let stdout = '';
			let rttmBuf: Buffer | undefined;
			try {
				const res = await execFileAsync(nemoSpeech, args, {
					maxBuffer: 256 * 1024 * 1024,
					timeout: 30 * 60 * 1000,
					env,
				});
				stdout = res.stdout || '';

				// RTTM export: standalone `diarize` subcommand, same diarizer
				// model and WAV, run before the WAV is cleaned up. Runs
				// automatically whenever Diarize is on — Output Format text/
				// srt/vtt never surface per-word speaker tags (only json does),
				// so RTTM is the only reliable way to get diarization output
				// regardless of the chosen transcript format.
				if (diarize && diarPath) {
					const tmpRttm = tmpWav + '.rttm';
					try {
						await execFileAsync(nemoSpeech, ['diarize', tmpWav, '--model', diarPath, '--format', 'rttm', '--output', tmpRttm], {
							maxBuffer: 64 * 1024 * 1024,
							timeout: 10 * 60 * 1000,
							env,
						});
						rttmBuf = fs.readFileSync(tmpRttm);
					} finally {
						try { fs.unlinkSync(tmpRttm); } catch { /* ignore */ }
					}
				}
			} catch (e) {
				const err = e as ExecException & { code?: number | string };
				throw new NodeOperationError(
					node,
					`nemo-speech failed (exit ${err.code ?? 'unknown'}): ${(err.stderr || err.message).toString().slice(-2000)}`,
				);
			} finally {
				try { fs.unlinkSync(tmpWav); } catch { /* ignore */ }
			}

			const transcript = stdout.trim();

			const outItem: INodeExecutionData = {
				json: {
					text: transcript,
					model: modelParam ? path.basename(modelParam) : 'parakeet-tdt-0.6b-v3 (default)',
					input: inputType === 'url' ? (this.getNodeParameter('audioUrl', i, '') as string) : binaryProperty,
				},
			};

			// If outputTxt was requested, surface the transcript as a binary property.
			if (outputTxt) {
				const base = options.outputFile
					? String(options.outputFile)
					: path.join(os.tmpdir(), `nemo-out-${Date.now()}-${i}`);
				const txtPath = base + '.txt';
				fs.writeFileSync(txtPath, transcript);
				const txtBuf = fs.readFileSync(txtPath);
				const binData = await this.helpers.prepareBinaryData(txtBuf, path.basename(txtPath), 'text/plain');
				outItem.binary = { transcript: binData };
			}

			// RTTM diarization export, if requested and generated above.
			if (rttmBuf) {
				const binData = await this.helpers.prepareBinaryData(rttmBuf, `diarization-${i}.rttm`, 'text/plain');
				outItem.binary = { ...outItem.binary, rttm: binData };
			}

			results.push(outItem);
		}

		return [results];
	}
}
