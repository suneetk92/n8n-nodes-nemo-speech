'use strict';
// Throwaway smoke test: load the compiled node and run execute() with a mock IExecuteFunctions.
const fs = require('fs');
const path = require('path');

const { NemoSpeech } = require('./dist/nodes/NemoSpeech/NemoSpeech.node.js');

// Use a local WAV file for testing (16kHz mono PCM16).
// Set NEMO_WAV_PATH to override; defaults to /root/projects/its/.build/jfk.wav
const wavPath = process.env.NEMO_WAV_PATH || '/root/projects/its/.build/jfk.wav';

function makeMockExecuteFunctions(params) {
	const node = { id: 'n1', name: 'NeMo Speech STT', typeVersion: 1, type: 'nemoSpeech' };
	return {
		getNode: () => node,
		getInputData: () => [
			{
				json: {},
				binary: {
					data: {
						data: fs.readFileSync(wavPath).toString('base64'),
						mimeType: 'audio/wav',
						fileExtension: 'wav',
						fileName: 'jfk.wav',
					},
				},
			},
		],
		getNodeParameter: (name, _i, fallback) => {
			if (name in params) return params[name];
			return fallback;
		},
		helpers: {
			httpRequest: async () => fs.readFileSync(wavPath),
			prepareBinaryData: async (buf, fileName, mimeType) => ({
				data: buf.toString('base64'),
				fileName,
				mimeType,
			}),
		},
	};
}

async function run(label, params) {
	const node = new NemoSpeech();
	const fn = makeMockExecuteFunctions(params);
	const out = await node.execute.call(fn);
	const item = out[0][0];
	console.log(`\n=== ${label} ===`);
	console.log('json.text:', JSON.stringify(item.json.text));
	console.log('json.model:', item.json.model);
	if (item.binary) console.log('binary keys:', Object.keys(item.binary));
	return item;
}

(async () => {
	// 1. basic: binary input, CPU, 2 threads
	await run('basic (binary, cpu, 2 threads)', {
		inputType: 'binary',
		binaryProperty: 'data',
		options: { threads: 2 },
	});

	// 2. quiet mode
	await run('quiet mode', {
		inputType: 'binary',
		binaryProperty: 'data',
		options: { threads: 2, noPrints: true },
	});

	// 3. write txt output
	const item3 = await run('write txt', {
		inputType: 'binary',
		binaryProperty: 'data',
		options: { threads: 2, outputTxt: true, outputFile: '/tmp/nemo-smoke' },
	});
	console.log('txt file exists:', fs.existsSync('/tmp/nemo-smoke.txt'));

	console.log('\nALL SMOKE TESTS PASSED');
})().catch((e) => {
	console.error('SMOKE TEST FAILED:', e);
	process.exit(1);
});
