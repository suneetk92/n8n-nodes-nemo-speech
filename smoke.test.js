'use strict';
// Throwaway smoke test: load the compiled node and run execute() with a mock IExecuteFunctions.
const fs = require('fs');
const path = require('path');

const { Parakeet } = require('./dist/nodes/Parakeet/Parakeet.node.js');

const wavPath = '/root/projects/its/.build/jfk.wav';
const modelPath = '/root/projects/its/.build/models/ggml-parakeet-tdt-0.6b-v3-q8_0.bin';

function makeMockExecuteFunctions(params) {
	const node = { id: 'n1', name: 'Parakeet STT', typeVersion: 1, type: 'parakeet' };
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
	const node = new Parakeet();
	const fn = makeMockExecuteFunctions(params);
	const out = await node.execute.call(fn);
	const item = out[0][0];
	console.log(`\n=== ${label} ===`);
	console.log('json.text:', JSON.stringify(item.json.text));
	if (item.binary) console.log('binary keys:', Object.keys(item.binary));
	return item;
}

(async () => {
	// 1. basic: binary input, no-gpu, 2 threads
	await run('basic (binary, -ng -t2)', {
		inputType: 'binary',
		binaryProperty: 'data',
		modelPath,
		options: { threads: 2 },
	});

	// 2. with segments + quiet
	const item2 = await run('segments+quiet (-ps -np)', {
		inputType: 'binary',
		binaryProperty: 'data',
		modelPath,
		options: { threads: 2, printSegments: true, noPrints: false },
	});

	// 3. write txt output
	const item3 = await run('write txt (-otxt -of)', {
		inputType: 'binary',
		binaryProperty: 'data',
		modelPath,
		options: { threads: 2, outputTxt: true, outputFile: '/tmp/parakeet-smoke' },
	});
	console.log('txt file exists:', fs.existsSync('/tmp/parakeet-smoke.txt'));

	console.log('\nALL SMOKE TESTS PASSED');
})().catch((e) => {
	console.error('SMOKE TEST FAILED:', e);
	process.exit(1);
});
