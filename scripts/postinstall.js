#!/usr/bin/env node
/**
 * postinstall: ensure the parakeet-cli binary is executable and the
 * ggml-parakeet-tdt-0.6b-v3-q8_0.bin model is present in the local cache.
 *
 * Model location (checked in order):
 *   1. $PARAKEET_MODEL_PATH          (explicit path to a .bin — used as-is, no download)
 *   2. <homedir>/.cache/n8n-nodes-parakeet/models/ggml-parakeet-tdt-0.6b-v3-q8_0.bin
 *
 * Env overrides:
 *   PARAKEET_MODEL_PATH  - use this exact file (skips download)
 *   PARAKEET_MODEL_URL   - override the download URL
 *   PARAKEET_NO_MODEL=1  - skip model download entirely (bring your own via PARAKEET_MODEL_PATH)
 */
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const https = require('https');
const http = require('http');

const MODEL_FILENAME = 'ggml-parakeet-tdt-0.6b-v3-q8_0.bin';
const DEFAULT_MODEL_URL =
  'https://huggingface.co/ggml-org/parakeet-GGUF/resolve/main/' + MODEL_FILENAME;
const EXPECTED_SIZE = 668757119; // bytes (q8_0)

function log(msg) {
  process.stderr.write('[n8n-nodes-parakeet] ' + msg + '\n');
}

function platformDir() {
  return path.join(__dirname, '..', 'bin', `${process.platform}-${process.arch}`);
}

function defaultModelPath() {
  return path.join(os.homedir(), '.cache', 'n8n-nodes-parakeet', 'models', MODEL_FILENAME);
}

function ensureBinary() {
  const binDir = platformDir();
  const exeName = process.platform === 'win32' ? 'parakeet-cli.exe' : 'parakeet-cli';
  const binPath = path.join(binDir, exeName);
  if (!fs.existsSync(binPath)) {
    log(`WARNING: prebuilt binary not found for ${process.platform}-${process.arch} at ${binPath}`);
    log('This package ships the linux-x64 binary. For other platforms, build via CI (see .github/workflows) and republish, or set PARAKEET_CLI_PATH.');
    return;
  }
  try {
    fs.chmodSync(binPath, 0o755);
  } catch (e) {
    log('WARNING: could not chmod +x the binary: ' + e.message);
  }
  // npm pack drops symlinks; recreate soname -> versioned-file links so the
  // dynamic loader can resolve DT_NEEDED entries (libparakeet.so.1, libggml.so.0, ...).
  const SONAME_LINKS = [
    ['libparakeet.so.1', 'libparakeet.so.1.9.4'],
    ['libggml.so.0', 'libggml.so.0.25.1'],
    ['libggml-base.so.0', 'libggml-base.so.0.25.1'],
    ['libggml-cpu.so.0', 'libggml-cpu.so.0.25.1'],
  ];
  for (const [link, target] of SONAME_LINKS) {
    const linkPath = path.join(binDir, link);
    const targetPath = path.join(binDir, target);
    if (!fs.existsSync(targetPath)) continue;
    try {
      fs.rmSync(linkPath, { force: true });
      fs.symlinkSync(target, linkPath);
    } catch (e) {
      log('WARNING: could not create symlink ' + link + ': ' + e.message);
    }
  }
}

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const tmp = dest + '.part';
    const file = fs.createWriteStream(tmp);
    let received = 0;
    let lastPct = -1;
    const reqMod = url.startsWith('https:') ? https : http;
    const req = reqMod.get(url, (res) => {
      if (res.statusCode !== 200) {
        // follow redirects
        if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location) {
          file.close();
          fs.unlink(tmp, () => {});
          return download(res.headers.location, dest).then(resolve, reject);
        }
        reject(new Error(`HTTP ${res.statusCode} for ${url}`));
        res.resume();
        return;
      }
      const total = parseInt(res.headers['content-length'] || '0', 10);
      res.on('data', (chunk) => {
        received += chunk.length;
        if (total) {
          const pct = Math.floor((received / total) * 100);
          if (pct > lastPct + 9 || pct === 100) {
            lastPct = pct;
            log(`downloading model... ${pct}% (${(received / 1e6).toFixed(0)} MB)`);
          }
        }
      });
      res.pipe(file);
      file.on('finish', () => {
        file.close(() => {
          fs.rename(tmp, dest, (err) => (err ? reject(err) : resolve()));
        });
      });
    });
    req.on('error', (e) => {
      file.close();
      fs.unlink(tmp, () => {});
      reject(e);
    });
  });
}

async function ensureModel() {
  if (process.env.PARAKEET_NO_MODEL === '1') {
    log('PARAKEET_NO_MODEL=1 set — skipping model download.');
    return;
  }

  const explicit = process.env.PARAKEET_MODEL_PATH;
  if (explicit) {
    if (fs.existsSync(explicit)) {
      log(`using model from PARAKEET_MODEL_PATH: ${explicit}`);
      return;
    }
    log(`WARNING: PARAKEET_MODEL_PATH=${explicit} does not exist.`);
  }

  const dest = defaultModelPath();
  if (fs.existsSync(dest) && fs.statSync(dest).size === EXPECTED_SIZE) {
    log(`model already present: ${dest}`);
    return;
  }

  const url = process.env.PARAKEET_MODEL_URL || DEFAULT_MODEL_URL;
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  log(`downloading ${MODEL_FILENAME} (~638 MB) from ${url}`);
  try {
    await download(url, dest);
    const size = fs.statSync(dest).size;
    if (size !== EXPECTED_SIZE) {
      log(`WARNING: downloaded size ${size} != expected ${EXPECTED_SIZE}. The model may be corrupt.`);
    } else {
      log(`model ready: ${dest}`);
    }
  } catch (e) {
    log('ERROR: model download failed: ' + e.message);
    log('Set PARAKEET_MODEL_PATH to a local copy of ' + MODEL_FILENAME + ' and re-run, or retry the install.');
    // Do not fail the whole npm install hard — the node will surface a clear error at run time.
  }
}

(async () => {
  ensureBinary();
  await ensureModel();
})();
