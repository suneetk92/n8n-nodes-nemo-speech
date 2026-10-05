# n8n-nodes-nemo-speech

An [n8n](https://n8n.io) community node for **NeMo Speech** — offline speech-to-text, speaker
diarization, voice-activity detection, punctuation/capitalization, inverse text normalization, and
translation, powered by [`NeMo-Speech.cpp`](https://github.com/NVIDIA/NeMo-Speech.cpp) (NVIDIA's
lightweight C++ inference runtime) and the `parakeet-tdt-0.6b-v3` model (Q8_0 GGUF, ~714 MB).

Runs fully on-device / offline on CPU. No API keys, no cloud, no GPU required.

## Install into n8n

```bash
n8n add-nodes n8n-nodes-nemo-speech
# or, in a container:
docker exec -it <n8n> n8n nodes:install n8n-nodes-nemo-speech
```

On first run the node self-heals (no `postinstall`):
1. downloads the static `nemo-speech` binary + `ffmpeg` to `~/.n8n/nemo-speech/bin/<platform>-<arch>/`, and
2. auto-downloads the default ASR model (`parakeet-tdt-0.6b-v3.q8_0.gguf`) to `~/.n8n/nemo-speech/models/` on first transcription.
3. auto-downloads any enabled companion models (diarization, VAD, punctuation, ITN grammars, translation) to the same directory on first use.

> No binary or model weights are shipped in the npm package. All are fetched once and cached
> under n8n's standard data volume (`~/.n8n`), so they survive container/pod restarts.
> To use local copies instead, set `NEMO_SPEECH_PATH` / `FFMPEG_PATH`, or point the relevant
> "Model Path" field at a local `.gguf`/directory.

## How it works

1. The audio (MP3, FLAC, OGG, M4A, WAV, …) is converted to 16 kHz mono PCM16 WAV via the bundled static `ffmpeg`.
2. `nemo-speech transcribe <wav> --device cpu` runs the NeMo-Speech.cpp runtime with the Q8_0 GGUF model.
3. The transcript (plain text, JSON, SRT, or WebVTT) is returned as a JSON property.

The parakeet-tdt model runs in offline full-context mode and handles arbitrarily long audio
(podcasts, lectures, meetings) — the >34-min empty-transcript bug some whisper.cpp-based tools hit
is specific to that implementation, not the model.

## Features

The static binary is built with **every** NeMo-Speech.cpp capability enabled: ASR, speaker
diarization, VAD, punctuation/capitalization (PnC), inverse text normalization (ITN), and
translation (NMT). Companion models are auto-downloaded from official sources on first use — see
[Companion models](#companion-models-auto-downloaded) below.

| Capability | On by default? | Notes |
|---|---|---|
| ASR transcription | ✅ always | `parakeet-tdt-0.6b-v3` (self-punctuating, multilingual). |
| PnC (punctuation/casing) | opt-in (`pnc`) | No-ops with a warning on self-punctuating models (e.g. parakeet-tdt), which already produce their own casing/punctuation; restores it for plain-text models (e.g. parakeet-ctc). |
| ITN (inverse text normalization) | opt-in (`itn`) | "twenty twenty four" → "2024". Adds marginal value on top of self-punctuating models (mainly ordinals/symbols); more useful for plain-text models. |
| Speaker diarization | opt-in (`diarize`) | Auto-downloads NVIDIA's `Nemotron-3-Diarization` model; tags each word with a 1-based speaker id (JSON output). |
| VAD masking / VAD-based endpointing | opt-in (`vadMasking` / `vadBasedEou`) | Auto-downloads the official Silero VAD 6.2.3 model. |
| Translation (NMT) | opt-in (`translateTo`) | Auto-downloads NVIDIA's `Riva-Translate-4B-Instruct-v2` model (4.2 GB) on first use. |

## Node inputs

| Input | Maps to `nemo-speech` | Description |
|---|---|---|
| **Input** = Binary / URL | — | Audio source: a binary property from a previous node, or a URL (wav/mp3/flac/ogg/m4a). |
| **Binary Property** | — | Name of the binary property holding the audio (default `data`). |
| **Audio URL** | positional arg | Used when Input = URL. |
| **Output Format** | `--format` | `text`, `json`, `srt`, `vtt`. JSON/SRT/VTT add word timestamps. |
| **VAD Masking** | `--vad-masking` | Enable VAD feature masking (auto-downloads the official Silero 6.2.3 GGUF). |
| **Diarize** | `--diarize` | Enable speaker diarization (auto-downloads Nemotron-3-Diarization). |
| **Endpointing** | `--endpointing` | Mid-stream end-of-utterance detection. |
| **Enable ITN** | `--itn-model-dir` (when on) | Off by default. Auto-downloads the official multi-language grammars on first use. |
| **Enable PnC** | `--pnc-model` (when on) | Off by default. Auto-downloads the official PnC BERT model on first use. |
| **Translate To** | `--translate-to` | Target language code (e.g. `es`, `de`, `fr`, `zh`). Enables translation (auto-downloads Riva-Translate-4B-Instruct-v2). |
| **Write Text File** | — | Also write a `.txt` (returned as a `transcript` binary property). |
| **Options → ASR Model** | `--model` | Select from supported models (Parakeet TDT 0.6B v3, Nemotron 3.5 ASR Streaming 0.6B, Nemotron Speech Streaming EN 0.6B, Parakeet CTC 1.1B) or Custom. Shows Hugging Face repo ID. |
| **Options → Custom ASR Model** | `--model` | Enter any Hugging Face repo ID (e.g. `nvidia/parakeet-tdt-0.6b-v3`) or local `.gguf` file path. Auto-downloads from HF. |
| **Options → Device** | `--device` | `cpu`, `cuda:0`, `metal`, `vulkan:0`. Default `cpu`. |
| **Options → Stream Mode** | `--stream` | Feed audio in 160 ms chunks through the streaming recognizer. |
| **Options → Word Timestamps** | `--word-times` | Include per-word timestamps (JSON). |
| **Options → Verbose Logs** | `--verbose` | Enable verbose diagnostics. |
| **Options → Quiet** | `--quiet` | Suppress all logging; only the transcript is returned. |
| **Options → LM Path** | `--lm-path` | KenLM `.bin`/`.arpa` for Flashlight beam search (requires a flashlight build). |
| **Options → Lexicon Path** | `--lexicon` | Flashlight lexicon TSV. |
| **Options → Tokenizer Path** | `--tokenizer` | SentencePiece tokenizer for OOV boosting. |
| **Options → Beam Size** | `--beam-size` | Flashlight beam width (default 32). |
| **Options → Beam Threshold** | `--beam-threshold` | Beam pruning threshold (default 20.0). |
| **Options → LM Weight** | `--lm-weight` | LM rescoring weight (default 0.8). |
| **Options → Word Score** | `--word-score` | Word insertion bonus (default 1.0). |
| **Options → Max Boost** | `--max-boost` | CTC max per-word boost (default 10.0). |
| **Options → Chunk Size** | `--chunk-sec` | CTC buffered window in seconds (default 0.16). |
| **Options → Left/Right Padding** | `--left-pad-sec` / `--right-pad-sec` | CTC context in seconds (default 1.92). |
| **Options → VAD Model Path** | `--vad-model` | Leave empty to auto-download the official Silero 6.2.3 GGUF when VAD masking/EOU is enabled. |
| **Options → VAD-Based Endpointing** | `--vad-based-eou` | Use the VAD timeline (instead of token-silence) for endpointing. |
| **Options → VAD Onset/Offset/Pad** | `--vad-onset` / `--vad-offset` / `--vad-pad-ms` | VAD thresholds and padding. |
| **Options → Diarization Model** | `--diar-model` | Select Nemotron-3-Diarization (default), Diar Streaming Sortformer 4spk-v2, or Custom. Shows Hugging Face repo ID. |
| **Options → Custom Diar Model** | `--diar-model` | Enter custom Hugging Face repo ID or local `.gguf` file path. |
| **Options → Profanity List Path** | `--profanity-list` | Path to a profanity filter file. |
| **Options → ITN Model Dir** | `--itn-model-dir` | Custom grammar directory; only used when Enable ITN is on. |
| **Options → PnC Model** | `--pnc-model` | NVIDIA PnC BERT Base EN (default) or Custom HF repo ID / local `.gguf` path. |
| **Options → NMT Model** | `--nmt-model` | Riva-Translate-4B-Instruct-v2 (default) or Custom HF repo ID / local `.gguf` path. |
| **Options → Speech Context** | `--speech-context` | Comma-separated words to boost in the transcript. |
| **Options → Output File Base Path** | — | Base path for the `.txt` output; only used when Write Text File is on. |

Every `nemo-speech transcribe` flag is exposed as a node input.

## Output

Each input item produces one output item:

```jsonc
{
  "json": {
    "text": "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
    "model": "parakeet-tdt-0.6b-v3 (default)",
    "input": "data"
  }
}
```

With **Diarize** + **Output Format** = JSON, each word in `text` (parsed as JSON) carries a
1-based `speaker` field. With **Write Text File** enabled, a `transcript` binary property (the
`.txt`) is also attached.

## Companion models (auto-downloaded)

All converted from **official sources** and hosted at
[suneetk/nemo-speech-companions](https://huggingface.co/suneetk/nemo-speech-companions) on
Hugging Face. Downloaded once, cached under `~/.n8n/nemo-speech/models/`.

| Companion | Source | Size |
|---|---|---|
| Diarization | `nvidia/Nemotron-3-Diarization` (official HF repo GGUF) | 107 MB |
| VAD | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) v6.2.3 (official pip package, converted) | 1.2 MB |
| PnC | NVIDIA's `punctuation_en_bert.nemo` (NGC, converted) | 139 MB |
| ITN grammars | NeMo-Speech.cpp's official `itn_configs.tar.bz2` release asset (Sparrowhawk, multi-language) | 2.3 MB |
| Translation | `nvidia/Riva-Translate-4B-Instruct-v2` (official HF repo, converted to Q8_0 GGUF) | 4.2 GB |

## Environment variables

| Variable | Purpose |
|---|---|
| `NEMO_SPEECH_PATH` | Use this exact `nemo-speech` binary; skips the auto-download. |
| `FFMPEG_PATH` | Use this exact `ffmpeg` binary; skips the auto-download. |
| `NEMO_SPEECH_MODEL_DIR` | Override the model cache directory (default `~/.n8n/nemo-speech/models`). |

## Binary (runtime download)

The npm package contains **no** binaries or weights. On first run the node downloads the static
`nemo-speech` binary (~14 MB) + static `ffmpeg` (~80 MB, from johnvansickle.com's static build)
from a GitHub Release asset, extracted to `~/.n8n/nemo-speech/bin/<platform>-<arch>/`. Both
binaries are fully static (zero shared-lib dependencies), so they run on both glibc and musl
(Alpine) targets.

The static binary is built from `NVIDIA/NeMo-Speech.cpp` source by
`.github/workflows/build-and-publish.yml` (a new GitHub Release is cut per tag). Two small patches
in `patches/` adapt the upstream build for a fully static (musl-compatible) binary:
`static-itn-build.patch` builds OpenFST/Sparrowhawk as static archives instead of shared libs, and
`static-itn-link.patch` links the ITN target against those archives with the linker flags needed
for OpenFST's static type-registration to work (`--whole-archive`) plus protobuf/re2/absl/zlib.

## Build from source

```bash
npm install --ignore-scripts   # n8n-workflow pulls a native dep that needs no build here
npm run build                  # tsc -> dist/nodes/NemoSpeech/NemoSpeech.node.js
node smoke.test.js             # end-to-end smoke test (needs model + jfk.wav)
```

## License

MIT. The `nemo-speech` binary is subject to the NeMo-Speech.cpp license (Apache-2.0).
The Parakeet ASR model is subject to the CC-BY-4.0 license. Companion models retain their
respective upstream licenses (NVIDIA Open Model License for Nemotron-3-Diarization, PnC, and
Riva-Translate; MIT for Silero VAD).
