# n8n-nodes-parakeet

An [n8n](https://n8n.io) community node for **Parakeet** — offline speech-to-text powered by
[`parakeet.cpp`](https://github.com/ggml-org/whisper.cpp) (the `parakeet-cli` binary from the
official `ghcr.io/ggml-org/whisper.cpp:main` image) and the
`ggml-parakeet-tdt-0.6b-v3-q8_0` model.

Runs fully on-device / offline. No API keys, no cloud.

## Install into n8n

```bash
n8n add-nodes n8n-nodes-parakeet
# or, in a container:
docker exec -it <n8n> n8n nodes:install n8n-nodes-parakeet
```

On first run the node self-heals (no `postinstall`):
1. downloads the ~3 MB `parakeet-cli` + shared libraries to `~/.n8n/parakeet/bin/<platform>-<arch>/`, and
2. downloads the ~638 MB model to `~/.n8n/parakeet/models/ggml-parakeet-tdt-0.6b-v3-q8_0.bin`.

> Neither the binary nor the model is shipped in the npm package. Both are fetched once and
> cached under n8n's standard data volume (`~/.n8n`), so they survive container/pod restarts.
> To use local copies instead, set `PARAKEET_CLI_PATH` / `PARAKEET_MODEL_PATH`.

## Node inputs

| Input | Maps to `parakeet-cli` | Description |
|---|---|---|
| **Input** = Binary / URL | `-f FILE` | Audio source: a binary property from a previous node, or a URL (wav/mp3/flac/ogg). |
| **Binary Property** | — | Name of the binary property holding the audio (default `data`). |
| **Audio URL** | `-f` | Used when Input = URL. |
| **Model Path** | `-m FILE` | Leave empty to use the cached `q8_0` model. |
| **Options → Threads** | `-t N` | CPU threads (default 4). |
| **Options → Use GPU** | *(omit `-ng`)* | Enable GPU; requires a CUDA/Metal build. Default off. |
| **Options → GPU Device** | `-dev N` | GPU device index. |
| **Options → Print Segments** | `-ps` | Include per-segment / per-token detail. |
| **Options → Write Text File** | `-otxt` | Also write a `.txt` (returned as a `transcript` binary property). |
| **Options → Output File Base Path** | `-of FILE` | Base path for the `.txt` output. |
| **Options → Quiet** | `-np` | Suppress parakeet-cli logging; only the transcript is returned. |

Every `parakeet-cli` flag is exposed as a node input.

## Output

Each input item produces one output item:

```jsonc
{
  "json": {
    "text": "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
    "model": "ggml-parakeet-tdt-0.6b-v3-q8_0.bin",
    "input": "data"
  }
}
```

With **Write Text File** enabled, a `transcript` binary property (the `.txt`) is also attached.

## Environment variables

| Variable | Purpose |
|---|---|
| `PARAKEET_MODEL_PATH` | Use this exact `.bin` file; skips the auto-download. |
| `PARAKEET_CLI_PATH` | Use this exact `parakeet-cli` binary; skips the auto-download. |
| `PARAKEET_MODEL_URL` | Override the model download URL. |

## Binary + model (runtime download)

The npm package contains **no** binaries or weights. On first run the node downloads:

- `parakeet-cli` + its `libparakeet`/`libggml` shared libraries (~3 MB) from a GitHub Release
  asset, extracted to `~/.n8n/parakeet/bin/<platform>-<arch>/`. The binary uses `$ORIGIN` rpath
  so it runs with no `LD_LIBRARY_PATH`.
- the ~638 MB model from Hugging Face to `~/.n8n/parakeet/models/`.

Both live under n8n's standard data volume, so they survive container/pod restarts. The binary
is extracted from the official `ghcr.io/ggml-org/whisper.cpp:main` Docker image by
`.github/workflows/build-and-publish.yml` (a new GitHub Release is cut per tag).

## Build from source

```bash
npm install --ignore-scripts   # n8n-workflow pulls a native dep that needs no build here
npm run build                  # tsc -> dist/nodes/Parakeet/Parakeet.node.js
node smoke.test.js             # end-to-end smoke test (needs model + jfk.wav)
```

## License

MIT. The `parakeet-cli` binary and the Parakeet model are subject to their respective
licenses (whisper.cpp / Parakeet).
