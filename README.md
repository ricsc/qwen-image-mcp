# qwen-image-mcp

Standalone [MCP](https://modelcontextprotocol.io) sidecar for **Qwen-Image-2512** — local,
open-weights (Apache-2.0) text-to-image generation with strong in-image text rendering
(German/English labels, titles), no cloud API, no safety filter, no Web UI.

Built for Open WebUI: register it as an MCP server and the LLM gets a
`generate_image` tool that returns an inline thumbnail plus a full-size PNG link.

## Stack

| Component | Version |
|---|---|
| Base image | `python:3.11-slim-bookworm` (CUDA 13 runtime ships inside the torch wheels) |
| torch | 2.14.1 (cu130) |
| diffusers / transformers | 0.41.0 / 5.20.2 |
| Model | `unsloth/Qwen-Image-2512-unsloth-bnb-4bit` (~18 GB resident, 4-bit weights) |
| LoRA | `lightx2v/Qwen-Image-2512-Lightning` (8-step by default, 4-step variant available, cfg 1.0) |

## Quick start

```bash
docker compose up -d --build
curl http://127.0.0.1:8000/health   # {"status": "idle", ...}
```

First `generate_image` call downloads the model (~19 GB) into `./data/hf-cache` if not
present yet, loads it (~15–30 s), then generates (~10 s). Subsequent calls are fast.

### Register in Open WebUI

Add an MCP server with URL:

```
http://qwen-image:8000
```

(The explicit `http://qwen-image:8000/mcp` also works — the endpoint is mounted on both paths.)

(open-webui must share the Docker network `openwebui_default` — same as before.)

For a screenshot-based setup guide and sample images showing how to parametrize
generations (`width`/`height`, `steps`/`cfg`), see [examples/](examples/README.md).

### Register in OpenCode

OpenCode runs on the host, so it points at the bound localhost port (not the
container name). Add an `mcp` entry to `~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "mcp": {
    "image-gen": {
      "type": "remote",
      "url": "http://127.0.0.1:8000/mcp",
      "oauth": false
    }
  }
}
```

After restarting OpenCode, the `generate_image` / `list_images` tools are
available to any agent (verify with `/mcp`).

## GPU memory strategy (lazy load + idle unload)

The service does **not** hold its ~18 GB of weights permanently on the GPU:

- **Lazy load** — starts with zero VRAM footprint; loads on first generation.
- **Idle unload** — after `IDLE_UNLOAD_AFTER` seconds without a generation the
  pipeline is released (`del + gc + empty_cache`) and the VRAM is available to
  other services again.

## Configuration (environment)

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Listen port inside the container |
| `IMAGE_DIR` | `/srv/images-qwen` | Output directory (mounted at `./images`) |
| `IMAGE_BASE_URL` | — | Base URL for image links, must be reachable from the MCP client (`http://qwen-image:8000`) |
| `HF_HOME` | `/data/hf-cache` | HuggingFace cache (mounted at `./data`) |
| `MODEL_ID` | `unsloth/Qwen-Image-2512-unsloth-bnb-4bit` | Base model repo |
| `PIPELINE_SHELL_ID` | `unsloth/Qwen-Image-2512-unsloth-bnb-4bit` | Repo supplying VAE/text-encoder/tokenizer/scheduler |
| `LORA_ID` / `LORA_WEIGHT_NAME` | lightx2v Lightning 8-step | Distillation LoRA (auto-fallback to 50 steps/cfg 4.0 if missing). The step count follows the weight filename (`…4steps…` → 4); `LORA_STEPS` overrides it |
| `LORA_SHIFT` | `3.0` | Flow-match sigma shift on the LoRA path. 3.0 tracks the Lightning distillation schedule; 12–13 (unsloth's "detail fix" for their 40-step workflow) pushes denoising into the final steps — test before adopting |
| `LORA_CFG` | `1.0` | CFG scale on the LoRA path. Lightning was distilled with cfg 1.0 — higher values deviate from the distillation recipe, test before adopting |
| `IDLE_UNLOAD_AFTER` | `300` | Seconds of inactivity before GPU release |

## Endpoints

- `POST /mcp` **and** `POST /` — MCP Streamable HTTP (both paths work; tools: `generate_image`, `list_images`)
- `GET  /health` — `200` idle/ready, `503` loading/error; includes VRAM usage
- `GET  /images` — JSON list of generated PNGs
- `GET  /images/{file}` — the PNG itself

## Notes

- bitsandbytes 4-bit weights are kept fully on the GPU on purpose:
  `enable_model_cpu_offload()` is incompatible with bnb modules on this host
  (silent host-RAM OOM kill).
- Generation is serialized (one at a time); the idle-unload reaper waits for an
  in-flight generation to finish.
