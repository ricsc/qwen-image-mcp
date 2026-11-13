"""
Qwen-Image — standalone MCP sidecar service (lazy GPU residency).

Local, open-weights (Apache-2.0) text-to-image generation with Qwen-Image-2512.
Chosen for its significantly better in-image text rendering (multilingual,
incl. German) and because there is no external safety filter rejecting
legitimate prompts (e.g. network diagrams for presentations).

Memory strategy — this service does NOT keep its ~18 GB of weights permanently
on the GPU, so it can coexist with other GPU workloads:

  * Lazy load: the pipeline is loaded on the first generate_image call only.
  * Idle unload: after IDLE_UNLOAD_AFTER seconds without a generation the
    pipeline is released again (del + gc + torch.cuda.empty_cache), returning
    the VRAM to the other processes.

bitsandbytes 4-bit weights are deliberately kept fully on the GPU: diffusers'
enable_model_cpu_offload()/enable_sequential_cpu_offload() are incompatible
with bnb-quantized modules on this host (observed: ~11 GB unexpected RSS
growth and a silent host-RAM OOM kill).

Endpoints:
    POST /mcp                  MCP Streamable HTTP transport (JSON-RPC)
    GET  /health               Liveness probe (200 idle/ready, 503 loading/error)
    GET  /images               JSON list of generated images
    GET  /images/{filename}    Serve a generated PNG

Environment:
    PORT                Listen port (default: 8000)
    IMAGE_DIR           Output directory (default: /srv/images-qwen)
    IMAGE_BASE_URL      Absolute base URL used for ResourceLinks, must be
                        reachable from the MCP client, e.g. http://qwen-image:8000
    MODEL_ID            HuggingFace model (default: unsloth bnb-4bit build)
    LORA_ID             Lightning distillation LoRA repo (default: lightx2v 8-step)
    LORA_WEIGHT_NAME    Adapter filename inside LORA_ID
    IDLE_UNLOAD_AFTER   Seconds of inactivity before the model is unloaded
"""

import base64
import gc
import io
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ImageContent, ResourceLink, TextContent
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qwen-image")

PORT = int(os.environ.get("PORT", "8000"))
IMAGE_DIR = Path(os.environ.get("IMAGE_DIR", "/srv/images-qwen"))
BASE_URL = os.environ.get("IMAGE_BASE_URL", "").rstrip("/")
THUMB_MAX_WIDTH = 512  # px, width cap for the inline thumbnail sent to the LLM

MODEL_ID = os.environ.get(
    "MODEL_ID",
    "unsloth/Qwen-Image-2512-unsloth-bnb-4bit",  # 4-bit weights, ~18 GB resident
)
LORA_ID = os.environ.get("LORA_ID", "lightx2v/Qwen-Image-2512-Lightning")
LORA_WEIGHT_NAME = os.environ.get(
    "LORA_WEIGHT_NAME", "Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors"
)

# Idle-unload: release the GPU after this many seconds without a generation.
IDLE_UNLOAD_AFTER = float(os.environ.get("IDLE_UNLOAD_AFTER", "300"))

# Defaults for the two operating modes (auto-selected at load time).
LORA_STEPS, LORA_CFG = 8, 1.0      # Lightning-distilled: few steps, no CFG
PLAIN_STEPS, PLAIN_CFG = 20, 4.0   # undistilled fallback: more steps + true-CFG

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Qwen-Image")
# NOTE: clients reach us via the Docker service name (Host header "qwen-image:8000"),
# which the SDK's default DNS-rebinding allowlist (127.0.0.1/localhost only) rejects
# with HTTP 421. The port is bound to 127.0.0.1 on the host and the container sits
# on a trusted bridge network, so disable the header check for this service.
mcp = FastMCP(
    "Qwen-Image",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

# ---------------------------------------------------------------------------
# Model lifecycle (lazy load on first use, idle unload afterwards)
#   _lifecycle_lock: guards the load/unload transitions (one at a time)
#   _gen_lock:       serializes pipeline usage — and blocks the reaper while
#                    a generation is in flight. Acquired before _lifecycle_lock.
# ---------------------------------------------------------------------------
_lifecycle_lock = threading.Lock()
_gen_lock = threading.Lock()

MODEL_STATE: str = "idle"      # idle | loading | ready | error
MODEL_ERROR: str = ""
PIPELINE = None
LORA_LOADED: bool = False
LAST_USED: float = 0.0         # monotonic timestamp of last successful use
_LAST_ERROR_TS: float = 0.0
_RETRY_COOLDOWN_S = 30.0       # don't hammer a failing load in a tight loop


# ---------------------------------------------------------------------------
# Model load / unload
# ---------------------------------------------------------------------------
def _load_pipeline():
    """
    Load the pipeline (4-bit weights, bf16 compute) fully onto the GPU and
    attach the Lightning LoRA. Returns (pipe, lora_loaded).

    NOTE: enable_model_cpu_offload()/enable_sequential_cpu_offload() are
    deliberately NOT used — they are incompatible with bitsandbytes 4-bit
    modules on this host (silent host-RAM OOM kill, ~11 GB unexpected RSS).
    """
    import torch
    from diffusers import DiffusionPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available inside this container")

    logger.info("Loading %s (bf16 + 4-bit weights) ...", MODEL_ID)
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    lora_ok = True
    try:
        pipe.load_lora_weights(LORA_ID, weight_name=LORA_WEIGHT_NAME)
        # Make sure the adapter is actually active for inference (diffusers
        # versions differ on whether load_lora_weights auto-activates it).
        peft_cfg = getattr(pipe.transformer, "peft_config", None)
        if peft_cfg:
            pipe.transformer.set_adapters(list(peft_cfg.keys()))
            logger.info("Activated LoRA adapter(s): %s", list(peft_cfg.keys()))
        logger.info("Loaded Lightning LoRA %s (%s)", LORA_ID, LORA_WEIGHT_NAME)
    except Exception as exc:  # keep working undistilled if the adapter fails
        lora_ok = False
        logger.warning("Lightning LoRA unavailable, falling back to %d steps: %s", PLAIN_STEPS, exc)

    return pipe, lora_ok


def _ensure_loaded():
    """
    Return a ready pipeline, loading it on first use (lazy load).

    Concurrent callers block until the single load finishes. On failure the
    error is surfaced to the caller and retriable after _RETRY_COOLDOWN_S.
    """
    global MODEL_STATE, MODEL_ERROR, PIPELINE, LORA_LOADED, LAST_USED, _LAST_ERROR_TS

    if MODEL_STATE == "ready" and PIPELINE is not None:
        return PIPELINE
    if MODEL_STATE == "error" and time.monotonic() - _LAST_ERROR_TS < _RETRY_COOLDOWN_S:
        raise RuntimeError(
            f"Model load failed recently ({MODEL_ERROR}); please retry in a few seconds"
        )

    with _lifecycle_lock:
        if MODEL_STATE == "ready" and PIPELINE is not None:
            return PIPELINE
        MODEL_STATE = "loading"
        MODEL_ERROR = ""
        t0 = time.monotonic()
        try:
            pipe, lora_ok = _load_pipeline()
        except Exception as exc:
            MODEL_STATE = "error"
            MODEL_ERROR = str(exc)
            _LAST_ERROR_TS = time.monotonic()
            logger.exception("Failed to load model: %s", exc)
            raise RuntimeError(f"Model load failed: {exc}") from exc

        PIPELINE = pipe
        LORA_LOADED = lora_ok
        MODEL_STATE = "ready"
        LAST_USED = time.monotonic()
        import torch

        vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        logger.info(
            "Model ready in %.1fs (lora=%s, VRAM allocated: %.1f GB)",
            time.monotonic() - t0, lora_ok, vram_gb,
        )
        return PIPELINE


def _unload_pipeline() -> None:
    """Release the pipeline from GPU and RAM (called by the idle reaper)."""
    global PIPELINE, LORA_LOADED
    import torch

    del PIPELINE
    PIPELINE = None
    LORA_LOADED = False  # will be re-established (and reported) on next load
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reaper_loop() -> None:
    """
    Idle-unload watchdog: after IDLE_UNLOAD_AFTER seconds without a
    generation, release the model so other GPU consumers get the VRAM
    back. Blocks on _gen_lock until any in-flight generation has finished.
    """
    global MODEL_STATE
    while True:
        time.sleep(15)
        if MODEL_STATE != "ready" or PIPELINE is None:
            continue
        idle_for = time.monotonic() - LAST_USED
        if idle_for < IDLE_UNLOAD_AFTER:
            continue
        with _gen_lock:  # no generation in flight (blocks until it finishes)
            if MODEL_STATE != "ready":
                continue
            logger.info("Idle for %.0fs > %ds -> unloading model from GPU", idle_for, IDLE_UNLOAD_AFTER)
            MODEL_STATE = "idle"
            try:
                _unload_pipeline()
            except Exception as exc:
                logger.exception("Idle unload failed (model stays loaded): %s", exc)
            else:
                logger.info("Model unloaded — VRAM returned to other services")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def _resolve_base_url() -> str:
    """
    Return the base URL used for absolute ResourceLink URIs.

    Prefers IMAGE_BASE_URL (must be reachable from the MCP client);
    falls back to localhost for direct local testing.
    """
    if BASE_URL:
        return BASE_URL
    logger.warning("IMAGE_BASE_URL not set -> using http://localhost:%d", PORT)
    return f"http://localhost:{PORT}"


def _make_thumbnail(png_bytes: bytes) -> ImageContent:
    """
    Downscale a PNG to at most THUMB_MAX_WIDTH px wide and wrap it as
    an MCP ImageContent (base64), so the model can visually check the
    result without blowing up the context.
    """
    img = Image.open(io.BytesIO(png_bytes))
    if img.width > THUMB_MAX_WIDTH:
        ratio = THUMB_MAX_WIDTH / img.width
        img = img.resize((THUMB_MAX_WIDTH, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return ImageContent(
        type="image",
        data=base64.b64encode(buf.getvalue()).decode("ascii"),
        mimeType="image/png",
    )


def _clamp_dim(value: int) -> int:
    """Clamp a resolution dimension to a sane range, rounded down to /32."""
    value = max(512, min(int(value), 1536))
    return (value // 32) * 32


def _save_image(png_bytes: bytes) -> tuple[str, Path]:
    """
    Persist a generated image and return (filename, path).

    Filenames are stable, sortable and collision-safe:
    ``qwen_<unix-ts>_<6-hex>.png``.
    """
    filename = f"qwen_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
    out_path = IMAGE_DIR / filename
    out_path.write_bytes(png_bytes)
    return filename, out_path


def _vram_allocated_mb() -> int | None:
    """VRAM held by this process (only meaningful while the model is resident)."""
    if PIPELINE is None:
        return None  # avoid importing torch (and a CUDA context) while idle
    try:
        import torch

        return int(torch.cuda.memory_allocated() // (1024 * 1024))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------
@mcp.tool()
def generate_image(
    prompt: str,
    width: int = 1344,
    height: int = 768,
    steps: int | None = None,
    cfg: float | None = None,
    seed: int | None = None,
) -> list:
    """
    Generate an image with Qwen-Image (local, open-weights, no external safety filter).

    USE THIS TOOL for polished visuals — presentations, hero images, illustrations —
    and for images containing readable text (titles, labels, German/English text),
    because Qwen-Image renders in-image text well.

    Note: the first call after startup loads the model (~15-30s extra).

    Args:
        prompt: Image description. Describe layout, style and any exact text to include.
        width: Output width in px (512-1536, multiples of 32; default 1344 = 16:9 with 768).
        height: Output height in px (512-1536, multiples of 32; default 768).
        steps: Sampling steps. Default: auto (8 with Lightning LoRA, 20 without).
        cfg: True-CFG scale. Default: auto (1.0 with LoRA, 4.0 without).
        seed: Optional RNG seed for reproducibility.

    Returns:
        Text summary + inline thumbnail image + ResourceLink to the full-size PNG.
    """
    global LAST_USED
    import torch

    width, height = _clamp_dim(width), _clamp_dim(height)

    with _gen_lock:
        pipe = _ensure_loaded()  # lazy load on first call
        # Sampling params AFTER loading: LORA_LOADED is only known once the
        # pipeline exists (lazy load), and Lightning's 8-step/cfg-1.0 setting
        # must not be mixed with the plain 20-step/cfg-4.0 fallback.
        n_steps = steps if steps is not None else (LORA_STEPS if LORA_LOADED else PLAIN_STEPS)
        true_cfg = cfg if cfg is not None else (LORA_CFG if LORA_LOADED else PLAIN_CFG)

        generator = torch.Generator(device="cuda").manual_seed(
            seed if seed is not None else int(time.time() * 1000)
        )
        t0 = time.monotonic()
        logger.info("Generating %dx%d (%d steps, cfg %.1f): %.80s", width, height, n_steps, true_cfg, prompt)
        kwargs = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=n_steps,
            generator=generator,
        )
        if true_cfg > 1.0:
            kwargs["true_cfg_scale"] = true_cfg
            kwargs["negative_prompt"] = "blurry, low quality, watermark, deformed"
        else:
            kwargs["true_cfg_scale"] = 1.0
        try:
            output = pipe(**kwargs)
        except Exception as exc:
            logger.exception("Generation failed")
            raise RuntimeError(f"Image generation failed: {exc}") from exc
        image = output.images[0]
        LAST_USED = time.monotonic()
    elapsed = time.monotonic() - t0

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    filename, _ = _save_image(png_bytes)
    base_url = _resolve_base_url()
    uri = f"{base_url}/images/{filename}"

    logger.info("Generated %s (%.1f KiB in %.1fs) -> %s", filename, len(png_bytes) / 1024, elapsed, uri)

    summary = {
        "status": "ok",
        "model": MODEL_ID,
        "lora": LORA_LOADED,
        "file": filename,
        "url": uri,
        "width": width,
        "height": height,
        "steps": n_steps,
        "seed": seed,
        "elapsed_s": round(elapsed, 1),
    }
    return [
        TextContent(type="text", text=json.dumps(summary, ensure_ascii=False)),
        _make_thumbnail(png_bytes),
        ResourceLink(
            type="resource_link",
            name=filename,
            uri=uri,
            description="Generated image (PNG) via Qwen-Image",
            mimeType="image/png",
        ),
    ]


@mcp.tool()
def list_images(limit: int = 10) -> list:
    """
    List recently generated images (newest first).

    Args:
        limit: Maximum number of entries to return.
    """
    files = sorted(IMAGE_DIR.glob("qwen_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    base_url = _resolve_base_url()
    entries = [
        {
            "file": f.name,
            "url": f"{base_url}/images/{f.name}",
            "size_kb": round(f.stat().st_size / 1024, 1),
            "modified": int(f.stat().st_mtime),
        }
        for f in files[:limit]
    ]
    return [TextContent(type="text", text=json.dumps(entries, ensure_ascii=False))]


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """
    Liveness probe: 200 while idle/ready (service usable), 503 while the
    model is loading or a load attempt failed.
    """
    body = {
        "status": MODEL_STATE,
        "service": "qwen-image",
        "model": MODEL_ID,
        "lora": LORA_LOADED,
        "vram_allocated_mb": _vram_allocated_mb(),
        "idle_unload_after_s": IDLE_UNLOAD_AFTER,
        "error": MODEL_ERROR or None,
    }
    if MODEL_STATE in ("ready", "idle"):
        return body
    raise HTTPException(status_code=503, detail=body)


@app.get("/images")
def list_images_api() -> dict:
    """JSON listing of generated images (REST alternative to the MCP tool)."""
    files = sorted(IMAGE_DIR.glob("qwen_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "count": len(files),
        "images": [
            {"file": f.name, "url": f"/images/{f.name}", "size_kb": round(f.stat().st_size / 1024, 1)}
            for f in files
        ],
    }


@app.get("/images/{filename}")
def get_image(filename: str):
    """Serve a generated image PNG (guards against path traversal)."""
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    out_path = IMAGE_DIR / filename
    if not out_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(out_path, media_type="image/png")


# Mount the MCP Streamable HTTP transport. Expose the endpoint at BOTH "/" and
# "/mcp" behind one shared session manager, so clients can use either the bare
# base URL (http://qwen-image:8000) or the explicit /mcp path — both work.
# streamable_http_app() must be called once first: it creates the session
# manager lazily (its task group is started in _serve() below).
from mcp.server.fastmcp.server import StreamableHTTPASGIApp  # SDK internal
from starlette.applications import Starlette as StarletteApp
from starlette.routing import Route as McpRoute

mcp.streamable_http_app()  # triggers lazy session-manager creation
_mcp_handler = StreamableHTTPASGIApp(mcp.session_manager)
app.mount("/", StarletteApp(routes=[
    McpRoute("/", endpoint=_mcp_handler),
    McpRoute("/mcp", endpoint=_mcp_handler),
]))

# Idle-unload watchdog: releases the GPU after IDLE_UNLOAD_AFTER seconds of
# inactivity so other services get the VRAM back (lazy load means the
# service starts with zero model footprint).
threading.Thread(target=_reaper_loop, name="idle-unload", daemon=True).start()

if __name__ == "__main__":
    import asyncio

    import uvicorn

    async def _serve() -> None:
        """
        Run the FastAPI app while keeping the MCP Streamable HTTP session
        manager's task group alive (required by mcp>=1.x when embedding the
        transport in an existing ASGI app).
        """
        async with mcp.session_manager.run():
            config = uvicorn.Config(app, host="0.0.0.0", port=PORT)
            await uvicorn.Server(config).serve()

    logger.info(
        "Starting Qwen-Image on 0.0.0.0:%d (out: %s, base: %s, lazy-load, idle-unload after %ds)",
        PORT, IMAGE_DIR, BASE_URL or "unset", IDLE_UNLOAD_AFTER,
    )
    asyncio.run(_serve())
