# VoiceForge. Local-first voice synthesis architecture

> Technical analysis of local TTS and voice cloning pipelines. No cloud dependencies. All inference on-device.

---

## The problem with cloud voice synthesis

Every commercial TTS service, ElevenLabs, Play.ht, WellSaid, operates on the same model: your text goes up, audio comes back down. Your voice samples live on someone else's servers. Your content passes through someone else's inference pipeline. For prototyping, that's fine. For production audio work with sensitive content, proprietary characters, or contractual voice talent, it's a non-starter.

The question I started investigating in early 2024: can you build a voice synthesis studio that runs entirely on local hardware, clones a voice from a few seconds of audio, and produces output good enough for production use? Without sending a single byte to the internet?

The short answer turned out to be yes. But the architecture required to make it work reliably is more interesting than the answer itself.

---

## How voice cloning actually works, the 30-second version

A modern neural TTS model takes two inputs: text and a speaker embedding, a high-dimensional vector (typically 256 to 512 floats) that encodes the spectral characteristics of a specific voice. Timbre, pitch range, speaking rate tendencies, formant structure. The model conditions its output on this vector, generating speech that sounds like the target speaker saying the input text.

The speaker embedding is extracted from a reference audio clip, as short as 3-5 seconds in current architectures (Casanova et al., "XTTS", 2024). The extraction model (a speaker encoder) maps the raw waveform to a fixed-length vector in a learned speaker space, where similar voices cluster together.

This is the critical insight: once you have the embedding, you don't need the original audio anymore. The embedding is the voice, compressed to a few kilobytes. And computing it takes milliseconds.

---

## The chunking problem nobody talks about

Neural TTS models have a dirty secret: they fall apart on long inputs. Feed a 2000-character paragraph to most transformer-based TTS architectures and you'll get degrading prosody after the first 500-800 characters, occasional hallucinated syllables, and sometimes outright truncation. The attention mechanism wasn't designed to track coherent prosody across minutes of output.

The solution is chunked generation with crossfade stitching:

The input text is segmented at sentence boundaries (configurable maximum, typically 600-800 characters). Each chunk is synthesized independently, same speaker embedding, same model, independent attention context. The resulting audio segments are joined through linear crossfade: the last N milliseconds of chunk $k$ overlap with the first N milliseconds of chunk $k+1$, with amplitude ramping:

$$y(t) = (1 - \alpha(t)) \cdot a_k(t) + \alpha(t) \cdot a_{k+1}(t)$$

where $\alpha(t)$ ramps linearly from 0 to 1 over the crossfade window (typically 30-50 ms). A final loudness normalization pass equalizes levels across chunks.

Without this pipeline, you cannot reliably synthesize text longer than about 30 seconds. With it, you can synthesize books.

---

## Inference serialization: why a queue matters for GPU workloads

GPU memory is not thread-safe in the way people assume. Two concurrent CUDA inference calls on the same device will either crash with an out-of-memory error or produce corrupted output, depending on the framework and the phase of the moon. This is especially dangerous in a server context where multiple UI actions (preview, full generation, re-generation) can fire simultaneously.

The standard pattern, and the one that works, is an async inference queue:

```
Request → asyncio.Queue (FIFO) → Worker thread → GPU → Result → Callback
```

One worker, one GPU, strict serialization. The queue provides backpressure (the UI shows position in queue), cancellation (remove from queue before processing starts), and progress reporting (the worker emits chunk-level progress events). All other API endpoints remain responsive because the inference runs in `asyncio.to_thread()`, off the event loop.

I've seen three projects that tried to "optimize" this with concurrent GPU access. All three shipped with intermittent crash bugs that took months to diagnose.

---

## Speaker embedding cache and lazy invalidation

Computing a speaker embedding from a 5-second audio clip takes about 200ms on a mid-range GPU. Not slow. But if you're iterating on generations, tweaking text, adjusting parameters, comparing outputs, that 200ms adds up. Fifty iterations and you've burned 10 seconds on redundant computation.

The fix is obvious: cache the embedding, keyed to the hash of the reference audio files. But the invalidation logic has a subtlety. A voice profile can have multiple audio samples. Adding, removing, or replacing any sample should invalidate the cache. Modifying text or generation parameters should not. The cache key is therefore:

$$\text{key} = \text{SHA256}\left(\bigoplus_{i=1}^{n} \text{bytes}(s_i)\right)$$

where $s_i$ are the audio sample files in deterministic order. Any change to the sample set produces a different hash; everything else is a cache hit.

---

## Platform-dependent inference: the MLX factor

Apple Silicon changed the equation for local ML inference. The MLX framework (Apple, 2023) runs transformer models on the Metal GPU and Neural Engine of M-series chips with 4-5x speedup over PyTorch on the same hardware. For voice synthesis, this means real-time factor below 0.3 on an M2, you generate 1 second of audio in 0.3 seconds of compute.

The architecture I arrived at uses a factory pattern for inference backends:

```python
def create_backend(platform: str) -> InferenceBackend:
    if platform == "darwin" and has_mlx():
        return MLXBackend()
    elif has_cuda():
        return CUDABackend()
    elif has_rocm():
        return ROCmBackend()
    return CPUBackend()
```

Client code never knows which backend is running. Adding a new backend means implementing the interface and registering it. The detection runs once at startup.

On x86 Linux with an NVIDIA GPU, CUDA gives excellent performance. On AMD, ROCm works but with occasional compatibility hiccups. CPU fallback exists for machines with no discrete GPU, it's slow (real-time factor ~3-5x) but functional.

---

## Schema-driven full-stack typing

One pattern I've come to consider non-negotiable for any project with a Python backend and TypeScript frontend: Pydantic models as the single source of truth.

The backend defines every request and response as a Pydantic model. FastAPI auto-generates an OpenAPI schema from these models. A code generator (openapi-typescript or similar) reads the schema and produces TypeScript types. The frontend API client is typed against these generated types.

The result: change a field name in a Python model, and the TypeScript compiler catches every broken reference at build time. Zero runtime surprises from contract drift. Zero manual type synchronization. I've maintained projects where the API had 40+ endpoints and this pattern caught breaking changes within seconds of making them.

---

## The effects chain

Post-synthesis audio processing runs as an ordered list of DSP operations on raw numpy arrays. Pitch shift (in semitones), reverb (configurable room size), and EQ (bass/mid/treble bands). Each effect has parameters and an enable flag. Chains are saved as presets, built-in and user-defined. A preset can be bound to a voice profile, so effects apply automatically on every generation.

This sounds simple, and it is. The interesting part is that the effects chain runs after TTS but before caching, so the same text + voice + seed combination with different effects produces different cached outputs. The cache key includes the serialized effects chain hash.

---

## What this study doesn't cover

This is an architecture analysis, not a product. I didn't build a shipping application. I studied the problem space, prototyped the critical subsystems (chunking pipeline, inference queue, embedding cache, backend abstraction), and documented what I learned.

What's deliberately out of scope: real-time speech-to-text, language models for response generation, voice assistants, dialog agents, multi-GPU parallelism. Voice synthesis is one problem. It's a deep one. Mixing it with adjacent problems produces shallow solutions to all of them.

---

## References

1. Casanova, E. et al. "XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model." arXiv:2406.04904, 2024.
2. Apple Machine Learning Research. "MLX: An array framework for Apple Silicon." github.com/ml-explore/mlx, 2023.
3. Pydantic documentation. "Schema generation and validation." docs.pydantic.dev, 2024.
4. Kim, J. et al. "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech." ICML, 2021.

---



Author: Serghei Brinza, AI engineer. Other projects: [github.com/SergheiBrinza](https://github.com/SergheiBrinza)
