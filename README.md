
# MiniMax H3 Token Refiner — Cache + CPU Offload

Custom ComfyUI node for the current `MiniMaxH3Model`.

## What it does

Current ComfyUI runs the H3 `token_refiner` inside `_forward()` whenever the
Qwen3-VL context is 5120-dimensional. It is therefore called again on every
denoising call, not only once.

This node:

1. clones the input MODEL;
2. detects the MiniMax H3 diffusion model;
3. records the source conditioning tensor identity;
4. lets the real Token Refiner execute once for each distinct conditioning
   tensor (normally positive and negative CFG contexts);
5. caches the refined 5376-dimensional text embedding;
6. moves the Token Refiner module to CPU;
7. on later denoise calls, returns the cached embedding without running the
   refiner attention/MLP;
8. if a genuinely new conditioning tensor appears, temporarily moves the
   refiner back to the compute device, computes it once, caches it, and
   offloads again.

The 50 main H3 blocks are not modified.

## Installation

Copy the folder:

`MiniMaxH3TokenRefinerOffload`

to:

`ComfyUI/custom_nodes/`

Then restart ComfyUI.

## Workflow

```text
Load Diffusion Model
        |
        v
MiniMax H3 Token Refiner — Cache + CPU Offload
        |
        v
your existing H3 optimization nodes / sampler
```

Put it as early as practical in the MODEL chain, before nodes that execute
the model.

## Important

This is deliberately implemented as a runtime cache, not as a checkpoint
conversion. The first occurrence of each conditioning tensor still executes
the original refiner, so the first pass preserves the original H3 computation.
Subsequent calls reuse the exact refined tensor.

The node uses the current ComfyUI H3 object structure:
`diffusion_model.condition_proj -> token_refiner -> blocks/final_norm`.

If another custom node replaces `condition_proj`, `token_refiner`, or their
forward methods, place this node after that node or disable one of the
conflicting patches.

## Expected VRAM behavior

The two refiner blocks can be large. Once a conditioning has been refined,
their parameters are moved to CPU and CUDA cache is emptied. The cached
5376-dimensional text features remain on the GPU because the main H3 blocks
need them on every denoise call.

The exact VRAM reduction depends on the checkpoint, quantization method, and
ComfyUI casting implementation.

## Safety

The node works on a cloned MODEL. It does not edit ComfyUI's source files and
does not globally monkey-patch `MiniMaxH3Model`.

After disabling/removing the node, restart ComfyUI if another custom node has
also modified the same H3 modules.
