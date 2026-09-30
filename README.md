# MiniMax H3 Token Refiner — norefiner-style offload v3

Goal: during sampling the model behaves like `MiniMax-H3-...-norefiner` (the Token Refiner is not part of
the loaded model), but the conditioning is still refined exactly like in the original checkpoint.

## How current ComfyUI uses the refiner

`MiniMaxH3.extra_conds` calls `preprocess_text_embeds` (condition_proj + token_refiner) **once per
conditioning at the start of sampling**. `_forward` then receives already-refined text tokens and skips the
refiner. So there is nothing to cache per step - v3 has no cache (v2's identity-keyed cache is removed:
it could return stale results when a freed GPU address was reused).

## What the node does

* Replaces `diffusion_model.token_refiner` with a parameter-free stand-in via a ModelPatcher **object patch**
  (clone-local, undone automatically on unpatch; the shared base model is never mutated permanently).
* The real refiner is not in the module tree: ModelPatcher never loads/pins/streams it and it is not in
  `state_dict()`. It stays in system RAM (~1.4 GiB for the int8 model).
* When ComfyUI asks for the refined conditioning, the real refiner runs with ComfyUI's per-layer weight
  casting: each layer copies only its own weight to the GPU for its matmul. Nothing stays resident afterwards
  (weights that were left on the GPU by an earlier load are moved back to CPU; `empty_cache()` is called).
  Output is bit-identical to the original refiner.
* Subtracts the refiner size from the patcher's `model_size()` so full-load/partial-load decisions match the
  norefiner checkpoint.

## Notes

* Put all LoRA nodes **before** this node. LoRA patches that target `token_refiner.*` raise an error.
* Cost vs. the norefiner file: ~1.4 GiB extra system RAM, and a short refiner run per conditioning
  (positive/negative) at the start of every sampling.
* `debug=true` logs GPU allocated MiB before/after each refiner run and whether refiner weights are on GPU.
* With dynamic VRAM (aimdo), modules that already own a VBAR are left to ComfyUI and not moved by hand.
* If the base model was previously loaded *without* this node in the same session, use "Unload models" once
  so no stale refiner copy stays on the GPU.
