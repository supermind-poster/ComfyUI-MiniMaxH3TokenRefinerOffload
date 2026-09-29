
from __future__ import annotations

import gc
import logging
import types
from typing import Any

import torch

import comfy.model_management

logger = logging.getLogger("MiniMaxH3TokenRefinerOffload")


def _get_diffusion_model(model):
    inner = getattr(model, "model", None)
    if inner is None:
        raise RuntimeError("Input MODEL has no .model (expected ComfyUI ModelPatcher).")
    dm = getattr(inner, "diffusion_model", None)
    if dm is None:
        raise RuntimeError("MODEL.model has no diffusion_model.")
    return dm


def _module_parameter_bytes(module) -> int:
    total = 0
    seen = set()
    for p in module.parameters(recurse=True):
        if id(p) in seen:
            continue
        seen.add(id(p))
        try:
            total += p.numel() * p.element_size()
        except Exception:
            pass
    for b in module.buffers(recurse=True):
        if id(b) in seen:
            continue
        seen.add(id(b))
        try:
            total += b.numel() * b.element_size()
        except Exception:
            pass
    return total


def _device_of_module(module):
    for p in module.parameters(recurse=True):
        try:
            return p.device
        except Exception:
            pass
    for b in module.buffers(recurse=True):
        try:
            return b.device
        except Exception:
            pass
    return None


def _move_module(module, device):
    """
    Move the refiner as a whole. H3's refiner is an ordinary nn.Module tree,
    including ComfyUI operation modules. We intentionally do not touch the
    surrounding ModelPatcher or the 50 main H3 blocks.
    """
    module.to(device)
    return module


class _RefinerRuntimeState:
    def __init__(self, diffusion_model, offload_device="cpu", debug=False):
        self.dm = diffusion_model
        self.offload_device = torch.device(offload_device)
        self.debug = bool(debug)

        self.refiner = getattr(diffusion_model, "token_refiner", None)
        if self.refiner is None:
            raise RuntimeError("This model has no token_refiner.")

        self.original_condition_forward = diffusion_model.condition_proj.forward
        self.original_refiner_forward = self.refiner.forward

        # Maps the freshly-created condition_proj output to the identity of the
        # source text tensor. This lets us cache by the actual conditioning
        # tensor instead of by the transient projection tensor.
        self.projection_keys: dict[int, tuple] = {}

        # key -> refined [L, 5376] tensor.
        # We keep the result on the compute device because the whole point is
        # to avoid rerunning the refiner on every denoise step.
        self.cache: dict[tuple, torch.Tensor] = {}

        self.refiner_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.offloaded = False
        self.original_refiner_device = _device_of_module(self.refiner)
        self.bytes_estimate = _module_parameter_bytes(self.refiner)

        # Keep a stable owner reference so the bound methods are not lost.
        self._install()

    @staticmethod
    def _tensor_key(t: torch.Tensor) -> tuple:
        # Comfy normally reuses the same conditioning tensors for all sampler
        # steps. data_ptr is therefore a very cheap and robust identity key.
        # Storage offset/shape/dtype are included to avoid accidental reuse.
        try:
            return (
                int(t.untyped_storage().data_ptr()),
                int(t.storage_offset()),
                tuple(t.shape),
                str(t.dtype),
                str(t.device),
            )
        except Exception:
            return (id(t), tuple(t.shape), str(t.dtype), str(t.device))

    def _install(self):
        state = self

        def condition_forward(this, text_states):
            out = state.original_condition_forward(text_states)
            if isinstance(out, torch.Tensor):
                state.projection_keys[out.data_ptr()] = state._tensor_key(text_states)
                # Avoid unbounded growth if Comfy allocates transient tensors.
                if len(state.projection_keys) > 32:
                    state.projection_keys = dict(list(state.projection_keys.items())[-16:])
            return out

        def refiner_forward(this, x, transformer_options={}):
            key = state.projection_keys.get(x.data_ptr())

            # If this is a conditioning tensor we have already refined, return
            # the cached result and completely bypass attention + MLP.
            if key is not None and key in state.cache:
                state.cache_hits += 1
                cached = state.cache[key]
                if cached.device != x.device:
                    cached = cached.to(device=x.device, dtype=x.dtype)
                    state.cache[key] = cached
                return cached

            state.cache_misses += 1

            # If the refiner was offloaded and a genuinely new conditioning
            # tensor appears, temporarily bring it back, compute once, then
            # offload it again. This keeps the node correct for positive and
            # negative CFG conditioning and for a new prompt.
            if state.offloaded:
                _move_module(state.refiner, x.device)
                state.offloaded = False
                if state.debug:
                    logger.info(
                        "[H3 Refiner Offload] cache miss -> refiner back on %s",
                        x.device,
                    )

            state.refiner_calls += 1
            out = state.original_refiner_forward(
                x, transformer_options=transformer_options
            )

            if key is not None:
                # Detach: inference output does not need a backward graph.
                # Preserve the compute dtype and device.
                cached = out.detach()
                state.cache[key] = cached

            # Once we have computed this conditioning, the refiner is no longer
            # needed until another unseen conditioning tensor appears.
            if state.offload_device.type == "cpu":
                _move_module(state.refiner, state.offload_device)
                state.offloaded = True
                if torch.cuda.is_available():
                    # Synchronize before empty_cache so CUDA is not still
                    # holding references from the refiner kernels.
                    try:
                        torch.cuda.synchronize(x.device)
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
                gc.collect()

                if state.debug:
                    logger.info(
                        "[H3 Refiner Offload] refined %s -> cached; refiner moved to CPU",
                        tuple(x.shape),
                    )

            return out

        # Use bound-method-like functions. Assigning to an instance does not
        # alter ComfyUI's class implementation or other H3 models.
        diffusion_model = self.dm
        diffusion_model.condition_proj.forward = types.MethodType(
            lambda this, text_states: condition_forward(this, text_states),
            diffusion_model.condition_proj,
        )
        self.refiner.forward = types.MethodType(
            lambda this, x, transformer_options={}: refiner_forward(
                this, x, transformer_options=transformer_options
            ),
            self.refiner,
        )

    def cleanup(self):
        # Restore original methods. The node is normally used on a model clone,
        # but this makes repeated execution and model eviction safe.
        try:
            self.dm.condition_proj.forward = self.original_condition_forward
        except Exception:
            pass
        try:
            self.refiner.forward = self.original_refiner_forward
        except Exception:
            pass

    def status(self) -> str:
        mb = self.bytes_estimate / (1024 ** 2)
        return (
            f"MiniMax H3 Token Refiner Offload: "
            f"{self.refiner_calls} refiner pass(es), "
            f"{self.cache_hits} cache hit(s), "
            f"{self.cache_misses} cache miss(es), "
            f"estimated refiner parameters {mb:.1f} MiB, "
            f"offloaded={'yes' if self.offloaded else 'no'}"
        )


class MiniMaxH3TokenRefinerOffload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "offload_device": (["cpu"], {"default": "cpu"}),
                "debug": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model/optimization"
    DESCRIPTION = (
        "Caches MiniMax H3 Token Refiner output per conditioning tensor and "
        "moves the two refiner blocks to CPU after use. The 50 main H3 blocks "
        "remain untouched."
    )

    def apply(self, model, enabled=True, offload_device="cpu", debug=True):
        work_model = model.clone()

        if not enabled:
            return (work_model,)

        dm = _get_diffusion_model(work_model)

        # Only target MiniMax H3-like models. This avoids accidentally touching
        # a similarly named custom model.
        if not hasattr(dm, "token_refiner") or not hasattr(dm, "condition_proj"):
            raise RuntimeError(
                "MiniMax H3 Token Refiner Offload: input model does not look "
                "like the current ComfyUI MiniMaxH3Model."
            )

        existing = getattr(dm, "_h3_token_refiner_offload_state", None)
        if existing is not None:
            logger.warning("[H3 Refiner Offload] node already installed on this model clone.")
            return (work_model,)

        state = _RefinerRuntimeState(dm, offload_device=offload_device, debug=debug)
        dm._h3_token_refiner_offload_state = state

        # Important: the state is intentionally owned by the cloned diffusion
        # model. We do not install a global wrapper and do not modify ComfyUI's
        # ModelPatcher code.
        logger.info(
            "[H3 Refiner Offload] installed: %.1f MiB parameter/buffer estimate; "
            "refiner will be cached and offloaded after each unseen conditioning.",
            state.bytes_estimate / (1024 ** 2),
        )

        return (work_model,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TokenRefinerOffload": MiniMaxH3TokenRefinerOffload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TokenRefinerOffload": "MiniMax H3 Token Refiner — Cache + CPU Offload",
}
