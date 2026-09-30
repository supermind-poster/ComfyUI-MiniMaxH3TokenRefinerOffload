from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger("MiniMaxH3TokenRefinerOffload")

_PATCH_KEY = "diffusion_model.token_refiner"
_MISSING = object()


# --------------------------------------------------------------------------- helpers
def _get_dm(model):
    inner = getattr(model, "model", None)
    dm = getattr(inner, "diffusion_model", None) if inner is not None else None
    if dm is None:
        raise RuntimeError("Input MODEL has no diffusion_model (not a ComfyUI ModelPatcher).")
    return dm


def _module_bytes(m):
    """Weight size of a module, measured the same way ModelPatcher measures it."""
    try:
        import comfy.model_management as mm
        return int(mm.module_size(m))
    except Exception:
        seen, total = set(), 0
        for t in list(m.parameters()) + list(m.buffers()):
            if id(t) not in seen:
                seen.add(id(t))
                total += t.numel() * t.element_size()
        return total


def _has_vbar(m):
    # dynamic-VRAM (aimdo) weights: ComfyUI streams them itself, never move them by hand
    return any(hasattr(x, "_v") for x in m.modules())


def _on_non_cpu(m):
    return any(p.device.type != "cpu" for p in m.parameters())


def _force_cast(real):
    """Make every ComfyUI-ops layer stream its weight to the input's device per call.

    Weights stay where they are (CPU); each layer copies only its own weight to the GPU
    for the duration of its matmul. Nothing is registered anywhere, nothing stays resident.
    """
    saved = []
    for mod in real.modules():
        if hasattr(mod, "comfy_cast_weights"):
            saved.append((mod, mod.__dict__.get("comfy_cast_weights", _MISSING)))
            mod.comfy_cast_weights = True
    return saved


def _restore_cast(saved):
    for mod, old in saved:
        if old is _MISSING:
            mod.__dict__.pop("comfy_cast_weights", None)
        else:
            mod.comfy_cast_weights = old


def _cannot_cast(real):
    """True if some weight-bearing layer is a plain torch layer without ComfyUI cast support."""
    for mod in real.modules():
        if any(True for _ in mod.parameters(recurse=False)) and not hasattr(mod, "comfy_cast_weights"):
            return True
    return False


# --------------------------------------------------------------------------- host
class _RefinerHost(nn.Module):
    """Parameter-free stand-in for `diffusion_model.token_refiner`.

    The real TokenRefiner lives in __dict__ (invisible to named_modules/state_dict), so the
    ModelPatcher never loads, pins or accounts it. It is executed only when someone calls the
    stand-in - in current ComfyUI that is `MiniMaxH3.extra_conds -> preprocess_text_embeds`,
    once per conditioning at the start of sampling. Math is identical to the original refiner.
    """

    def __init__(self, real, debug=True):
        super().__init__()
        d = self.__dict__
        d["_real"] = real
        d["_debug"] = bool(debug)
        d["_runs"] = 0
        d["_bytes"] = _module_bytes(real)

    @property
    def real(self):
        return self.__dict__["_real"]

    def forward(self, x, transformer_options={}):
        d = self.__dict__
        real = d["_real"]
        cuda = x.is_cuda
        before = torch.cuda.memory_allocated(x.device) if cuda else 0
        moved_by_hand = False
        saved = []
        try:
            if _cannot_cast(real) and _on_non_cpu(real) is False and cuda:
                real.to(x.device)          # fallback for non-ComfyUI layers only
                moved_by_hand = True
            else:
                saved = _force_cast(real)
            out = real(x, transformer_options=transformer_options)
        finally:
            _restore_cast(saved)
            # never leave refiner weights on the GPU (also cleans up stale residency
            # from an earlier load that still had the refiner registered)
            if (moved_by_hand or _on_non_cpu(real)) and not _has_vbar(real):
                real.to("cpu")
        d["_runs"] += 1
        if cuda:
            torch.cuda.synchronize(x.device)
            torch.cuda.empty_cache()
        if d["_debug"]:
            after = torch.cuda.memory_allocated(x.device) if cuda else 0
            logger.info(
                "[H3 Refiner Offload] refiner run #%d (refiner weights %.1f MiB, kept on CPU). "
                "GPU allocated: %.1f -> %.1f MiB, refiner on GPU now: %s",
                d["_runs"], d["_bytes"] / 1024**2, before / 1024**2, after / 1024**2,
                _on_non_cpu(real) and not _has_vbar(real),
            )
        return out


# --------------------------------------------------------------------------- node
class MiniMaxH3TokenRefinerOffload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": True}),
            "debug": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model/optimization"
    DESCRIPTION = (
        "Behaves like the 'norefiner' checkpoint during sampling (Token Refiner is not part of the "
        "loaded model), but still applies the refiner to the text conditioning before sampling starts."
    )

    def apply(self, model, enabled=True, debug=True):
        work = model.clone()
        if not enabled:
            return (work,)
        if _PATCH_KEY in work.object_patches:          # already installed upstream
            return (work,)

        dm = _get_dm(work)
        cur = getattr(dm, "token_refiner", None)
        real = cur.real if isinstance(cur, _RefinerHost) else cur
        if real is None or not hasattr(dm, "condition_proj") or not hasattr(real, "blocks"):
            raise RuntimeError("H3 Token Refiner Offload: model does not look like current ComfyUI MiniMaxH3Model.")
        if len(real.blocks) == 0:
            logger.info("[H3 Refiner Offload] token_refiner has 0 blocks (norefiner checkpoint) - nothing to do.")
            return (work,)

        # LoRA keys that target the refiner cannot be applied once it is outside the module tree.
        bad = [k for k in work.patches
               if "token_refiner." in str(k[0] if isinstance(k, tuple) else k)]
        if bad:
            raise RuntimeError(
                "H3 Token Refiner Offload: the model carries %d LoRA patch(es) for token_refiner.* . "
                "Bake them first or disable this node." % len(bad))

        host = _RefinerHost(real, debug=debug)
        # Object patch: applied by ModelPatcher.patch_model, undone by unpatch_model, and local to
        # this clone - the shared base model is never permanently modified.
        work.add_object_patch(_PATCH_KEY, host)

        # ModelPatcher decides full-load vs. partial-load from model_size(); the refiner is no
        # longer loadable, so remove it from the estimate (otherwise the model looks permanently
        # "partially loaded" and VRAM planning is off by the refiner size).
        try:
            full = work.model_size()
            work.size = max(int(full) - host.__dict__["_bytes"], 1)
        except Exception as e:                                   # pragma: no cover
            logger.warning("[H3 Refiner Offload] could not adjust model size: %s", e)

        logger.info(
            "[H3 Refiner Offload] installed: %.1f MiB refiner is outside the loaded model; "
            "it runs on the text conditioning before sampling and is not resident afterwards.",
            host.__dict__["_bytes"] / 1024**2)
        return (work,)


NODE_CLASS_MAPPINGS = {"MiniMaxH3TokenRefinerOffload": MiniMaxH3TokenRefinerOffload}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3TokenRefinerOffload": "MiniMax H3 Token Refiner — norefiner-style offload v3"}
