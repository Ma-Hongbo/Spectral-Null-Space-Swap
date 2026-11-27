"""Process-wide compatibility shim for the isolated Qwen3-Omni vLLM env.

The vLLM 0.9.2 worker uses Python ``spawn``.  Loading this module through
``PYTHONPATH`` makes the Transformers 4.57 duplicate-config workaround
available in both the parent and every spawned worker.
"""

from __future__ import annotations


def _patch() -> None:
    try:
        from transformers.models.auto.configuration_auto import AutoConfig
    except Exception:
        return

    register = AutoConfig.register
    if getattr(register, "_sar_qwen3_omni_compat", False):
        return

    def compatible_register(model_type, config, exist_ok=False):
        return register(model_type, config, exist_ok=True)

    compatible_register._sar_qwen3_omni_compat = True
    AutoConfig.register = compatible_register

    try:
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeThinkerConfig,
        )

        # vLLM 0.9.2's Qwen2.5-Omni rotary helper uses *_token_index;
        # Transformers' Qwen3-Omni config renamed these fields to *_token_id.
        for old_name, new_name in (
            ("audio_token_index", "audio_token_id"),
            ("image_token_index", "image_token_id"),
            ("video_token_index", "video_token_id"),
        ):
            if not hasattr(Qwen3OmniMoeThinkerConfig, old_name):
                setattr(
                    Qwen3OmniMoeThinkerConfig,
                    old_name,
                    property(
                        lambda self, target=new_name: getattr(self, target)
                    ),
                )
    except Exception:
        pass

    # The vLLM 0.9.2 multiprocessing worker imports a fresh module graph.
    # Register the lazy model entry here as well as in the parent runner, so
    # every rank selects the Qwen3-Omni implementation instead of silently
    # falling back to TransformersForCausalLM.
    try:
        from vllm.model_executor.models import ModelRegistry

        implementation = (
            "qwen3_omni_vllm_plugin:Qwen3OmniMoeThinkerForConditionalGeneration"
        )
        for architecture in (
            "Qwen3OmniMoeForConditionalGeneration",
            "Qwen3OmniMoeModel",
            "Qwen3OmniMoeThinkerForConditionalGeneration",
        ):
            ModelRegistry.register_model(architecture, implementation)
    except Exception:
        # The main runner will perform the same registration after vLLM is
        # imported.  Startup must remain harmless for `python --help`.
        pass


_patch()
