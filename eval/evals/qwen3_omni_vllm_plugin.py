"""vLLM model-registry adapter for the Qwen3-Omni thinker.

The local vLLM 0.9.2 wheel predates Qwen3-Omni.  The Qwen-maintained
``qwen3_omni`` source branch contains the model implementation, while the
wheel supplies the compiled vLLM kernels.  This module bridges those two
pieces without modifying the installed vLLM package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _patch_transformers_config_registration() -> None:
    """Make vLLM 0.9.2's legacy config registration coexist with TF 4.57."""

    from transformers.models.auto.configuration_auto import AutoConfig

    register = AutoConfig.register
    if getattr(register, "_sar_qwen3_omni_compat", False):
        return

    def compatible_register(model_type, config, exist_ok=False):
        # vLLM 0.9.2 registers a few configs that Transformers 4.57 already
        # knows.  The existing Transformers definition is the same contract;
        # allowing the duplicate is sufficient for the vLLM import path.
        return register(model_type, config, exist_ok=True)

    compatible_register._sar_qwen3_omni_compat = True
    AutoConfig.register = compatible_register


_patch_transformers_config_registration()

try:
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerConfig,
    )

    for _old_name, _new_name in (
        ("audio_token_index", "audio_token_id"),
        ("image_token_index", "image_token_id"),
        ("video_token_index", "video_token_id"),
    ):
        if not hasattr(Qwen3OmniMoeThinkerConfig, _old_name):
            setattr(
                Qwen3OmniMoeThinkerConfig,
                _old_name,
                property(lambda self, target=_new_name: getattr(self, target)),
            )
except Exception:
    pass

# Import vLLM only after the compatibility patch above.  The source branch is
# kept outside the repository and pinned by commit in the experiment log.
import vllm  # noqa: E402,F401


_SOURCE_ROOT = Path(
    "/home/hongbo/.cache/sar-reproduction/vllm-qwen3-omni-ca66cbf"
)
_SOURCE_FILE = _SOURCE_ROOT / "vllm/model_executor/models/qwen3_omni_moe_thinker.py"
_MODULE_NAME = "vllm.model_executor.models.qwen3_omni_moe_thinker"

if _MODULE_NAME in sys.modules:
    _implementation = sys.modules[_MODULE_NAME]
else:
    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE_FILE)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"cannot load Qwen3-Omni vLLM implementation from {_SOURCE_FILE}")
    _implementation = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = _implementation
    _spec.loader.exec_module(_implementation)


Qwen3OmniMoeThinkerForConditionalGeneration = (
    _implementation.Qwen3OmniMoeThinkerForConditionalGeneration
)
