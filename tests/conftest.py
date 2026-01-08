import os
import sys
import types


def _install_osl_stub():
    """Provide a minimal stub for osl_ephys to avoid heavy optional deps in tests."""
    if "osl_ephys" in sys.modules:
        return

    osl = types.ModuleType("osl_ephys")
    preprocessing = types.ModuleType("osl_ephys.preprocessing")
    preprocessing.run_proc_batch = lambda *args, **kwargs: None
    glm = types.ModuleType("osl_ephys.glm")
    source_recon = types.ModuleType("osl_ephys.source_recon")
    parcellation = types.ModuleType("osl_ephys.source_recon.parcellation")
    parcellation.plot_source_topo = lambda *args, **kwargs: None
    source_recon.parcellation = parcellation
    report = types.ModuleType("osl_ephys.report")

    osl.preprocessing = preprocessing
    osl.glm = glm
    osl.source_recon = source_recon
    osl.report = report

    sys.modules["osl_ephys"] = osl
    sys.modules["osl_ephys.preprocessing"] = preprocessing
    sys.modules["osl_ephys.glm"] = glm
    sys.modules["osl_ephys.source_recon"] = source_recon
    sys.modules["osl_ephys.source_recon.parcellation"] = parcellation
    sys.modules["osl_ephys.report"] = report


def _install_pnpl_stub():
    """Stub pnpl.datasets classes used in datasplitter to avoid optional installs."""
    if "pnpl" in sys.modules:
        return

    pnpl = types.ModuleType("pnpl")
    datasets = types.ModuleType("pnpl.datasets")

    class _DummyLibriBrain:
        def __init__(self, *args, **kwargs):
            self.partition = kwargs.get("partition", "train")

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            import torch

            return torch.zeros(1), torch.tensor(0)

    datasets.LibriBrainPhoneme = _DummyLibriBrain
    datasets.GroupedDataset = type("GroupedDataset", (), {})
    pnpl.datasets = datasets

    sys.modules["pnpl"] = pnpl
    sys.modules["pnpl.datasets"] = datasets


def _install_transformers_stub():
    """Provide minimal HF classes referenced by optional adapters."""
    import importlib
    import torch
    from torch import nn

    try:
        base_mod = importlib.import_module("transformers")
    except Exception:
        base_mod = types.ModuleType("transformers")
        sys.modules["transformers"] = base_mod

    if not hasattr(base_mod, "AutoModel"):

        class _AutoModel(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                hidden = int(kwargs.get("hidden_size", 8))
                self.emb = nn.Embedding(64, hidden)

            def forward(self, input_ids=None, **kwargs):
                if input_ids is None:
                    raise ValueError("input_ids required")
                return self.emb(input_ids)

        class _AutoTokenizer:
            def __init__(self, *args, **kwargs):
                pass

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        base_mod.AutoModel = _AutoModel
        base_mod.AutoTokenizer = _AutoTokenizer
        sys.modules["transformers"] = base_mod

    config_mod_name = "transformers.models.smollm3.configuration_smollm3"
    modeling_mod_name = "transformers.models.smollm3.modeling_smollm3"
    qwen_cfg_mod_name = "transformers.models.qwen2_5_vl.configuration_qwen2_5_vl"
    qwen_model_mod_name = "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"
    qwen3_cfg_mod_name = "transformers.models.qwen3_vl.configuration_qwen3_vl"
    qwen3_model_mod_name = "transformers.models.qwen3_vl.modeling_qwen3_vl"
    minimax_cfg_mod_name = "transformers.models.minimax.configuration_minimax"
    minimax_model_mod_name = "transformers.models.minimax.modeling_minimax"

    if config_mod_name in sys.modules and modeling_mod_name in sys.modules:
        return

    config_mod = types.ModuleType(config_mod_name)
    modeling_mod = types.ModuleType(modeling_mod_name)
    masking_utils_mod = types.ModuleType("transformers.masking_utils")
    qwen_cfg_mod = types.ModuleType(qwen_cfg_mod_name)
    qwen_model_mod = types.ModuleType(qwen_model_mod_name)
    qwen3_cfg_mod = types.ModuleType(qwen3_cfg_mod_name)
    qwen3_model_mod = types.ModuleType(qwen3_model_mod_name)
    minimax_cfg_mod = types.ModuleType(minimax_cfg_mod_name)
    minimax_model_mod = types.ModuleType(minimax_model_mod_name)

    class SmolLM3Config:
        def __init__(self, **kwargs):
            self.hidden_size = int(kwargs.get("hidden_size", 16))
            self.block_size = kwargs.get("block_size", 1)

    class _Output:
        def __init__(self, hidden: torch.Tensor):
            self.last_hidden_state = hidden
            self.past_key_values = None

    class SmolLM3Model(nn.Module):
        def __init__(self, config: SmolLM3Config):
            super().__init__()
            self.config = config
            self.embed_tokens = nn.Embedding(64, config.hidden_size)

        def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            attention_mask=None,
            past_key_values=None,
            use_cache=None,
            **kwargs,
        ):
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds required")
                hidden = self.embed_tokens(input_ids)
            else:
                hidden = inputs_embeds
            return _Output(hidden)

    class Qwen2_5_VLTextConfig:
        def __init__(self, **kwargs):
            self.use_sliding_window = bool(kwargs.get("use_sliding_window", False))
            self.num_hidden_layers = int(kwargs.get("num_hidden_layers", 1))
            self.layer_types = kwargs.get("layer_types", [])
            self.hidden_size = int(kwargs.get("hidden_size", 16))

    class Qwen2_5_VLTextModel(nn.Module):
        def __init__(self, config: Qwen2_5_VLTextConfig):
            super().__init__()
            self.config = config
            self.embed_tokens = nn.Embedding(64, config.hidden_size)

        def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            attention_mask=None,
            past_key_values=None,
            use_cache=None,
            **kwargs,
        ):
            hidden = (
                self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
            )

            return _Output(hidden)

    config_mod.SmolLM3Config = SmolLM3Config
    modeling_mod.SmolLM3Model = SmolLM3Model
    qwen_cfg_mod.Qwen2_5_VLTextConfig = Qwen2_5_VLTextConfig
    qwen_model_mod.Qwen2_5_VLTextModel = Qwen2_5_VLTextModel

    class Qwen3VLTextConfig:
        default_theta = 500000.0

        def __init__(self, **kwargs):
            self.hidden_size = int(kwargs.get("hidden_size", 16))
            self.rope_parameters = kwargs.get("rope_parameters")
            self.use_cache = kwargs.get("use_cache", False)

    class Qwen3VLTextModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.embed_tokens = nn.Embedding(64, config.hidden_size)

        def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            attention_mask=None,
            past_key_values=None,
            use_cache=None,
            position_ids=None,
            **kwargs,
        ):
            hidden = (
                self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
            )
            return _Output(hidden)

    qwen3_cfg_mod.Qwen3VLTextConfig = Qwen3VLTextConfig
    qwen3_model_mod.Qwen3VLTextModel = Qwen3VLTextModel
    minimax_cfg_mod.MiniMaxConfig = SmolLM3Config
    minimax_model_mod.MiniMaxModel = SmolLM3Model
    masking_utils_mod.create_causal_mask = lambda **kwargs: torch.zeros(
        1, 1, kwargs["input_embeds"].shape[1], kwargs["input_embeds"].shape[1]
    )

    sys.modules[config_mod_name] = config_mod
    sys.modules[modeling_mod_name] = modeling_mod
    sys.modules[qwen_cfg_mod_name] = qwen_cfg_mod
    sys.modules[qwen_model_mod_name] = qwen_model_mod
    sys.modules[qwen3_cfg_mod_name] = qwen3_cfg_mod
    sys.modules[qwen3_model_mod_name] = qwen3_model_mod
    sys.modules[minimax_cfg_mod_name] = minimax_cfg_mod
    sys.modules[minimax_model_mod_name] = minimax_model_mod
    sys.modules["transformers.masking_utils"] = masking_utils_mod


def pytest_configure(config):
    # _install_osl_stub()
    # _install_pnpl_stub()
    # _install_transformers_stub()
    # Ensure local package import resolves to workspace path
    root = os.path.dirname(os.path.dirname(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
