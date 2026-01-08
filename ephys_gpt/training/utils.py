from importlib import import_module

_MODEL_CLASS_PATHS = {
    "NTD": "ephys_gpt.models.ntd.NTD",
    "BENDRForecast": "ephys_gpt.models.bendr.BENDRForecast",
    "WavenetFullChannel": "ephys_gpt.models.wavenet.WavenetFullChannel",
    "Wavenet3D": "ephys_gpt.models.wavenet.Wavenet3D",
    "TASA3D": "ephys_gpt.models.tasa3d.TASA3D",
    "CNNLSTM": "ephys_gpt.models.cnnlstm.CNNLSTM",
    "CNNLSTMSimple": "ephys_gpt.models.cnnlstm.CNNLSTMSimple",
    "ClassifierContinuous": "ephys_gpt.models.classifier.ClassifierContinuous",
    "ClassifierQuantized": "ephys_gpt.models.classifier.ClassifierQuantized",
    "ClassifierQuantizedImage": "ephys_gpt.models.classifier.ClassifierQuantizedImage",
    "BrainOmniCausalTokenizer": "ephys_gpt.models.brainomni.BrainOmniCausalTokenizer",
    "BrainOmniCausalForecast": "ephys_gpt.models.brainomni.BrainOmniCausalForecast",
    "IBQMEGTokenizer": "ephys_gpt.models.tokenizers.ibq.IBQMEGTokenizer",
    "FlatGPT": "ephys_gpt.models.flatgpt.FlatGPT",
    "FlatGPTEmbeds": "ephys_gpt.models.flatgpt.FlatGPTEmbeds",
    "FlatGPTRVQ": "ephys_gpt.models.flatgpt.FlatGPTRVQ",
    "FlatGPTMix": "ephys_gpt.models.flatgpt.FlatGPTMix",
    "FlatGPTEmbedsRVQ": "ephys_gpt.models.flatgpt.FlatGPTEmbedsRVQ",
    "STGPT2MEG": "ephys_gpt.models.stgpt2meg.STGPT2MEG",
}

_LOSS_CLASS_PATHS = {
    "CrossEntropy": "ephys_gpt.losses.classification.CrossEntropy",
    "CrossEntropyWeighted": "ephys_gpt.losses.classification.CrossEntropyWeighted",
    "MSE": "ephys_gpt.losses.reconstruction.MSE",
    "CrossEntropyWithCodes": "ephys_gpt.losses.classification.CrossEntropyWithCodes",
    "NTDLoss": "ephys_gpt.losses.diffusion.NTDLoss",
    "VQNSPLoss": "ephys_gpt.losses.reconstruction.VQNSPLoss",
    "NLL": "ephys_gpt.losses.reconstruction.NLL",
    "VQVAELoss": "ephys_gpt.losses.reconstruction.VQVAELoss",
    "ChronoFlowLoss": "ephys_gpt.losses.reconstruction.ChronoFlowLoss",
    "CrossEntropyBalanced": "ephys_gpt.losses.classification.CrossEntropyBalanced",
    "CrossEntropyMasked": "ephys_gpt.losses.classification.CrossEntropyMasked",
    "CrossEntropySpectral": "ephys_gpt.losses.classification.CrossEntropySpectral",
    "SpectralLoss": "ephys_gpt.losses.reconstruction.SpectralLoss",
    "BrainOmniCausalTokenizerLoss": "ephys_gpt.losses.brainomni.BrainOmniCausalTokenizerLoss",
    "BrainOmniCausalForecastLoss": "ephys_gpt.losses.brainomni.BrainOmniCausalForecastLoss",
    "CosmosTokenizerLoss": "ephys_gpt.losses.cosmos.CosmosTokenizerLoss",
    "VQVAEHF": "ephys_gpt.losses.reconstruction.VQVAEHF",
    "IBQSimpleLoss": "ephys_gpt.losses.ibq.IBQSimpleLoss",
    "VQLPIPSWithDiscriminator": "ephys_gpt.losses.ibq.VQLPIPSWithDiscriminator",
}


def _load_from_path(dotted_path: str):
    """Resolve a dotted path to an object, importing lazily."""
    module_path, attr = dotted_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, attr)


def get_model_class(model_name: str):
    """Get model class by name."""
    try:
        return _load_from_path(_MODEL_CLASS_PATHS[model_name])
    except KeyError as exc:
        raise ValueError(f"Unknown model name: {model_name}") from exc


def get_loss_class(loss_name: str):
    """Get loss class by name."""
    try:
        return _load_from_path(_LOSS_CLASS_PATHS[loss_name])
    except KeyError as exc:
        raise ValueError(f"Unknown loss name: {loss_name}") from exc
