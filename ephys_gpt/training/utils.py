from ..losses import (
    CrossEntropy,
    MSE,
    BrainTokenizerLoss,
    CrossEntropyWithCodes,
    NTDLoss,
    VQNSPLoss,
    NLL,
    VQVAELoss,
)
from ..models import (
    GPT2MEG,
    STGPT2MEG,
    BrainTokenizer,
    BrainOmniSystem,
    NTD,
    BENDRForecast,
    MEGFormer,
    VideoGPTTokenizer,
    VideoGPT,
    Emu3VisionVQ,
    VQGPT2MEG,
    LITRA,
    WavenetFullChannel,
    Wavenet3D,
    TACA,
    TASA3D,
    CK3D,
    CNNLSTM,
)


def get_model_class(model_name: str):
    """Get model class by name."""
    model_classes = {
        "GPT2MEG": GPT2MEG,
        "STGPT2MEG": STGPT2MEG,
        "BrainTokenizer": BrainTokenizer,
        "BrainOmniSystem": BrainOmniSystem,
        "NTD": NTD,
        "BENDRForecast": BENDRForecast,
        "MEGFormer": MEGFormer,
        "VideoGPTTokenizer": VideoGPTTokenizer,
        "VideoGPT": VideoGPT,
        "Emu3VisionVQ": Emu3VisionVQ,
        "VQGPT2MEG": VQGPT2MEG,
        "LITRA": LITRA,
        "WavenetFullChannel": WavenetFullChannel,
        "Wavenet3D": Wavenet3D,
        "TACA": TACA,
        "TASA3D": TASA3D,
        "CK3D": CK3D,
        "CNNLSTM": CNNLSTM,
    }
    if model_name not in model_classes:
        raise ValueError(f"Unknown model name: {model_name}")
    return model_classes[model_name]


def get_loss_class(loss_name: str):
    """Get loss class by name."""
    loss_classes = {
        "CrossEntropy": CrossEntropy,
        "MSE": MSE,
        "BrainTokenizerLoss": BrainTokenizerLoss,
        "CrossEntropyWithCodes": CrossEntropyWithCodes,
        "NTDLoss": NTDLoss,
        "VQNSPLoss": VQNSPLoss,
        "NLL": NLL,
        "VQVAELoss": VQVAELoss,
    }
    if loss_name not in loss_classes:
        raise ValueError(f"Unknown loss name: {loss_name}")
    return loss_classes[loss_name]
