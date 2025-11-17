from ..losses import (
    CrossEntropy,
    CrossEntropyWeighted,
    MSE,
    BrainTokenizerLoss,
    CrossEntropyWithCodes,
    NTDLoss,
    VQNSPLoss,
    NLL,
    VQVAELoss,
    ChronoFlowLoss,
    CrossEntropyBalanced,
    CrossEntropyMasked,
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
    CNNLSTMSimple,
    ClassifierContinuous,
    ClassifierQuantized,
    ClassifierQuantizedImage,
    GPT2MEGMix,
    GPT2MEG_Trf,
    ChronoFlowSSM,
    GPT2MEG_Cond,
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
        "CNNLSTMSimple": CNNLSTMSimple,
        "ClassifierContinuous": ClassifierContinuous,
        "ClassifierQuantized": ClassifierQuantized,
        "ClassifierQuantizedImage": ClassifierQuantizedImage,
        "GPT2MEGMix": GPT2MEGMix,
        "GPT2MEG_Trf": GPT2MEG_Trf,
        "ChronoFlowSSM": ChronoFlowSSM,
        "GPT2MEG_Cond": GPT2MEG_Cond,
    }
    if model_name not in model_classes:
        raise ValueError(f"Unknown model name: {model_name}")
    return model_classes[model_name]


def get_loss_class(loss_name: str):
    """Get loss class by name."""
    loss_classes = {
        "CrossEntropy": CrossEntropy,
        "CrossEntropyWeighted": CrossEntropyWeighted,
        "MSE": MSE,
        "BrainTokenizerLoss": BrainTokenizerLoss,
        "CrossEntropyWithCodes": CrossEntropyWithCodes,
        "NTDLoss": NTDLoss,
        "VQNSPLoss": VQNSPLoss,
        "NLL": NLL,
        "VQVAELoss": VQVAELoss,
        "ChronoFlowLoss": ChronoFlowLoss,
        "CrossEntropyBalanced": CrossEntropyBalanced,
        "CrossEntropyMasked": CrossEntropyMasked,
    }
    if loss_name not in loss_classes:
        raise ValueError(f"Unknown loss name: {loss_name}")
    return loss_classes[loss_name]
