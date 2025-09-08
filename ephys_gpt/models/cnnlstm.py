from torch import nn


class CNNLSTM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
