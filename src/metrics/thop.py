import torch
from src.metrics.thop_nets import profile, clever_format
from src.metrics.base_metric import BaseMetric


class THOPMetric(BaseMetric):
    """
    Computes MACs and number of parameters using THOP.
    """

    def __init__(self, input_shape=(1, 1, 16000), verbose=True, *args, **kwargs):
        """
        Args:
            input_shape (tuple): Shape of dummy input (B, C, T)
            verbose (bool): Whether to print formatted MACs and params
        """
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.verbose = verbose
        self.macs = None
        self.params = None

    def __call__(self, model, **kwargs):
        model.eval()
        dummy_input = torch.randn(*self.input_shape).to(next(model.parameters()).device)

        with torch.no_grad():
            macs, params = profile(model, inputs=(dummy_input,), verbose=False)
            macs_readable, params_readable = clever_format([macs, params], "%.3f")

        self.macs = macs
        self.params = params

        if self.verbose:
            print(f"[THOP] MACs: {macs_readable}, Params: {params_readable}")

        return {"macs": macs, "params": params, "macs_readable": macs_readable, "params_readable": params_readable}
