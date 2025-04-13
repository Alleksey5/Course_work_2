import time
import torch
from src.metrics.base_metric import BaseMetric


class RTF(BaseMetric):
    def __init__(self, sample_rate=16000, *args, **kwargs):
        """
        Computes Real-Time Factor (RTF):
        RTF = total_inference_time / total_audio_duration
        """
        super().__init__(*args, **kwargs)
        self.sample_rate = sample_rate
        self.total_inference_time = 0.0
        self.total_audio_duration = 0.0

    def __call__(self, model, inputs, **kwargs):
        """
        Measures and accumulates inference time and audio duration.

        Args:
            model (torch.nn.Module): model to evaluate.
            inputs (Tensor): (B, T) or (B, 1, T) input waveform
        """
        device = inputs.device
        model = model.to(device)
        model.eval()

        batch_size = inputs.shape[0]
        signal_length = inputs.shape[-1]
        duration = (signal_length / self.sample_rate) * batch_size

        with torch.no_grad():
            start = time.time()
            _ = model(inputs)
            end = time.time()

        self.total_inference_time += (end - start)
        self.total_audio_duration += duration

    def compute(self):
        if self.total_audio_duration == 0:
            return float("inf")
        return self.total_inference_time / self.total_audio_duration

    def reset(self):
        self.total_inference_time = 0.0
        self.total_audio_duration = 0.0
