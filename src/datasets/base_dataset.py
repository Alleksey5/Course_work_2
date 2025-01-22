import torch
import librosa
import os
import random
import numpy as np
import scipy.signal

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_list,
        segment_size=8192,
        sampling_rate=16000,
        split=True,
        shuffle=False,
        device=None,
        input_freq=None,
        lowpass="default",
    ):
        self.audio_files = file_list
        if shuffle:
            random.seed(1234)
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.device = device
        self.input_freq = input_freq
        self.lowpass = lowpass

    def split_audios(self, audios):
        audios = [torch.FloatTensor(audio).unsqueeze(0) for audio in audios]
        if self.split:
            if audios[0].size(1) >= self.segment_size:
                max_audio_start = audios[0].size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audios = [
                    audio[:, audio_start : audio_start + self.segment_size]
                    for audio in audios
                ]
            else:
                audios = [
                    torch.nn.functional.pad(
                        audio,
                        (0, self.segment_size - audio.size(1)),
                        "constant",
                    )
                    for audio in audios
                ]
        audios = [audio.squeeze(0).numpy() for audio in audios]
        return audios

    def low_pass_filter(self, audio):
        if self.lowpass == "default":
            tmp = librosa.resample(
                audio, self.sampling_rate, self.input_freq * 2, res_type="polyphase"
            )
        elif self.lowpass == "decimate":
            sub = self.sampling_rate / (self.input_freq * 2)
            assert int(sub) == sub
            tmp = scipy.signal.decimate(audio, int(sub))
        else:
            raise NotImplementedError
        tmp = librosa.resample(tmp, self.input_freq * 2, self.sampling_rate, res_type="polyphase")
        return tmp[: audio.size]

    def normalize(self, audio):
        return audio / (np.max(np.abs(audio)) + 1e-8)

    def __len__(self):
        return len(self.audio_files)