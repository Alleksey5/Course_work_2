import os
import random
import torch
import librosa
import numpy as np
from scipy.io.wavfile import read
from librosa.util import normalize
from src.datasets.base_dataset import BaseDataset

class VCTKDataset(BaseDataset):
    """
    VCTK Dataset implementation inheriting from BaseDataset.
    This dataset handles VCTK audio files for speech processing tasks.
    """

    def __init__(
        self, dataset_split_file, vctk_wavs_dir, segment_size=8192, sampling_rate=16000,
        split=True, shuffle=False, instance_transforms=None, limit=None
    ):
        """
        Args:
            dataset_split_file (str): Path to the dataset split file.
            vctk_wavs_dir (str): Directory where VCTK WAV files are stored.
            segment_size (int): Size of each audio segment.
            sampling_rate (int): Sampling rate for audio.
            split (bool): Whether to split the audio into segments.
            shuffle (bool): Whether to shuffle the dataset.
            instance_transforms (dict | None): Dictionary of transforms to apply to instances.
            limit (int | None): Maximum number of dataset elements to include.
        """
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split

        self.audio_files = self._load_file_list(dataset_split_file, vctk_wavs_dir)
        if shuffle:
            random.shuffle(self.audio_files)

        index = self._create_index()
        super().__init__(index, limit=limit, shuffle_index=False, instance_transforms=instance_transforms)

    def _load_file_list(self, dataset_split_file, vctk_wavs_dir):
        """Load list of audio file paths from dataset split file."""
        with open(dataset_split_file, "r", encoding="utf-8") as f:
            files = [os.path.join(vctk_wavs_dir, line.strip()) for line in f if line.strip()]
        return files

    def _create_index(self):
        """Create index for dataset."""
        index = []
        for file_path in self.audio_files:
            index.append({"path": file_path})
        return index

    def load_object(self, path):
        """Load an audio file from disk and preprocess it."""
        audio, _ = librosa.load(path, sr=self.sampling_rate, res_type="polyphase")
        if self.split:
            audio = self._split_audio(audio)
        audio = torch.FloatTensor(normalize(audio) * 0.95)
        return audio

    def _split_audio(self, audio):
        """Split audio into smaller segments if needed."""
        if len(audio) >= self.segment_size:
            max_start = len(audio) - self.segment_size
            start = random.randint(0, max_start)
            return audio[start : start + self.segment_size]
        return np.pad(audio, (0, self.segment_size - len(audio)), mode="constant")
