import os
import random
import torch
import librosa
from torch.utils.data import Dataset


class VCTKDataset(BaseDataset):
    def __init__(
        self,
        dataset_split_file,
        vctk_wavs_dir,
        segment_size=8192,
        sampling_rate=16000,
        split=True,
        shuffle=False,
        device=None,
        input_freq=None,
        lowpass="default",
    ):
        """
        Класс для работы с VCTK датасетом. 
        Наследует логику от BaseDataset.
        """
        audio_files = get_dataset_filelist(dataset_split_file, vctk_wavs_dir)

        random.seed(1234)
        if shuffle:
            random.shuffle(audio_files)

        super().__init__(data_paths=audio_files)

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.device = device
        self.input_freq = input_freq
        self.lowpass = lowpass
        self.clean_wavs_dir = vctk_wavs_dir

    def __getitem__(self, index):
        """
        Загружает и возвращает данные для одного примера.
        """
        vctk_fn = self.data_paths[index]

        vctk_audio = librosa.load(
            vctk_fn,
            sr=self.sampling_rate,
            res_type="polyphase",
        )[0]

        (vctk_audio,) = split_audios([vctk_audio], self.segment_size, self.split)

        lp_inp = low_pass_filter(
            vctk_audio,
            self.input_freq,
            lp_type=self.lowpass,
            orig_sr=self.sampling_rate,
        )

        input_audio = normalize(lp_inp)[None] * 0.95
        assert input_audio.shape[1] == vctk_audio.size, "Несоответствие размеров аудио!"

        input_audio = torch.FloatTensor(input_audio)
        audio = torch.FloatTensor(normalize(vctk_audio) * 0.95).unsqueeze(0)

        return input_audio, audio
