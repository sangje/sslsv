from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.0001
    patience: int = 100
    tracked_metric: str = 'test_eer'
    tracked_mode: str = 'min'
    optimizer: str = 'adam'
    weight_reg: float = 0
    mixed_precision: bool = False


@dataclass
class WavAugmentConfig:
    enable: bool = True
    rir: bool = True
    musan: bool = True
    musan_noise_snr: Tuple[int, int] = (0, 15)
    musan_speech_snr: Tuple[int, int] = (13, 20)
    musan_music_snr: Tuple[int, int] = (5, 15)


@dataclass
class DataConfig:
    wav_augment: WavAugmentConfig = None
    frame_length: int = 32000
    max_samples: int = None
    train: str = '/media/user/Samsung_T5/sslsv/data/voxceleb1'
    trials: str = './data/vox1_test_trials'
    base_path: str = './data/'
    enable_cache: bool = False
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class ModelConfig:
    __type__: str = None


@dataclass
class Config:
    training: TrainingConfig
    data: DataConfig
    model: ModelConfig
    name: str = 'test'
    seed: int = 1717
    reproducibility: bool = False
