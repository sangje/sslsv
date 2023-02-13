import numpy as np
import soundfile as sf
from io import BytesIO
import os
from tqdm import tqdm
from glob import glob
import librosa


class AudioCache:
    data = {}

def create_audio_cache(dataset_config, verbose=False):
    if not dataset_config.enable_cache:
        return

    base_path = dataset_config.base_path

    files = []
    files += glob(os.path.join(base_path, 'simulated_rirs', '*/*/*.wav'))
    files += glob(os.path.join(base_path, 'musan_split', '*/*/*.wav'))
    files += glob(os.path.join(base_path, 'voxceleb1', '*/*/*.wav'))
    if 'voxceleb2' in dataset_config.trials:
        glob(os.path.join(base_path, 'voxceleb2', '*/*/*.wav'))
    
    print('Creating cache of audio files...')
    if verbose: files = tqdm(files)
    for path in files:
        with open(path, 'rb') as file_data:
            AudioCache.data[path] = file_data.read()

def read_audio(path):
    if AudioCache.data:
        if path in AudioCache.data:
            audio, sr = librosa.load(AudioCache.data[path], sr=22050)
            return audio
            #return sf.read(BytesIO(AudioCache.data[path]))
        else:
            raise Exception(f'File {path} was not cached')
    
    # 확장자에 따라 다르게 로드함.
    if path.endswith(".wav"):
        audio, sr = librosa.load(path, sr=22050)
    elif path.endswith(".npy"):
        audio = np.load(path)
    return audio

def load_audio(path, frame_length, num_frames=1, min_length=None):
    audio = read_audio(path)

    # Pad signal if it is shorter than min_length
    if min_length is None: min_length = frame_length
    if min_length and len(audio) < min_length:
        audio = np.pad(audio, (0, min_length - len(audio) + 1), 'wrap')

    # Load entire audio data if frame_length is not specified
    if frame_length is None: frame_length = len(audio)

    # Determine frames start indices
    idx = []
    if num_frames == 1:
        idx = [np.random.randint(0, len(audio) - frame_length + 1)]
    else:
        idx = np.linspace(0, len(audio) - frame_length, num=num_frames)

    # Extract frames
    data = [audio[int(i):int(i)+frame_length] for i in idx]
    data = np.stack(data, axis=0).astype(np.float32)

    return data # (num_frames, T)
