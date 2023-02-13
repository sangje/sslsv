import os
import numpy as np
import librosa
import glob

def convert_wav_to_npy(filepath):
      if filepath.endswith(".wav"):
          audio, sr = librosa.load(filepath)
          # Filepath 에서 .wav 확장자 제외
          filename = os.path.splitext(filepath)[0]
          # 'wb' 로 이미 있는 파일의 경우 삭제 후 저장.
          with open('{}.npy'.format(filename),'wb') as f:
            np.save(f, audio)
          print(f"Converted {filepath} to {filename}")
      else:
        print(f'{filepath} file does not exist...')

if __name__ == "__main__":
  for file in glob.glob("data/voxceleb1/**/*.wav",recursive=True):
    convert_wav_to_npy(file)