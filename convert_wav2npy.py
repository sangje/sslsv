import os
import numpy as np
import soundfile as sf
import glob

def convert_wav_to_npy(filename):
      if filename.endswith(".wav"):
          audio, sr = sf.read(filename)
          npy_filename = os.path.splitext(filename)[0] + ".npy"
          np.save(npy_filename, audio)
          print(f"Converted {filename} to {npy_filename}")

if __name__ == "__main__":
  for file in glob.glob("**/*.wav",recursive=True):
    convert_wav_to_npy(file)