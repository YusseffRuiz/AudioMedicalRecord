import os
import uuid

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioRecorder:
    def __init__(self, duration=10, path="_temp_audio", rate = 16000, channels = 1):
        self.duration = duration
        self.path = path
        self.streaming = False
        self.rate = rate
        self.channels = channels # Mono audio
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if self.duration <= 0:
            self.streaming=True

    def ensure_mono_16k(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Convierte a mono y 16 kHz; devuelve float32 [-1,1]."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != self.rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.rate)
        # a float32 [-1,1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        # si viene en int16 rango grande, normaliza
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32767.0
        return audio

    def record_seconds(self):
        print(f"Grabando {self.duration}s… habla ahora")
        audio = sd.rec(int(self.duration*self.rate), samplerate=self.rate,
                       channels=self.channels, dtype='int16')
        sd.wait()
        print("Se acabó la grabación")
        return audio.flatten()

    def save_audio(self, audio, sr=16000, path=None):
        if path is None:
            path = self.path
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f"rec_{uuid.uuid4().hex}.wav")
        audio = self.ensure_mono_16k(audio, sr)  # downsample a 16k mono
        sf.write(str(path), audio, self.rate, subtype="PCM_16")
        print(f"Saved in {str(path)}")
        return path

    def resample(self, audio_array, sr=16000):
        if sr != self.rate:
            input_arr = librosa.resample(audio_array, orig_sr=sr, target_sr=self.rate)
        else:
            input_arr = audio_array
        return input_arr
