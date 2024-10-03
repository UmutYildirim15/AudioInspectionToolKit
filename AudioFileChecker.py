import librosa
import numpy as np
from pydub.utils import mediainfo

class AudioFileChecker:
    def __init__(self, supported_formats, target_rate):
        self.supported_formats = supported_formats
        self.target_rate = target_rate
        self.audio_cache = {}  # Cache for loaded audio files to avoid redundant loads

    def load_audio(self, file_path):
        if file_path not in self.audio_cache:
            try:
                y, sr = librosa.load(file_path, sr=None)
                self.audio_cache[file_path] = (y, sr)
            except Exception as e:
                print(f"Error loading audio: {e}")
                return None, None
        return self.audio_cache[file_path]

    def check_format(self, file_path):
        try:
            file_extension = file_path.split('.')[-1].lower()
            return file_extension in self.supported_formats, file_extension
        except Exception as e:
            return False, None

    def check_sampling_rate(self, file_path, target_rates):
        sample_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 176400, 192000, 384000]
        y, sr = self.load_audio(file_path)
        if y is None or sr is None:
            return False, None

        # Compute Short-Time Fourier Transform (STFT) to analyze frequency content
        stft = np.abs(librosa.stft(y))

        # Compute the effective sampling rate
        freqs = librosa.fft_frequencies(sr=sr)
        max_freq_index = np.argmax(stft, axis=0)
        max_freq = freqs[max_freq_index].max()
        effective_sr = max_freq * 2

        # Find closest sample rate
        closest_sample_rate = min(sample_rates, key=lambda x: abs(x - effective_sr))
        rate_ok = closest_sample_rate in target_rates
        return rate_ok, closest_sample_rate

    def calculate_rms(self, file_path, noise_threshold_db=50):
        y, sr = self.load_audio(file_path)
        if y is None or sr is None:
            return None, False
        rms = np.mean(librosa.feature.rms(y=y))
        rms_db = librosa.amplitude_to_db([rms])[0]
        return rms_db, rms_db < noise_threshold_db

    def calculate_snr(self, file_path):
        y, sr = self.load_audio(file_path)
        if y is None or sr is None:
            return None, False

        epsilon = 1e-10
        signal_power = np.mean(librosa.feature.rms(y=y)) + epsilon
        noise_power = np.percentile(librosa.feature.rms(y=y), 10) + epsilon

        if noise_power == 0 or signal_power == 0:
            return None, False

        snr_db = 20 * np.log10(signal_power / noise_power)
        return snr_db, snr_db >= 15

    def detect_clipping(self, file_path, clipping_threshold=0.99):
        y, sr = self.load_audio(file_path)
        if y is None or sr is None:
            return False, []

        clipping_points = np.where(np.abs(y) > clipping_threshold)[0]
        return len(clipping_points) > 0, clipping_points

    def check_channel_mode(self, file_path):
        y, sr = self.load_audio(file_path)
        if y is None or sr is None:
            return None, 0

        num_channels = y.shape[0] if len(y.shape) > 1 else 1
        return 'stereo' if num_channels > 1 else 'mono', num_channels

    def check_bit_depth(self, file_path, bit_rates):
        try:
            info = mediainfo(file_path)

            codec = info.get('codec_name', '').lower()
            format_name = info.get('format', '').lower()

            if 'alac' in codec or 'alac' in format_name:
                bit_depth = int(info.get('bits_per_sample', 16))
            else:
                bit_depth = int(info.get('bits_per_sample', 16)) if int(info.get('bits_per_sample', 0)) != 0 else 16

            return bit_depth, str(bit_depth) in bit_rates

        except Exception as e:
            return None, False

    def calculate_reverb(self, file_path):
        y, sr = self.load_audio(file_path)
        if y is None or sr is None:
            return None

        return np.sqrt(np.mean(y**2))
