import librosa
import numpy as np
from pydub.utils import mediainfo


class AudioFileChecker:
    def __init__(self, supported_formats, target_rate):
        self.supported_formats = supported_formats
        self.target_rate = target_rate

    def check_format(self, file_path):
        try:
            file_extension = file_path.split('.')[-1].lower()
            return file_extension in self.supported_formats, file_extension
        except Exception as e:
            return False, None

    def check_sampling_rate(self, file_path, target_rates):
        try:
            _, sr = librosa.load(file_path, sr=None)
            rate_ok = sr in target_rates
            return rate_ok, sr
        except Exception as e:
            print(f"Error: {e}")
            return False, None

    def calculate_rms(self, file_path, noise_threshold_db=50):
        try:
            y, sr = librosa.load(file_path)
            rms = np.mean(librosa.feature.rms(y=y))
            rms_db = librosa.amplitude_to_db([rms])[0]
            return rms_db, rms_db < noise_threshold_db
        except Exception as e:
            print(f"Error: {e} ")
            return None, False

    def calculate_snr(self, file_path):
        print(1)
        try:
            y, sr = librosa.load(file_path)

            epsilon = 1e-10
            signal_power = np.mean(librosa.feature.rms(y=y)) + epsilon
            print(f"Signal Power: {signal_power}")

            noise_power = np.percentile(librosa.feature.rms(y=y), 10) + epsilon
            print(f"Noise Power: {noise_power}")
            if noise_power == 0 or signal_power == 0:
                return None, False

            snr_db = 20 * np.log10(signal_power / noise_power)

            return snr_db, snr_db >= 15
        except Exception as e:
            print(f"Error: {e} ")
            return None, False

    def detect_clipping(self, file_path, clipping_threshold=0.99):
        try:
            y, sr = librosa.load(file_path)
            clipping_points = np.where(np.abs(y) > clipping_threshold)[0]
            return len(clipping_points) > 0, clipping_points
        except Exception as e:
            return False, []

    def check_channel_mode(self, file_path):
        try:
            y, sr = librosa.load(file_path, mono=False)
            num_channels = y.shape[0] if len(y.shape) > 1 else 1
            return 'stereo' if num_channels > 1 else 'mono', num_channels
        except Exception as e:
            print(f"Error: {e} ")
            return None, 0

    def check_bit_depth(self, file_path):
        try:
            info = mediainfo(file_path)
            bit_depth = int(info.get('bits_per_sample', 0))
            return bit_depth, bit_depth in [8, 16, 24, 32]
        except Exception as e:
            return None, False

    def calculate_reverb(self, file_path):

        try:
            y, sr = librosa.load(file_path, sr=None, mono=True)
            return np.sqrt(np.mean(y**2))
        except Exception as e:
            return None
