import os
import matplotlib.pyplot as plt
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

            bit_depth_value = info.get('bits_per_sample', None)
            if bit_depth_value is not None:
                bit_depth = int(bit_depth_value)
            else:
                bit_depth = 16

            if 'alac' in codec or 'alac' in format_name:
                bit_depth = bit_depth if bit_depth else 16

            return bit_depth, str(bit_depth) in bit_rates

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None, False

    def calculate_reverb(self, file_path):
        y, sr = self.load_audio(file_path)
        if y is None or sr is None:
            return None

        return np.sqrt(np.mean(y**2))

    def detect_copy_paste(self, source_files, target_files, min_wave_length=100):
        results = []

        for source_file in source_files:
            y_source, sr_source = self.load_audio(source_file)

            if y_source is None:
                results.append((source_file, False, "Source audio file could not be loaded."))
                continue

            for target_file in target_files:
                y_target, sr_target = self.load_audio(target_file)

                if y_target is None:
                    results.append((source_file, False, f"Target audio file {target_file} could not be loaded."))
                    continue

                # Calculate the spectrograms
                source_spectrogram = np.abs(librosa.stft(y_source))
                target_spectrogram = np.abs(librosa.stft(y_target))

                # Function to find the start index based on wave length
                def get_wave_start_index(spectrogram, wave_length_threshold):
                    energy = np.sum(spectrogram, axis=0)
                    above_threshold = np.where(energy > wave_length_threshold)[0]

                    # Return the start index of the first significant wave
                    return above_threshold[0] if above_threshold.size > 0 else 0

                # Get the start index based on the minimum wave length for both source and target
                source_start_index = get_wave_start_index(source_spectrogram, min_wave_length)
                target_start_index = get_wave_start_index(target_spectrogram, min_wave_length)

                # Create the source pattern starting from the determined start index
                source_pattern = source_spectrogram[:, source_start_index:]

                # Pattern length
                pattern_length = source_pattern.shape[1]

                # Check for matches in the target spectrogram starting from the target start index
                match_found = False
                for start in range(target_start_index, target_spectrogram.shape[1] - pattern_length + 1):
                    segment = target_spectrogram[:, start:start + pattern_length]
                    if segment.shape == source_pattern.shape and np.array_equal(segment, source_pattern):
                        target_file_name = os.path.basename(target_file)
                        results.append((source_file, True,
                                        f"Copy/paste detected in target file: {target_file_name}"))
                        match_found = True

                if not match_found:
                    results.append((source_file, False, f"No copy/paste detected in target file: {target_file}"))

        return results

    def plot_spectrogram(self, file_path):
        try:
            # Load the audio file
            y, sr = self.load_audio(file_path)
            if y is None or sr is None:
                return "Error: Audio file could not be loaded."

            # Compute the spectrogram (STFT)
            spectrogram = np.abs(librosa.stft(y))

            # Create a figure and axis for the plot
            plt.figure(figsize=(10, 6))

            # Display the spectrogram
            librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max),
                                     sr=sr, x_axis='time', y_axis='linear', cmap='magma')

            # Add title and labels
            plt.title(f"Spectrogram of {os.path.basename(file_path)}")
            plt.colorbar(format='%+2.0f dB')
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (s)')

            # Show the plot
            plt.tight_layout()
            plt.show()

            # Return success message
            return "Spectrogram displayed successfully."

        except FileNotFoundError as fnf_error:
            return f"Error: {fnf_error}"

        except ValueError as val_error:
            return f"Error: {val_error}"

        except Exception as e:
            return f"Error: An unexpected error occurred - {e}"
