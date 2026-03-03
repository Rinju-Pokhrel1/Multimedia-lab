import sounddevice as sd  # For audio recording
import matplotlib.pyplot as plt  # For plotting waveforms
from scipy.io.wavfile import write  # For saving audio as .wav file
import numpy as np  # Missing import added

# Set parameters
sample_rate = 44100  # Standard sampling rate (CD quality)
duration = 5  # Duration to record audio in seconds

# Record audio
print("Recording... Speak now!")
audio_data = sd.rec(
    int(sample_rate * duration),  # Record audio for given duration
    samplerate=sample_rate,  # Use defined sampling rate
    channels=1,  # Mono channel
    dtype=np.int16  # Data type for audio samples
)
sd.wait()  # Wait until recording is finished
print("Recording finished!")

# Save recorded audio
file_name = "recorded_audio.wav"  # Output file name
write(file_name, sample_rate, audio_data)  # Save audio to a .wav file
print(f"Audio saved as {file_name}")

# Convert to numpy array
audio_data = audio_data.flatten()  # Flatten 2D array to 1D

# Time axis for plotting
time_axis = np.linspace(0, duration, num=len(audio_data))  # Time values for x-axis

# Plot the original sound wave
plt.figure(figsize=(12, 4))
plt.plot(time_axis, audio_data, color='blue')
plt.title("Original Sound Wave (Time Domain)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# Show Sampling Process
downsample_rate = 4000  # Simulated lower sample rate
downsampled_audio = audio_data[::sample_rate // downsample_rate]  # Downsample the signal
time_downsampled = np.linspace(0, duration, num=len(downsampled_audio))

plt.figure(figsize=(12, 4))
plt.plot(time_axis, audio_data, color='gray', alpha=0.5, label="Original Signal")
plt.scatter(time_downsampled, downsampled_audio, color='red', label="Sampled Points", zorder=3)
plt.title("Sampling Process (Digital Representation)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Show Quantization Process
quantization_levels = 16  # Number of quantization levels
max_amp = np.max(np.abs(audio_data))  # Max amplitude of the original signal

quantized_audio = np.round(
    audio_data / max_amp * (quantization_levels // 2)
) * (max_amp / (quantization_levels // 2))

plt.figure(figsize=(12, 4))
plt.plot(time_axis, audio_data, color='gray', alpha=0.5, label="Original Signal")
plt.scatter(time_axis, quantized_audio, color='green', s=5, label="Quantized Signal")
plt.title("Quantization Process")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()