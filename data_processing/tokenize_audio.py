from pydub import AudioSegment
import numpy as np
import matplotlib.patches as patches
import librosa
import librosa.display
import matplotlib.pyplot as plt
from io import BytesIO

def load_audio(file_name, filter = True):
    """Loads an MP3 file and returns it as an AudioSegment."""
    try:
        audio = AudioSegment.from_file(file_name)
        if filter: audio = audio.high_pass_filter(1000)
        return audio
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def get_amplitude_array(audio_segment):
    """Converts an AudioSegment to a numpy array of amplitudes."""
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        # Stereo audio: take the average of the two channels
        samples = samples.reshape((-1, 2)).mean(axis=1)
    return samples

def save_segments(segments, prefix="segment"):
    """Saves the extracted segments as separate audio files."""
    for idx, segment in enumerate(segments):
        segment.export(f"{prefix}_{idx}.mp3", format="mp3")
        print(f"Saved segment {idx} as {prefix}_{idx}.mp3")

def plot_mel_spectrogram(audio_segment,duration):
    # Convert pydub AudioSegment to numpy array
    audio_bytes = BytesIO()
    audio_segment.export(audio_bytes, format="wav")
    audio_bytes.seek(0)

    # Load the audio with librosa
    y, sr = librosa.load(audio_bytes, sr=None)

    # Compute the Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # Convert the power spectrogram (amplitude squared) to decibel (dB) units
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot the Mel spectrogram
    plt.figure(figsize=(10, 4))
    ax = plt.subplot()
    ax.imshow(mel_spec_db)
    #librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    #plt.colorbar(format='%+2.0f dB')
    ax.set_title('Mel Spectrogram')

    # Add patches
    # Add a rectangle patch to the plot
    temporal_bins = mel_spec_db.shape[1]
    offset = int((temporal_bins/duration) * 3000)
    print(f"{offset/3} correspond to 1 second")
    rect_patch = patches.Rectangle(
        (offset, 1),  # Lower-left corner (x, y)
        temporal_bins - 2 * offset,  # Width of the patch (duration in time)
        126,  # Height of the patch (frequency range)
        linewidth=2, edgecolor="red", facecolor='none'
    )
    ax.add_patch(rect_patch)

    # Show the plot
    plt.tight_layout()
    plt.show()
