import os
import os
import subprocess

def extract_audio_only(input_folder, audio_output_folder):
    os.makedirs(audio_output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.mov', '.webm', '.mp4')):
            input_path = os.path.join(input_folder, filename)
            new_filename = filename.replace(" ", "").replace("-", "_")
            audio_output_path = os.path.join(audio_output_folder, f"{os.path.splitext(new_filename)[0]}.mp3")

            print(f"Extracting audio from {new_filename}...")
            subprocess.run([
                'ffmpeg',
                '-i', input_path,
                '-vn',  # Skip video
                '-acodec', 'libmp3lame',
                '-q:a', '2',
                audio_output_path
            ], check=True)

    print("All audio files have been extracted successfully.")


if __name__ == "__main__":
    # Input folder containing .mov files and output folders for video and audio
    cwd = os.getcwd()
    input_folder = os.path.join(cwd, 'data/raw')
    audio_output_folder = os.path.join(cwd, 'data/audio')

    # Split the videos and audios
    extract_audio_only(input_folder, audio_output_folder)
