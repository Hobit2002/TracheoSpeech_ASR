import whisper, torch, librosa
import numpy as np
from pathlib import Path
import time, random
import torchaudio, csv
import torchaudio.transforms as at
import pickle, os
from whisper_config import *
import pandas as pd
from tqdm import tqdm
from augmentations import insert_silence

loaded_audios = {}


def load_wave(wave_path, segment_start, segment_end, sample_rate:int=16000) -> torch.Tensor:
    global loaded_audios
    if not wave_path in loaded_audios.keys():
        waveform, sr = torchaudio.load(wave_path, normalize=True)
        if sample_rate != sr: waveform = at.Resample(sr, sample_rate)(waveform)
        loaded_audios = {}
        loaded_audios[wave_path] = (waveform, sample_rate)
    else: waveform, sample_rate = loaded_audios[wave_path]
    segment_of_interest = waveform[:, int(segment_start * sample_rate/1000):int(segment_end * sample_rate/1000)]

    return segment_of_interest


### CREATE THE DATASET OBJECT
class JasmiSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.debug = False

    def __len__(self):
        return len(self.audio_info_list)

    def __getitem__(self, id):
        extend_condition = lambda filename:int("".join(ch for ch in str(filename)[:-4] if ch.isnumeric())) < 24
        audio_path, temporal_coordinates, text = self.audio_info_list[id]

        # audio
        pickle_path = os.path.join(os.getcwd(), f"data/pickled/{len(self)}_{id}.pickle")
        if not os.path.isfile(pickle_path):
    
            # Merge the segments if they originated from reading sessions
            if extend_condition(audio_path):
                look_ahead = 2
                while temporal_coordinates[1] - temporal_coordinates[0] < 5000:
                    try_id = ((-1) ** (look_ahead % 2)) * look_ahead // 2 
                    try_path, try_coords, try_text = self.audio_info_list[id + try_id]
                    # Don't merge segments from different sessions
                    if try_path != audio_path:
                        look_ahead += 1
                        continue
                    # Otherwise merge the segments
                    if try_id < 0:
                        while len(try_text.split()) and try_text.split()[-1] == text.split()[0]:
                            try_text = " ".join(try_text.split()[:-1])
                        text = f"{try_text} {text}"
                        temporal_coordinates[0] = try_coords[0]
                    else:
                        while len(text.split()) and text.split()[-1] == try_text.split()[0]:
                            text = " ".join(text.split()[:-1])
                        text = f"{text} {try_text}"
                        temporal_coordinates[1] = try_coords[1]
                    # Increase the lookahead
                    look_ahead += 1

            segment_start, segment_end = temporal_coordinates
            text = text.replace("\"","").replace("'","")

            audio = load_wave(audio_path, segment_start, segment_end, sample_rate=self.sample_rate)
            audio = whisper.pad_or_trim(audio.flatten())
            mel = whisper.log_mel_spectrogram(audio)

            text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
            labels = text[1:] + [self.tokenizer.eot]
            to_return = {
                "input_ids": mel,
                "labels": labels,
                "dec_input_ids": text
            }
            with open(pickle_path, 'wb') as store: pickle.dump(to_return, store, protocol=pickle.HIGHEST_PROTOCOL)
        else: 
            to_return = pickle.load(open(pickle_path, "rb"))

        return to_return
    
### DEFINE THE DATA COLLATOR
class WhisperDataCollatorWhithPadding:
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, features):
        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)
        

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids #self.vocalize_model(input_ids)

        return batch
    
    def pad_and_stack(self, tensors, pad_value=0.0):
        """
        Pads a list of 2D tensors (timesteps, vocab_size) along the timesteps dimension 
        so they all have the same length, then stacks them into a single tensor.

        Args:
            tensors (list of torch.Tensor): List of 2D tensors of shape (timesteps, vocab_size).
            pad_value (float, optional): The value used for padding. Default is 0.0.

        Returns:
            torch.Tensor: A stacked tensor of shape (batch_size, max_timesteps, vocab_size).
        """
        # Find the max timesteps length
        max_timesteps = max(tensor.shape[0] for tensor in tensors)

        # Pad each tensor to max_timesteps
        padded_tensors = [
            torch.cat([tensor, torch.full((max_timesteps - tensor.shape[0], tensor.shape[1]), pad_value, device=tensor.device)])
            if tensor.shape[0] < max_timesteps else tensor
            for tensor in tensors
        ]

        # Stack them together
        return torch.stack(padded_tensors)
    
    def pad_with_eot(self, tensors):
        """
        Pads a list of 2D tensors (timesteps, vocab_size) along the timesteps dimension.
        Padding rows contain 1 at the position corresponding to whisper.tokenizer.eot.

        Args:
            tensors (list of torch.Tensor): List of 2D tensors (timesteps, vocab_size).
            tokenizer (whisper.tokenizer): Whisper tokenizer to get the EOT token index.

        Returns:
            torch.Tensor: A stacked tensor of shape (batch_size, max_timesteps, vocab_size).
        """
        # Find the max timesteps length
        max_timesteps = max(tensor.shape[0] for tensor in tensors)

        # Process each tensor
        padded_tensors = []
        for tensor in tensors:
            if tensor.shape[0] < max_timesteps:
                # Find the EOT token index (argmax of last timestep logits)
                eot_token_index = torch.argmax(tensor[-1], dim=-1).item()
                
                # Create padding tensor filled with zeros
                pad_tensor = torch.zeros((max_timesteps - tensor.shape[0], tensor.shape[1]), device=tensor.device)
                
                # Set the EOT token position to 1 in each row of padding
                pad_tensor[:, eot_token_index] = 1.0
                
                # Concatenate the original tensor with the padding
                padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
            else:
                padded_tensor = tensor

            padded_tensors.append(padded_tensor)

        # Stack them into a single tensor
        return torch.stack(padded_tensors)

### LOAD AUDIO AND ANNOTATIONS
def load_audio_and_annotations():
    # Annotations 
    annotations_path = Path(ANNOTATIONS_PATH)

    ### CREATE LIST OF PAIRED FILES
    def get_audio_file_list(transcripts_path_list, used_splits, text_max_length=120, audio_max_sample_length=480000, sample_rate=16000):
        audio_transcript_pair_list = []

        with open(transcripts_path_list, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                _,transcription,session_id,start_time,end_time,split = row
                if split in used_splits:

                    # Check text length and audio segment length
                    if len(transcription) > text_max_length or (end_time - start_time) * sample_rate/1000 > audio_max_sample_length:
                        if len(transcription) > text_max_length: print(f"Skipping line due to transcription being too long {annotation_path}: {transcription} has {len(transcription)} chars ")
                        else: print(f"Skipping line due to audio being too long: ({end_time}, {start_time}) - {(end_time - start_time) * sample_rate/1000} frames")
                        continue

                    audio_transcript_pair_list.append((f"{session_id}.mp3", [start_time, end_time], transcription))
        return audio_transcript_pair_list

    train_audio_transcript_pair_list = get_audio_file_list(annotations_path, ["train","val"], TEXT_MAX_LENGTH, AUDIO_MAX_LENGTH, SAMPLE_RATE)
    eval_audio_transcript_pair_list = get_audio_file_list(annotations_path, ["test"], TEXT_MAX_LENGTH, AUDIO_MAX_LENGTH, SAMPLE_RATE)
    print("TRAIN AUDIO DATASET NUM: ", len(train_audio_transcript_pair_list))
    print("EVAL AUDIO DATASET NUM: ", len(eval_audio_transcript_pair_list))
    return train_audio_transcript_pair_list, eval_audio_transcript_pair_list

### DATASET FOR NON-PERIODIC DATA
class CommonVoiceDataset(torch.utils.data.Dataset):
    def __init__(self, audio_dir, tsv_name, tokenizer, sample_rate=16000):
        """
        Args:
            tsv_path (str): Path to the .tsv file containing metadata.
            audio_dir (str): Path to the directory containing audio files.
            sample_rate (int): Target sample rate for audio files.
        """
        self.data = pd.read_csv(os.path.join(audio_dir, tsv_name), sep="\t")
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the row in the .tsv file.
        
        Returns:
            tuple: (spectrogram, transcription)
        """
        # Fetch row data
        row = self.data.iloc[idx]
        audio_path = f"{self.audio_dir}/{row['path']}"
        text = row['sentence']  # The transcription
        
        # Load audio
        waveform, orig_sample_rate = torchaudio.load(audio_path, normalize=True)
        
        # Pre-process the audio
        waveform = torch.tensor(insert_silence(librosa.effects.time_stretch(waveform.detach().numpy()[0], rate = 1/4), orig_sample_rate, np.random.randint(7,12), 0.125)).unsqueeze(0).float()

        # Resample if needed
        if orig_sample_rate != self.sample_rate: waveform = at.Resample(orig_sample_rate, self.sample_rate)(waveform)
        
        # Convert waveform to mel spectrogram
        waveform = whisper.pad_or_trim(waveform.flatten())
        mel = whisper.log_mel_spectrogram(waveform)
        mel = mel

        # Encode text
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]

        return {
                "input_ids": mel,
                "labels": labels,
                "dec_input_ids": text
            }