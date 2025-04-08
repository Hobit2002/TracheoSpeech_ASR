from pydub.playback import play
from data_processing.tokenize_audio import plot_mel_spectrogram, get_amplitude_array
from data_processing.clean_annotations import process_csv
from IPython.display import clear_output, display
import csv, os, sys
from tqdm import tqdm
from data_processing.premerge_api import *
# Auto-completion input libraries
import ipywidgets as widgets
import time  

def tokenize_text(file_name):
    file = open(file_name).read()
    return file.split()


class LabelingProcedure():

    def __init__(self, text_tokens, audio_tokens, audio, loud_per_syllab = 48500.24, autoplay = True) -> None:
        self.text_tokens = text_tokens
        self.audio_tokens = audio_tokens
        self.audio = audio
        self.comitted_pairs = []
        self.tt = self.at = 0
        self.loud_per_syllab = loud_per_syllab
        self.autoplay = autoplay

    def add_labels(self, fix_conversations = False, reverse_order = True):
        continue_loop = True
        skip_play = False
        action_list = []
        input_f = self.auto_completion_input if fix_conversations else input
        if fix_conversations == reverse_order:
            self.text_tokens = self.text_tokens[::-1]
            self.audio_tokens = self.audio_tokens[::-1]
        while continue_loop:
            if (self.tt == len(self.text_tokens) and self.at == len(self.audio_tokens) ): break
            self.tt, self.at = min([self.tt, len(self.text_tokens) - 1]), min([self.at, len(self.audio_tokens) - 1])
            clear_output()
            # Show text and play
            print("Previous text:",self.text_tokens[self.tt - 1])
            print("Text:",self.text_tokens[self.tt])
            print("Next text:",self.text_tokens[self.tt + 1] if self.tt + 1 < len(self.text_tokens) - 1 else "<END>")
            segment_start, segment_end = self.audio_tokens[self.at][0], self.audio_tokens[self.at][1]
            plot_mel_spectrogram(self.audio[segment_start - 3000 : segment_end+3000], segment_end - segment_start + 6000)
            if not skip_play and self.autoplay: 
                try: play(self.audio[segment_start:segment_end])
                except KeyboardInterrupt: pass
            else: skip_play = False
            # Take and resolve an action
            while True:
                action = input_f("Choose an action:") if not len(action_list) else action_list.pop(0)
                # Support for giving a list of multiple actions to be executed sequentially
                if ";" in action: 
                    action_list = action.split(";")
                    action = action_list.pop(0)
                # Pair actions
                if action == "C": # Commit
                    self.comitted_pairs.append((self.text_tokens[self.tt], self.audio_tokens[self.at].copy()))
                    self.at, self.tt = self.at + 1, self.tt + 1
                    print("Commited")
                    break
                elif action == "U": # Uncommit
                    current_pair = (self.text_tokens[self.tt], self.audio_tokens[self.at])
                    if current_pair in self.comitted_pairs:
                        del self.comitted_pairs[self.comitted_pairs.index(current_pair)]
                elif action == "END": # End the labelling procedure
                    continue_loop = False
                    break
                # Only audio actions
                elif action[0] == "E": # Extend audio from the END
                    duration = 800 if len(action) == 1 else -1000 if action[1] == "-" and len(action) == 2 else float(action[1:])
                    self.audio_tokens[self.at][1] += duration
                    if not len(action_list): # Show updated spectrogram and play the audio
                        clear_output()
                        print("Text:",self.text_tokens[self.tt])
                        segment_start, segment_end = self.audio_tokens[self.at][0], self.audio_tokens[self.at][1]
                        plot_mel_spectrogram(self.audio[segment_start - 3000 : segment_end+3000], segment_end - segment_start + 6000) 
                        if self.autoplay: play(self.audio[segment_start:segment_end])
                    # Propagate the update, if in fix_conversations mode:
                    if fix_conversations:
                        for audio_tok in self.audio_tokens[self.at + 1:]: audio_tok[1] += duration
                elif action[0] == "B": # Extend audio from the BEGINING
                    duration = 1000 if len(action) == 1 else -1000 if action[1] == "-" and len(action) == 2 else float(action[1:])
                    self.audio_tokens[self.at][0] -= duration
                    if not len(action_list): # Show updated spectrogram and play the audio
                        clear_output()
                        print("Text:",self.text_tokens[self.tt])
                        segment_start, segment_end = self.audio_tokens[self.at][0], self.audio_tokens[self.at][1]
                        plot_mel_spectrogram(self.audio[segment_start - 3000 : segment_end+3000], segment_end - segment_start + 6000) 
                        if self.autoplay:  play(self.audio[segment_start:segment_end])
                    # Propagate the update, if in fix_conversations mode:
                    if fix_conversations:
                        for audio_tok in self.audio_tokens[self.at + 1:]: audio_tok[0] -= duration
                elif action[0] == "M": # Replay te audio
                    if "s" in action: # Different offset in the beggining and the end
                        extension_b, extension_e = action[1:].replace("s"," s ").split("s")
                        play(self.audio[self.audio_tokens[self.at][0] - float(extension_b):self.audio_tokens[self.at][1] + float(extension_e)])
                    else: 
                        extension = 0 if len(action) == 1 else float(action[1:]) # The same or no offset
                        play(self.audio[self.audio_tokens[self.at][0] - extension:self.audio_tokens[self.at][1] + extension])
                # Only textual actions
                elif action == "W": # Rewrite the textual token
                    new_str = input("Correct caption: ")
                    self.text_tokens[self.tt] = new_str
                    print(50 * "\r","Updated text:",self.text_tokens[self.tt])
                elif action == "N": # Merge with next
                    self.tt += 1
                    self.text_tokens[self.tt] = self.text_tokens[self.tt - 1] + " " + self.text_tokens[self.tt]
                    print(50 * "\r","Updated text:",self.text_tokens[self.tt])
                elif action[0] == "L": # SpLit the string
                    split_point = int(action[1:])
                    if split_point > 0: # Cut string from the beggining
                        new_text = " ".join(self.text_tokens[self.tt].split()[:split_point])
                        self.text_tokens[self.tt] = " ".join(self.text_tokens[self.tt].split()[split_point:])
                        self.text_tokens.insert(self.tt, new_text)
                        self.tt += 1
                    elif split_point < 0: # Cut string from the end
                        new_text = " ".join(self.text_tokens[self.tt].split()[split_point:])
                        self.text_tokens[self.tt] = " ".join(self.text_tokens[self.tt].split()[split_point:])
                        self.text_tokens.insert(self.tt + 1, new_text)
                    break
                # Actions both universal and specific
                elif "S" in action: # Skip
                    hop_length = 1 if action[action.index("S") + 1:] == "" else int(action[2:])
                    if action[0] != "T": self.at += hop_length
                    if action[0] != "A": 
                        skip_play = True
                        self.tt += hop_length
                    break
                elif "R"  in action: # Return to the previous
                    hop_length = 1 if action[action.index("R") + 1:] == "" else int(action[2:])
                    if action[0] != "T": self.at -= hop_length
                    if action[0] != "A":
                        skip_play = True
                        self.tt -= hop_length
                    break
                elif action[-1] == "P": # Merge with the previous token
                    if action[0] != "T": self.audio_tokens[self.at][0] = self.audio_tokens[self.at - 1][0]
                    if action[0] != "A": 
                        self.text_tokens[self.tt] = self.text_tokens[self.tt - 1] + " " + self.text_tokens[self.tt]
                        print(50 * "\r","Updated text:",self.text_tokens[self.tt])
                else:
                    print("Invalid action: ", action)
        return self.comitted_pairs

    def auto_completion_input(self, prompt_text, default_value="C", timeout=2.6):
        # Display the prompt in a notebook-friendly way
        print(f"{prompt_text} - defaulting to {default_value}")
        try:
            time.sleep(timeout)
            return default_value
        except KeyboardInterrupt:
            play(self.audio[self.audio_tokens[self.at][0]:self.audio_tokens[self.at][1]])
            return input(prompt_text)


def save_labels(path, commited_pairs):
    try:
        # Open the CSV file in write mode
        temp_path = path.replace(".csv","_tmp.csv")
        with open(temp_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write each tuple to the CSV
            for tr, times in commited_pairs:
                if len(times) == 2:  # Ensure that the second item is a list of exactly 2 elements
                    writer.writerow([tr] + times)
                else:
                    print(f"Skipping row with incorrect format: {tr}, {times}")
        process_csv(temp_path, path)
        os.remove(temp_path)            
        print(f"CSV written successfully to {path}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")


