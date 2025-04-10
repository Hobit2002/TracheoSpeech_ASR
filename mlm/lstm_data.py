import torch, random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import whisper
from torch.nn.utils.rnn import pad_sequence


# External function to tokenize text (you'll need to define this elsewhere)
options = whisper.DecodingOptions(language="cs", without_timestamps=True)
tokenizer = whisper.tokenizer.get_tokenizer(True, language="cs", task=options.task)


def tokenize_whisper(text):
    text = [*tokenizer.sot_sequence_including_notimestamps] + tokenizer.encode(text) + [tokenizer.eot]
    return text

# --- 1. Dataset Definition ---
class TextDataset(Dataset):
    def __init__(self, file_paths, segment_length=15):
        self.segment_length = segment_length
        self.data = []
        # Account for bible verse
        remove_numeric_prefix = lambda s: ' '.join([sub.lstrip('0123456789') if sub.lstrip('0123456789') and sub.lstrip('0123456789')[0] != sub[0] else sub for sub in s.split()])
        for path in file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                lines = [line for line in lines 
                                  if sum([1 if any([ch.isalpha() for ch in word]) else 0 for word in line.split()]) >= 5]
                text = "\n".join(lines)
                # Split into segments
                words = text.split()

                segments = [remove_numeric_prefix(' '.join(words[i:i+segment_length]) )
                            for i in range(0, len(words), segment_length)]
                
                # Keep only lines with 5 or more words containing letters
                self.data.extend(segments)

        # Tokenize using the external tokenized function
        self.tokenized_data = [tokenize_whisper(segment[:512]) for segment in self.data]
        print(f"Collected {len(self.tokenized_data)} segments")

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        tokens = self.tokenized_data[idx]
        # Randomly choose a token as the target
        index_of_interest = random.randint(0, min(len(tokens) - 1, 512))

        # Input: All except the selected token; Target: Selected token
        context_tokens = tokens[:index_of_interest] + tokenizer.encode("<|uz|>", allowed_special = {"<|uz|>"})+ tokens[index_of_interest + 1:]
        target_token = tokens[index_of_interest]

        return torch.tensor(context_tokens, dtype=torch.long), torch.tensor(target_token, dtype=torch.long), index_of_interest

def collate_fn(batch):
    """
    Collate function for padding sequences in a batch with the padding token ID (50257).
    """
    padding_token = 50257

    # Separate the batch into context tokens, target tokens, and indices
    context_tokens, target_tokens, indices = zip(*batch)

    # Pad context tokens with the padding token from the right
    padded_context_tokens = pad_sequence(context_tokens, batch_first=True, padding_value=padding_token)

    # Convert target tokens to a tensor
    target_tokens = torch.stack(target_tokens)

    # Return padded context tokens, target tokens, and indices
    return padded_context_tokens, target_tokens, torch.tensor(indices)
