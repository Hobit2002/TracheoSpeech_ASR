import torch
import torch.nn as nn
import whisper, time
from whisper_config import DEVICE

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, tokenizer):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers=6, bidirectional=True, batch_first=True)
        self.ln = nn.LayerNorm(embedding_dim)
        self.tokenizer = tokenizer

    def forward(self, x, index_of_interest):
        # Embed input tokens
        x = self.embedding(x)
        # Pass through BiLSTM
        output, _ = self.lstm(x)
        batch_size, seq_len, hidden_dim_times_2 = output.shape
        hidden_dim = hidden_dim_times_2 // 2
        output_at_index = output[torch.arange(output.size(0)), index_of_interest, :]
        output_reshaped = output_at_index.view(batch_size, 2, hidden_dim)
        x = output_reshaped.mean(dim=1) 
        # Concatenate forward and backward LSTM hidden states
        # Output layer
        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.embedding.weight.to(x.dtype), 0, 1)
        ).float()
        return logits
    
    def get_soft_targets(self, inputs, labels, predictions):
        # Step 1: Construct `predict_on`
        # `predict_on` is `labels` with the first column of `inputs` added at the beginning
        predict_on = torch.cat(
            [inputs[:, :1], labels], dim=1
        )  # Concatenate first column of inputs to the start of labels
        predict_on = predict_on.clone()
        predict_on[predict_on == -100] = self.tokenizer.eot  # Replace -100 entries with tokenizer.eot

        # Step 2: Create T copies of `predict_on` with modifications
        N, T = inputs.shape
        predict_on_copies = []
        predict_here = self.tokenizer.encode("<|uz|>", allowed_special={"<|uz|>"})[0]
        for t in range(T):
            # Copy and replace (t+1)-th column with the result of tokenizer.encode("<|uz|>")
            modified_copy = predict_on.clone()
            modified_copy[:, t + 1] = predict_here
            predict_on_copies.append(modified_copy)

        # Step 3: Call `self(copy, idx)` for each copy and collect predictions
        soft_targets = []
        for t, copy in enumerate(predict_on_copies):
            # Get predictions for the current copy
            predictions_for_t = self(copy, t+1)  # Assumes `self(copy, idx)` gives predictions of shape N x O
            soft_targets.append(predictions_for_t)

        # Step 4: Concatenate predictions into tensor `soft_targets` of shape N x T x O
        soft_targets = torch.stack(soft_targets, dim=1)  # Shape: N x T x O

        # Step 5: Handle the special case where labels[n, t] = -100
        soft_targets[labels == -100] = predictions[labels == -100]
        """for n in range(N):
            for t in range(T):
                if labels[n, t] == -100: soft_targets[n, t, :] = predictions[n, t]"""

        return soft_targets.detach().clone()
