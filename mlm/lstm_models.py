import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers=6, bidirectional=True, batch_first=True)
        self.ln = nn.LayerNorm(embedding_dim)

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