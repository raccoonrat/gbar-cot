import torch
import torch.nn as nn
import numpy as np

class SimplifiedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, dim_feedforward, num_layers):
        super(SimplifiedTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn positionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        src = self.embedding(src) * np.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.fc(output)
        return output

class positionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(positionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# For demonstration, we'll use a small version of the model
vocab_size = 10000  # Adjust based on your tokenizer
d_model = 128
nhead = 4
dim_feedforward = 256
num_layers = 2

model = SimplifiedTransformer(vocab_size, d_model, nhead, dim_feedforward, num_layers)
