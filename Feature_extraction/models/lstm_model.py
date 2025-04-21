
import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, vocab_size=4, embedding_dim=64, hidden_size=512, num_layers=2, num_classes=4, bidirectional=False, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        direction = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        features = hn[-1]
        features = self.dropout(features)
        out = self.fc(features)
        return out, features
