import torch
import torch.nn as nn


class BiLSTMSegement(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_segment, em_dropout=0.2, lstm_dropout=0.5):
        nn.Module.__init__(self)
        #
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        #
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.Dropout(em_dropout))
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            bias=True, dropout=lstm_dropout, bidirectional=True)
        self.scoring = nn.Sequential(nn.Linear(2 * self.hidden_size, num_segment),
                                     nn.Softmax(dim=2))

    def init_hidden(self, batch_size, device):
        # The axes semantics are (direct * num_layers, minibatch_size, hidden_size)
        # return axes are (hidden_state, cell_state)
        return (torch.zeros(2 * self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(2 * self.num_layers, batch_size, self.hidden_size).to(device))

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        score = self.scoring(lstm_out)
        return score, hidden


def test():
    seq_length, batch_size, num_segment = 48, 16, 2
    data = torch.randint(3, size=[seq_length, batch_size], dtype=torch.long)
    model = BiLSTMSegement(vocab_size=32, embedding_dim=8,
                           hidden_size=64, num_layers=2, num_segment=num_segment)
    hidden = model.init_hidden(batch_size)
    score, hidden = model(data, hidden)
    assert(score.shape == torch.Size([seq_length, batch_size, num_segment]))
    assert(torch.sum(score, dim=2).allclose(torch.ones(1)))


if __name__ == "__main__":
    test()
