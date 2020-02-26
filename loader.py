import random
import torch
from torch.nn.utils import rnn
from torch.utils.data import Dataset


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

    def __len__(self):
        assert(len(self.word2idx) == len(self.idx2word))
        return len(self.word2idx)


class CorpusDense:
    def __init__(self, device):
        self.device = device
        self.dictionary = Dictionary()
        #
        self._train_data = self._tokenize_data(
            'data/pku_train/pku_no_space.txt')
        self._train_label = self._tokenize_label(
            'data/pku_train/pku_label.txt')
        assert(self._train_data.shape == self._train_label.shape)
        #
        self._test_data = self._tokenize_data('data/pku_test/pku_no_space.txt')
        self._test_label = self._tokenize_label('data/pku_test/pku_label.txt')
        assert(self._test_data.shape == self._test_label.shape)

    def _tokenize_data(self, path):
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = list(line)
                for w in words:
                    self.dictionary.add_word(w)

        # Tokenize file content
        ids = []
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = list(line)
                words = [self.dictionary.word2idx[w] for w in words]
                ids.extend(words)
        return torch.LongTensor(ids)

    def _tokenize_label(self, path):
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = list(line)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        segs = []
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = list(line.replace('\n', '1'))
                words = [int(x) for x in words]
                segs.extend(words)
        return torch.LongTensor(segs)

    def train_size(self):
        assert(len(self._train_data) == len(self._train_label))
        return len(self._train_data)

    def test_size(self):
        assert(len(self._test_data) == len(self._test_label))
        return len(self._test_data)

    def vocab_size(self):
        return len(self.dictionary)

    def segment_num(self):
        return 2

    def get_info(self):
        return 'train_size: {} test_size: {} vocab_size: {} segment_num: {}'.format(self.train_size(), self.test_size(), self.vocab_size(), self.segment_num())
    
    def get_train_seq(self, seq_length):
        while True:
            assert(len(self._train_data) == len(self._train_label))
            start = random.randrange(len(self._train_data) - seq_length)
            while self._train_label[start].item() != 1:
                start += 1
            start += 1
            end = min(len(self._train_label) - 1, start + seq_length - 1)
            while self._train_label[end].item() != 1:
                end -= 1
            if end - start < seq_length // 2:
                continue
            data = self._train_data[start:end + 1]
            label = self._train_label[start:end + 1]
            return data.view(-1, 1).to(self.device), label.view(-1, 1).to(self.device)

    def get_train_batch(self, seq_length, batch_size):
        data, labels = [], []
        for _ in range(batch_size):
            start = random.randrange(len(self._train_data) - seq_length)
            data.append(self._train_data[start:start + seq_length])
            labels.append(self._train_label[start:start + seq_length])
        data = torch.cat(data).view([batch_size, -1]).t().contiguous()
        labels = torch.cat(labels).view([batch_size, -1]).t().contiguous()
        return data.to(self.device), labels.to(self.device)

    def get_train_packet(self, seq_length, batch_size):
        data, labels = [], []
        while True:
            assert(len(self._train_data) == len(self._train_label))
            start = random.randrange(len(self._train_data) - seq_length)
            while self._train_label[start].item() != 1:
                start += 1
            start += 1
            end = min(len(self._train_label), start + seq_length - 1)
            while self._train_label[end].item() != 1:
                end -= 1
            if end - start < seq_length // 2:
                continue
            data.append(self._train_data[start:end + 1])
            labels.append(self._train_label[start:end + 1])
            if len(data) == batch_size:
                break
        data = rnn.pack_sequence(sorted(data, key=lambda item: len(item), reverse=True))
        labels = rnn.pack_sequence(sorted(labels, key=lambda item: len(item), reverse=True))
        return data.to(self.device), labels.to(self.device)

    def iter_test_batch(self, seq_length):
        for start in range(0, len(self._test_data), seq_length):
            data = self._test_data[start:start + seq_length]
            labels = self._test_label[start:start + seq_length]
            yield data.view([-1, 1]).to(self.device), labels.view([-1, 1]).to(self.device)


class CJKData(Dataset):
    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


def test():
    device = torch.device('cpu')
    corpus = CorpusDense(device)
    corpus.train_size()
    corpus.vocab_size()
    corpus.segment_num()
    for _ in range(16):
        data, labels = corpus.get_train_batch(seq_length=40, batch_size=16)
        assert(data.shape == labels.shape)
        assert(data.shape == torch.Size([40, 16]))
    test_seq_length = 0
    for data, labels in corpus.iter_test_batch(seq_length=40):
        assert(data.shape == labels.shape)
        assert(data.shape <= torch.Size([40, 1]))
        test_seq_length += len(data)
    assert(test_seq_length == corpus.test_size())


if __name__ == "__main__":
    test()
