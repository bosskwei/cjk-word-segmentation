import time
import math
from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import metrics
import loader
import lstm


def main():
    corpus = loader.CorpusDense()
    model = lstm.BiLSTMSegement(vocab_size=corpus.vocab_size(),
                                embedding_dim=256,
                                hidden_size=256,
                                num_layers=2,
                                num_segment=corpus.segment_num())
    critical = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=1e-1,
    #                       momentum=0.9, weight_decay=1e-3)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    def accuracy(pred, target_labels):
        assert(len(pred.shape) == 3)
        pred_labels = pred.argmax(dim=2)
        assert(len(pred_labels.shape) == 2)
        assert(pred_labels.shape == target_labels.shape)
        return metrics.f1_score(y_true=target_labels.view(-1).tolist(),
                                y_pred=pred_labels.view(-1).tolist(),
                                labels=[x for x in range(corpus.segment_num())], average=None, zero_division=0)

    def train(seq_length, batch_size):
        model.train()
        model.zero_grad()
        hidden = model.init_hidden(batch_size)
        data, labels = corpus.get_train_batch(seq_length, batch_size)
        pred, _ = model(data, hidden)
        f1_scores = accuracy(pred, labels)
        loss = critical(pred.view([-1, corpus.segment_num()]), labels.view(-1))
        loss.backward()
        optimizer.step()
        return loss.item(), f1_scores

    def evaluate(seq_length):
        with torch.no_grad():
            model.eval()
            before = time.time()
            hidden = model.init_hidden(1)
            count = 0
            f1_scores = [0.0 for _ in range(corpus.segment_num())]
            for data, labels in corpus.iter_test_batch(seq_length):
                count += 1
                pred, hidden = model(data, hidden)
                f1_scores = [e + x for e,
                             x in zip(f1_scores, accuracy(pred, labels))]
            f1_scores = [x / count for x in f1_scores]
            after = time.time()
        return f1_scores, after - before

    # with open('model-1.pt', 'rb') as f:
    #     model = torch.load(f)

    for epoch in range(0, 80):
        for seq_length, batch_size in product([240, 120, 80, 60, 40, 20, 10], [64, 32, 16]):
            num_batch = corpus.train_size() // (seq_length * batch_size)
            interval = num_batch // 20
            loss_summed, f1_summed = 0.0, [
                0.0 for _ in range(corpus.segment_num())]
            print('--------------------')
            print('[epoch {}] seq-batch: {}-{}, corpus {}'
                  .format(epoch, seq_length, batch_size, corpus.get_info()))
            for i in range(num_batch):
                loss, f1_score = train(seq_length, batch_size)
                loss_summed += loss
                f1_summed = [e + x for e, x in zip(f1_summed, f1_score)]
                if i > 0 and i % interval == 0:
                    f1_summed = [x / interval for x in f1_summed]
                    print('[epoch {}] batch: {}/{}, loss: {:.3f}, loss_exp: {}, f1: {}'
                          .format(epoch, i, num_batch, loss_summed, math.exp(loss_summed), f1_summed))
                    loss_summed, f1_summed = 0.0, [
                        0.0 for _ in range(corpus.segment_num())]
            # end batch
            for seq_length in [160, 80, 40]:
                f1_scores, time_diff = evaluate(seq_length)
                print('[epoch {}] seq_length: {}, evaluate: {}, time_diff: {:.2f}'.format(
                    epoch, seq_length, f1_scores, time_diff))
        # end one epoch
        with open('model-{}-[{:.02f}, {:.02f}].pt'
                  .format(epoch, f1_scores[0], f1_scores[1]), 'wb') as f:
            torch.save(model, f)


if __name__ == "__main__":
    # TODO
    # 3. load space file
    # 4. refine args.num_segments
    # 5. train_batch, train_packed
    # 6. save and load dictionary
    # 7. load multifile
    main()
