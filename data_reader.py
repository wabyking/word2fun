import numpy as np
import torch
from torch.utils.data import Dataset

np.random.seed(12345)

from nltk.tokenize import word_tokenize

def tokenlize(text):
    text =text.lower()
    # print(" ".join(word_tokenize(text))) word_tokenize(text)
    return text.split()


class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, input_text, min_count):

        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.input_text = input_text
        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.input_text, encoding="utf8"):
            # line =line.lower()
            line = tokenlize(line)
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegatives(self, target, size):  # TODO check equality with target
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response


# -----------------------------------------------------------------------------------------------------------------

class Word2vecDataset(Dataset):
    def __init__(self, data, input_text = None, window_size= 5):
        self.data = data
        self.window_size = window_size
        if input_text is None:
            self.input_file = open(data.input_text, encoding="utf8")
        else:
            self.input_file = open(input_text, encoding="utf8")


    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()
            # line =line.lower()
            if len(line) > 1:
                words = tokenlize(line)

                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]

                    boundary = np.random.randint(1, self.window_size)
                    return [(u, v, self.data.getNegatives(v, 5)) for i, u in enumerate(word_ids) for j, v in
                            enumerate(word_ids[max(i - boundary, 0):i + boundary]) if u != v]

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)


class TimestampledWord2vecDataset(Dataset):
    def __init__(self, data, input_text = None, window_size=5, time_scale = 1):
        self.data = data
        self.window_size = window_size
        # self.input_file = open(data.inputFileName, encoding="utf8")
        self.time_scale = time_scale
        if input_text is None:
            self.input_file = open(data.input_text, encoding="utf8")
        else:
            self.input_file = open(input_text, encoding="utf8")

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        while True:
            line = self.input_file.readline()
            # line =line.lower()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line = tokenlize(line)
                time, words = int(words[0]),words[1:]
                time = time / self.time_scale
                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]

                    boundary = np.random.randint(1, self.window_size)
                    
                    return [(u, v, self.data.getNegatives(v, 5),time) for i, u in enumerate(word_ids) for j, v in
                            enumerate(word_ids[max(i - boundary, 0):i + boundary]) if u != v]
                    # for i, u in enumerate(word_ids):
                    #     for j, v in enumerate(word_ids[max(i - boundary, 0):i + boundary]):
                    #         if u != v:
                    #             return (u, v, self.data.getNegatives(v, 5),time)




    @staticmethod
    def collate(batches):

        all_u = [u for batch in batches for u, _, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _,_ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v,_ in batch if len(batch) > 0]
        time = [time for batch in batches for _, _, _ ,time in batch if len(batch) > 0]
        
        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v),torch.LongTensor(time)

