import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr,spearmanr
# from model import SkipGramModel, TimestampedSkipGramModel
from model import SkipGramModel, TimestampedSkipGramModel
from data_reader import DataReader, Word2vecDataset, TimestampledWord2vecDataset
import json

import os
import argparse
import pickle
# from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sys import platform
import numpy as np
import heapq
import scipy
import pandas as pd
import pickle

if platform != "darwin":
    plt.switch_backend('agg')

# word_sin word_cos word_mixed word_linear word_mixed_fixed
parser = argparse.ArgumentParser(description='parameter information')
parser.add_argument('--model_path', dest='time_type', type=str, default="coha",
                    help='model path with log.txt, vocab.txt and pytorch.bin')
args = parser.parse_args()


def get_score(a, b):
    from sklearn.metrics import mutual_info_score
    from scipy.stats import entropy
    score = mutual_info_score(a, b)
    _, p1 = np.unique(a, return_counts=True)
    _, p2 = np.unique(b, return_counts=True)
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()
    return score / (entropy(p1) + entropy(p2)) * 2


def get_score1(a, b):
    from sklearn.metrics import f1_score, fbeta_score
    x = [i == j for i in a for j in a]
    y = [i == j for i in b for j in b]
    # print(x)
    # print(y)
    return fbeta_score(x, y, beta=5)


def keep_top(arr, k=3):
    smallest = heapq.nlargest(k, arr)[-1]  # find the top 3 and use the smallest as cut off
    arr[arr < smallest] = 0  # replace anything lower than the cut off with 0
    return arr


def read_embeddings_from_file(file_name):
    embedding_dict = dict()
    with open(file_name, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                vocab_size, emb_dimension = [int(item) for item in line.split()]
                # embeddings= np.zeros([vocab_size,emb_dimension])
            else:
                tokens = line.split()
                word, vector = tokens[0], [float(num_str) for num_str in tokens[1:]]
                embedding_dict[word] = vector
    return embedding_dict


def read_alignment(filename="eval/yao/testset_2(1).csv"):
    results = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            w1, w2 = line.split(",")
            w1, y1 = w1.split("-")
            w2, y2 = w2.split("-")
            results.append((w1, y1, w2, y2))
    return pd.DataFrame(results, columns=["w1", "y1", "w2", "y2"])

class Timer():
    # time = Timer(start_year=1990)
    # print(time.get_index(1992))
    #
    #
    # time = Timer(save_path="coha.txt.raw.token.decade-output")
    # print(time.get_index(1992))

    mapping = {
        "newsit": 2007,
        "repubblica": 1984,
        "nyt_yao.txt": 1986,
        "nyt.txt": 1987,
        "yao_tiny.txt": 1990,
        "coha": 1810,
        "coca": 1990,
    }

    def __init__(self, start_year=None, corpus=None, scale=None):
        if start_year is not None:
            self.start_year = start_year
        elif corpus is not None:
            for name, year in Timer.mapping.items():
                # print(name,year)
                if name in corpus:
                    print("bingo, found the first year of {} for {}".format(corpus, year))
                    self.start_year = year
        else:
            print("error for input")
        if scale is None:
            if corpus is not None and "decade" in corpus:
                self.scale = 10
            else:
                self.scale = 1
        else:
            self.scale = scale
        print("time scale : {}".format(scale))
    def get_index(self, year):

        return (year - self.start_year) // self.scale

    def get_year(self,index):
        return index * self.scale + self.start_year

    def get_index_in_batch(self, years):
        return [self.get_index(year) for year in years]


class BaseDynamicWordEmbedding:
    def __init__(self, corpus ="nyt_yao_tiny.txt"):

        self.timer = Timer(corpus=corpus, scale=1 if "decade" not in corpus else 10)

    def encode_time(self, year):
        if type(year) == str:
            year = int(year)
        if year > 100:
            year = self.timer.get_index(year)
        return year

    def _get_temporal_embedding(self, word, time): # in single
        raise NotImplementedError

    def get_temporal_embedding(self, words, times): # in batch

        if type(words) == list and type(times) == list:

            embeddings = np.array([self._get_temporal_embedding(w, t) for w, t in zip(words, times)])
            # from collections import Counter
            # print(Counter( [ type(e) for e in embeddings]))

            return embeddings
        else:

            return self._get_temporal_embedding(words, times)

    def get_vocab(self):
        raise NotImplementedError

    def load_dict(self, vocab_file="pmi_vocab.txt", reversed=False):
        word2id = dict()
        index = -1
        with open(vocab_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                tokens = line.split()
                if len(tokens) ==2:
                    word, index =tokens
                else:
                    word = tokens[0]
                    index = index+1
                if not reversed:
                    word2id[word] = int(index)
                else:
                    word2id[int(index)] = word
        return word2id

class DynamicWordEmbeddingWithCompass(BaseDynamicWordEmbedding):
    def __init__(self, model_path = "model" , corpus = "yao_tiny.txt" ):
        super().__init__(corpus=corpus)

        COMPASS_FILE = "static.model"# "compass.model",
        from gensim.models.word2vec import Word2Vec

        self.embeddings = {   int(filename.split(".")[0]) :Word2Vec.load(os.path.join(model_path,filename))
                              for filename in os.listdir(model_path) if  filename != COMPASS_FILE and filename.endswith(".model")}
        self.compass = Word2Vec.load(os.path.join(model_path,COMPASS_FILE))
        self.vector_size = self.compass.vector_size
        # for e in self.embeddings.values():
        #     print(len(e.wv.vocab.keys()))
        # print( len(self.get_vocab()) )
        # different model has different vocab
    def _get_temporal_embedding(self, word, time):
        # exit()
        if type(time) == str:
            time = int(time)
        if time < 100:
            time = self.timer.get_year(time)
        if time not in  self.embeddings:
            time = self.timer.get_index(time)
        if word in self.embeddings[time].wv:
            return self.embeddings[time].wv[word]
        else:
            return np.random.randn(self.vector_size)

    def get_vocab(self):
        return self.compass.wv.vocab.keys()



class Compass(BaseDynamicWordEmbedding):
    def __init__(self, model_path = "model" , corpus = "yao_tiny.txt" ):
        super().__init__(corpus=corpus)

        COMPASS_FILE = "compass.model"
        from gensim.models.word2vec import Word2Vec
        self.compass = Word2Vec.load(os.path.join(model_path,COMPASS_FILE))
        self.vector_size = self.compass.vector_size
        # for e in self.embeddings.values():
        #     print(len(e.wv.vocab.keys()))
        # print( len(self.get_vocab()) )
        # different model has different vocab
    def _get_temporal_embedding(self, word, time):
        if type(time) == str:
            time = int(time)
        if time < 100:
            time = self.timer.get_year(time)
        if word in self.compass.wv:
            return self.compass.wv[word]
        else:
            return np.random.randn(self.vector_size)

    def get_vocab(self):
        return self.compass.wv.vocab.keys()

class DynamicWordEmbedding(BaseDynamicWordEmbedding):
    def __init__(self, timer=None, path="results2", corpus="nyt_yao_tiny.txt.norm",filename = "L10T50G100A1ngU_iter4.p",vocab_file = None):
        self.path = path

        super().__init__(corpus=corpus)
        filename = os.path.join(self.path, filename)
        self.embeddings = pickle.load(open(filename, "rb"))
        print("have {} years with shape {}".format(len(self.embeddings), self.embeddings[0].shape))
        if vocab_file is None:
            vocab_file = corpus + ".vocab"

        self.word2id = self.load_dict(vocab_file)
        self.id2word = self.load_dict(vocab_file, reversed=True)

    def _get_temporal_embedding(self, word, time):

        if type(word) != int:
            word = self.word2id[word]
        time = self.encode_time(time)
        # print(time)
        embedding = self.embeddings[time]


        return embedding[word]


    def get_vocab(self):
        return self.word2id.keys()




class Word2Fun(BaseDynamicWordEmbedding):

    def __init__(self,path="coha", epoch=None, do_prune_vocab=False,corpus="nyt_yao_tiny.txt.norm",step =None):
        super().__init__(corpus=corpus)
        self.id2word = self.load_vocab(os.path.join(path, "vocab.txt"), do_prune_vocab=do_prune_vocab)
        self.word2id = {value: int(key) for key, value in self.id2word.items()}

        if epoch is not None and step is None:  # load indivusual epoch, or the last epoch
            path = os.path.join(path, str(epoch))
            filename = "pytorch.bin"
        elif epoch is not None and step is not None:
            filename = "pytorch_{}_{}.bin".format(epoch,step)
        else:
            filename = "pytorch.bin"
        print(path,filename)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            self.model = torch.load(os.path.join(path, filename), map_location=torch.device('cpu')).to(
                self.device)
        else:
            self.model = torch.load(os.path.join(path, filename)).to(self.device)
        self.model = self.model.module if hasattr(self.model, 'module') else self.model
        self.model.eval()

    def get_vocab(self):
        return self.word2id.keys()

    def load_vocab(self, vocob_file, base_vocab_file="./vocab.txt", do_prune_vocab=False):
        id2word = dict()
        with open(vocob_file, encoding="utf-8") as f:
            for line in f:
                _id, word = line.strip().split()
                id2word[int(_id)] = word
        if os.path.exists(base_vocab_file) and do_prune_vocab:
            base_vocab = set()
            with open(base_vocab_file, encoding="utf-8") as f:
                for line in f:
                    _id, word = line.strip().split()
                    base_vocab.add(word)
            raw_length = len(id2word)
            id2word = {key: value for key, value in id2word.items() if value in base_vocab}
            print("pruning vocab from {} to {}".format(raw_length, len(id2word)))
        return id2word

    def transform(self,word_tensor, time_tensor):

        word_tensor = torch.LongTensor(word_tensor).to(self.device)
        time_tensor = torch.LongTensor([int(y) for y in time_tensor]).to(self.device)
        return word_tensor, time_tensor



    def _get_temporal_embedding(self, word=None, year=0):
        if word in self.word2id:
            word = [self.word2id[word]]
            year = [self.encode_time(year)]
            word_tensor, time_tensor = self.transform(word, year)
            return self.model.get_temporal_embedding(word_tensor, time_tensor).squeeze()
        else:
            return None



    def get_temporal_embedding(self, words=None, years=0, return_unknow_words=False):

        # unknow_words = [i for i, word in enumerate(words) if word not in self.word2id]
        word_tensor, time_tensor = [], []
        known_index = []
        years =[ self.encode_time(year) for year in years]
        for index, (word, year) in enumerate(zip(words, years)):
            if word in self.word2id:
                word_tensor.append(self.word2id[word])
                time_tensor.append(year)
                known_index.append(True)
            else:
                # print("unknown word" + word)
                known_index.append(False)

        word_tensor, time_tensor = self.transform(word_tensor, time_tensor)

        embeddings = self.model.get_temporal_embedding(word_tensor, time_tensor)  # .cpu().data.numpy()

        if return_unknow_words:
            return embeddings, np.array(known_index)
        return embeddings

    def get_sim_between_year(self, target, words=None, years=[i for i in range(1940, 2020, 1)], word2id=None,
                             name="nuclear"):
        name += "-" + target + "_".join(words)
        sims = []
        words.append(target)

        for year in years:
            embeddings = self.get_tempory_embedding(words, year)

            sim = cosine_similarity(embeddings[-1][np.newaxis, :], embeddings[:-1]).squeeze()
            # print(sim.shape)
            sims.append(sim)
        sims = np.array(sims)
        plt.figure(figsize=(10, 10))
        for i in range(len(sims[0])):
            plt.plot(years, sims[:, i], label=words[i])
        plt.legend(loc='upper left')
        if platform == "darwin_none":
            plt.show()
        else:
            plt.savefig("{}.pdf".format(name), bbox_inches="tight", pad_inches=0)
            plt.close()

    def get_embedding_by_year(self, word, year):
        if word not in self.word2id:
            return None
        else:
            word_tensor = torch.LongTensor([self.word2id[word]]).to(self.device)
            time_tensor = torch.LongTensor([year]).to(self.device)
            embedding = self.model.get_temporal_embedding(word_tensor, time_tensor)
            return embedding


# dwe = DynamicWordEmbedding(Timer(save_path="nyt_yao_tiny.txt.norm"))
# print(dwe.get_temporal_embedding(1,0))
# print(dwe.get_temporal_embedding("the",1990))
# # exit()


class Evaluator:
    def __init__(self, model, timer=None, debug_mode = False):
        self.model = model
        self.timer= timer
        self.debug_mode = debug_mode


        # self.skip_gram_model,self.id2word = load_old(path,time_type)

    def get_top_k_most_similar_words_in_specific_time(self, words, year, k=3):

        if year > 30:
            year = self.timer.get_index(year)
        all_vectors = self.get_tempory_embedding(self.word2id.keys(), year=year)

        # embeddings_vectors = np.array( [vector for word,vector in embeddings])
        # all_words = [word for word,vector in embeddings]
        not_found_words = [word for word in words if word not in self.word2id]
        if len(not_found_words) > 0:
            print("do not find {}".format(" ".join(not_found_words)))
        words_index = [self.word2id[word] for word in words if word in self.word2id]  ## index of in-vocab words

        selected_vectors = np.array([all_vectors[index] for index in words_index])

        a = np.dot(selected_vectors, all_vectors.T)  # /np.norm()
        # a = cosine_similarity(selected_vectors,embeddings_vectors)

        top_k = a.argsort()[:, -1 * k:]  # [::-1]
        # top_k = np.partition(a, -3)

        words_str = [" ".join([self.id2word[word] for word in top_k_per_word[::-1]]) for top_k_per_word in top_k]
        return words_str

    def word_change_rate(self, words, years=30):
        vectors = []
        for year in range(years):
            word2id = {value: key for key, value in self.id2word.items()}
            embeddings_vectors = self.get_tempory_embedding(self.embedding_dict.keys(), word2id=word2id, year=year)

            # embeddings_vectors = np.array( [vector for word,vector in embeddings])
            # all_words = [word for word,vector in embeddings]

            words_index = [word2id[word] for word in words]
            # print(words_index)

            selected_vectors = np.array([embeddings_vectors[word] for word in words_index])
            vectors.append(selected_vectors)

        for j in range(len(words)):
            change_rates = []
            for year in range(years):
                if year == 0:
                    cur_vector = vectors[year][j]
                else:

                    # change_rate = np.dot(cur_vector,vectors[year][j])
                    change_rate = scipy.spatial.distance.cosine(cur_vector, vectors[year][j])
                    cur_vector = vectors[year][j]
                    change_rates.append(change_rate)
            print(words[j], np.mean(np.array(change_rates)))
            print(change_rates)

        return

    def plot_words_in_many_years(self, words=None, years=[i for i in range(1977, 2020, 1)], word2id=None, name="image"):
        if words is None:
            words = ["president", "reagan", "trump", "biden", "obama", "bush", "carter", "clinton", "ford", "nixon"]
            # words = ["weapon" , "nuclear",   "energy"]
        if word2id is None:
            word2id = {value: key for key, value in self.id2word.items()}
        vectors = []
        names = []
        for year in years:
            names.extend(["{}-{}".format(word, year) for word in words])
            embeddings = self.get_tempory_embedding(words, year, word2id)
            vectors.extend(embeddings)
        embed = TSNE(n_components=2).fit_transform(vectors)
        # print(embed.shape)

        plt.figure(figsize=(12, 12))
        # from adjustText import adjust_text
        texts = []
        for i, point in enumerate(embed):
            plt.scatter(point[0], point[1], label=names[i])
            texts.append(plt.text(point[0], point[1], names[i], size=7))
        # plt.plot(embed[:,0],embed[:,1],names)

        # adjust_text(texts)
        # plt.legend()
        if platform == "win32":
            plt.show()
        else:
            plt.savefig("president-{}.pdf".format(name), bbox_inches="tight", pad_inches=0)
            plt.close()
        # plt.show()

    def check_ssd_driver(self):
        from data.ssd import Helper
        helper = Helper("data/grade.txt", tims_scale=10 )

        from scipy.spatial.distance import cosine  # cosine distance

        helper.adapt(self.model.get_vocab())
        words = helper.words

        time_stamped_embeddings = []

        for timespan in helper.timespans:
            all_embeddings = [self.model.get_temporal_embedding(words, [year ]* len(words)) for year in timespan]
            mean_embedding = np.mean(np.array(all_embeddings), 0)
            time_stamped_embeddings.append(mean_embedding)

        assert len(time_stamped_embeddings) == 2, "more timespans than two"
        scores = [cosine(time_stamped_embeddings[0][i], time_stamped_embeddings[1][i]) for i, word in enumerate(words)]

        return helper.evaluate(scores)

    def get_sim_words_diver(self, words, years, real_years, k=100, log_filename="sim_log.txt"):
        simwords = []
        for year in years:
            simwords.append(self.get_top_k_most_similar_words_in_specific_time(words=words, year=year, k=k))

        lines = []
        with open(log_filename, "w", encoding="utf-8") as f:
            for row in range(len(simwords[0])):
                line = [real_years[i] + " : " + simword[row] for i, simword in enumerate(simwords)]
                print(line)
                lines.extend(line)
            f.write("\n".join(lines))
        return

    def semantic_sim_driver(self, log_filename="yao_test1.txt"):  # time_mapping

        df = pd.read_csv("eval/yao/testset_1.csv")

        # df.real_year = df.year.apply(lambda x: int(self.timer.get_index(int(x))))
        # print(df.real_year.unique(),df.year.unique())
        df = df[df.word.isin(self.model.get_vocab()) ].reset_index()
        labels = set(df.label.unique())
        labels_mapping = {label: index for index, label in enumerate(labels)}
        df.label_id = df.label.apply(lambda x: labels_mapping[x])
        # print(df.label_id)

        embeddings = self.model.get_temporal_embedding(df.word.tolist(), df.year.tolist())

        from spherecluster import SphericalKMeans

        scores = []
        for n in [10, 15, 20]:
            skm = SphericalKMeans(n_clusters=n)
            skm.fit(embeddings)
            # print(skm.labels_.shape)
            # print(len(df.label_id[known_index]))
            # print(sum(known_index))
            score = get_score(skm.labels_, df.label_id)
            score1 = get_score1(skm.labels_, df.label_id)
            scores.append(score)
            scores.append(score1)

        # print(scores)

        with open(log_filename, "w", encoding="utf-8") as f:
            line = "\t&".join(["{0:.4f}".format(s) for s in scores]) + "\n"
            print(line)
            f.write(line)

        return None


    def alignment_quality_driver(self, log_filename="alignment_quality.log", do_normarlization = True):
        df1 = read_alignment("eval/yao/testset_2(1).csv")
        df2 = read_alignment("eval/yao/testset_2(2).csv")
        dfs = [df1, df2]
        if self.debug_mode:
            df3 = df1[df1.w2==df1.w1].reset_index()
            df4 = df1[df1.w2 != df1.w1].reset_index()
            # df5 = df2[df2.w2 == df2.w1].reset_index()
            # df6 = df2[df2.w2 != df2.w1].reset_index()

            dfs.extend( [df3,df4])
        with open(log_filename, "w", encoding="utf-8") as f:
            for df in dfs:
                # df.y1 = df.y1.apply(lambda x: int(self.timer.get_index(int(x))))
                # df.y2 = df.y2.apply(lambda x: int(self.timer.get_index(int(x))))
                length = len(df)
                df = df[df.w1.isin(self.model.get_vocab()) & df.w2.isin(self.model.get_vocab())].reset_index()

                print(" {} rows with valid ones counted {}".format(length, len(df)))

                # sources = df["w1"].unique()
                targets = [word for word in self.model.get_vocab()]
                targets_dict = {target: i for i, target in enumerate(targets)}
                id2word = {i: target for i, target in enumerate(targets)}
                timed_embeddings = dict()
                for year in df.y2.unique():

                    timed_embeddings.setdefault(year, self.model.get_temporal_embedding(targets, [year] * len(targets)) )
                    if do_normarlization:
                        norm =  np.linalg.norm(timed_embeddings[year], axis =-1 )
                        dim = len(timed_embeddings[year][0])

                        z = np.repeat(norm[:, np.newaxis], dim, axis=1)
                        timed_embeddings[year] = timed_embeddings[year] / z
                p1, mr, p3, p5, p10 = [], [], [], [], []

                for i, row in tqdm(df.iterrows()):

                    embedding = self.model._get_temporal_embedding(row.w1, row.y1)
                    if do_normarlization:
                        embedding = embedding / np.linalg.norm(embedding)
                    # timed_embeddings.setdefault(row.y2,self.get_embedding_in_a_year(targets, [row.y2]* len(targets), return_known_index=False))
                    candicates = timed_embeddings[row.y2]
                    # print(embedding.shape, candicates.shape)

                    ranking_scores = np.dot(embedding, candicates.transpose())
                    # print(np.max(ranking_scores))
                    ranks = np.argsort(ranking_scores)[::-1]
                    # print(ranks.shape)

                    target = targets_dict[row.w2]
                    # print(row.y1, row.y2, row.w1, row.w2,target)
                    # print(ranks)
                    first_index = -1
                    for index, rank in enumerate(ranks):
                        if rank == target:
                            first_index = index
                            break
                    assert first_index != -1, "wrong for calculating MRR"

                    p1.append(1 if first_index == 0 else 0)
                    p3.append(1 if first_index < 3 else 0)
                    p5.append(1 if first_index < 5 else 0)
                    p10.append(1 if first_index < 10 else 0)
                    mr.append(1 / (first_index + 1))
                    if self.debug_mode and i <2:
                        print([id2word[r] for r in ranks[:10]])
                        print(row.w1, row.y1, row.y2, row.w2,first_index)


                scores = [np.mean(s) for s in (mr, p1, p3, p5, p10)]
                # print(scores)
                # exit()

                line = "\t&".join(["{0:.4f}".format(s) for s in scores])
                print(line)
                f.write(line + "\n")


check_list = [("president", ["nixon", "ford", "carter", "reagan", "clinton", "bush", "obama", "trump", "biden"]),
              ("olympic",
               ["moscow", "los", "angeles", "seoul", "barcelona", "atlanta", "sydney", "athens", "beijing", "london",
                "rio", "tokyo"]),
              ("nuclear", ["technology", "threaten", "america", "russian", "cuba", "green", "energy", "china"]),
              ("nuclear", ["russian", "japan", "weapon", "energy", "ukrainian", "soviet"]),
              ("olympic", ["sydney", "athens", "beijing", "london", "rio", "tokyo"]),
              ("president", ["clinton", "bush", "obama", "trump", "biden"]),
              ]


def draw_figure():
    for output in ["coha.txt.raw.token-output/", "coca.txt.raw.token-output/", "arxiv.txt.raw.token-output/"]:
        if "coca" in output:
            years = [i - 1990 for i in range(1990, 2020, 1)]
        else:
            years = [i - 1810 for i in range(1810, 2020, 1)]
        for time_type in ["word_mixed_fixed", "word_cos"]:  # "word_cos",
            for epoch in range(1, 10, 1):
                args.iterations = epoch
                try:
                    checker = Word2VecChecker(path=output, time_type=time_type)
                    for target, checked_words in check_list:
                        # checker.plot_words_in_many_years(words=[target] + checked_words[-9:], years=years,
                        #                                  name="{}-{}".format(output.split(".")[0], time_type))
                        checker.get_sim_between_year(target, checked_words[-9:],
                                                     name="{}-{}-{}-".format(output.split(".")[0], time_type, epoch),
                                                     years=years)
                except Exception as e:
                    print(e)


timetypes = ["cos", "linear_shift", " mixed_shift", "sin_shift", "word_cos", "word_linear_shift", "word_mixed_fixed",
             "word_mixed_shift", "word_sin_shift",
             "cos_shift  mixed", "others_shift", "word2vec", "word_cos_shift", "word_mixed", "word_mixed_fixed_shift",
             "word_sin"]


def get_frequencies(model_path="nyt_yao.txt.train-output", timetypes=[], epoch=None, do_prune_vocab=True):
    for time_type in timetypes:  # "word_cos", , "word_cos"
        # for epoch in range(1, 10, 1):

        save_filename = "sim_word_{}_{}_{}\n".format(model_path, time_type, epoch)
        model_save_path = os.path.join(model_path, time_type)
        checker = Word2VecChecker(model_save_path, epoch=epoch, do_prune_vocab=do_prune_vocab)
        print(checker.model)
        # print(checker.model.time_encoder.para_embedding.weight)
        print(checker.model.time_encoder.para_embedding.weight.abs().mean())


def old_test():
    # yao_test(model_path=files[0], timetypes=["word_mixed_amplitude"], epoch=19)
    # files = ["nyt_yao_tiny.txt.norm.train-waby-300"]
    # yao_test(model_path=files[0], timetypes=["word_mixed_fixed"], epoch=12)
    files = ["nyt_yao_tiny.txt.norm-waby_amplitude-100", "nyt_yao_tiny.txt.norm.train-waby-300"]
    if not torch.cuda.is_available():
        files = ["models/{}".format(filename) for filename in files]

    yao_test(model_path=files[0], timetypes=["word_mixed_amplitude"], epoch=19)
    yao_test(model_path=files[1], timetypes=["word_mixed_fixed"], epoch=12)


words = ["apple", "amazon", "dna", "innovation", "data", "app", "twitter", "ranking", "quantum", "nuclear",
         "weapon", "president", "chairman", "soviet", "reagan", "trump", "biden", "obama", "olympic", "olympics",
         "china", "america", "ai", "artificial", "intelligence", "neural", "network", "language", "model", "wto",
         "media", "software", "computer", "car", "driving",
         "information", "retrieval"] + ["iphone", "mp3"] + ["guy", "gay", "program", "america", "television", "stock",
                                                            "european", "euro", "best-seller", "phone"] \
        + ["wi-fi", "browser", "android", "huawei", "cpu", "microsoft", "xbox", "pc"]


def case_studies_nyt():
    years = [i - 1990 for i in range(1990, 2017, 1)]
    real_years = [str(i) for i in range(1990, 2017, 1)]  # 1986
    files = ["nyt_yao_tiny.txt.norm-waby_amplitude-100", "nyt_yao_tiny.txt.norm.train-waby-300"]
    if not torch.cuda.is_available():
        files = ["models/{}".format(filename) for filename in files]
    checker = Word2VecChecker(os.path.join(files[0], "word_mixed_amplitude"), epoch=19)
    # checker = Word2VecChecker(os.path.join(files[1], "word_mixed_fixed"), epoch=12)
    checker.get_sim_words_diver(words, years, real_years)
    # checker = Word2VecChecker(os.path.join(files[0], "word_mixed_amplitude"), epoch=19)
    checker = Word2VecChecker(os.path.join(files[1], "word_mixed_fixed"), epoch=12)
    checker.get_sim_words_diver(words, years, real_years, log_filename="sim_log1.txt")


def case_studies():
    years = [(i - 1810) // 10 for i in range(1810, 2020, 10)]
    real_years = [str(i) + "s" for i in range(1810, 2020, 10)]  # 1986
    files = ["nyt_yao_tiny.txt.norm-waby_amplitude-100", "nyt_yao_tiny.txt.norm.train-waby-300"]
    if not torch.cuda.is_available():
        files = ["models/{}".format(filename) for filename in files]
    checker = Word2VecChecker(os.path.join(files[0], "word_mixed_amplitude"), epoch=19)
    # checker = Word2VecChecker(os.path.join(files[1], "word_mixed_fixed"), epoch=12)
    checker.get_sim_words_diver(words, years, real_years)
    # checker = Word2VecChecker(os.path.join(files[0], "word_mixed_amplitude"), epoch=19)
    checker = Word2VecChecker(os.path.join(files[1], "word_mixed_fixed"), epoch=12)
    checker.get_sim_words_diver(words, years, real_years, log_filename="sim_log1.txt")


def main():
    timetypes = ["word_mixed_fixed"]  # "word_cos",  "word_linear", "word_mixed","word_mixed_fixed","word_sin"
    files = ["nyt_yao_tiny.txt-20-nodecay-output", "nyt_yao_tiny.txt-20-100dim-output",
             "nyt_yao_tiny.txt-20-half-lr-output",
             "nyt_yao_tiny.txt-20-half-batchsize-output"]  # , "nyt_yao_tiny.txt-20-phase-output"
    files = ["nyt_yao_tiny.txt.norm-200-count-output",
             "nyt_yao_tiny.txt.norm-200output"]  # , "nyt_yao_tiny.txt-20-phase-output"
    # for file in files:
    #     yao_test(model_path=file, timetypes=["word_mixed_fixed"])
    # for file
    # yao_test("coha",timetypes=["word_mixed_fixed"] )
    # exit()

    for file in files:
        for epoch in range(5):
            # ssd_test("coha.txt.raw.token.train-decade-output",timetypes=timetypes,epoch=epoch)
            get_frequencies(model_path=file, timetypes=["word_mixed_amplitude"], epoch=epoch, do_prune_vocab=False)
    # yao_test(model_path="nyt_yao.txt.train-output", timetypes=["word_mixed_fixed"])

    # for epoch in range(5):
    #     yao_test(model_path="nyt_yao_tiny.txt.norm.train-output",timetypes=["word_mixed_fixed"], epoch=epoch)
    #  #, "word_mixed"
    # for epoch in range(5):
    #     yao_test(model_path="nyt_yao.txt.train-output",timetypes=["word_mixed_fixed"], epoch=epoch)
    #

    # exit()


# models/nyt_yao_tiny.txt.norm-waby_decay-10  13  0.538
# models/nyt_yao_tiny.txt.norm-waby_decay-1  16  0.55 +
# nyt_yao_tiny.txt.norm.train-waby-300 12 0.60
#  models/nyt_yao_tiny.txt.norm.train-waby1-300\word_mixed_fixed\9 0.5822
def main():
    # yao_test(model_path="models/nyt_yao_tiny.txt.norm-waby_decay-1",timetypes = ["word_mixed_fixed"] )
    # yao_test(model_path="models/nyt_yao_tiny.txt.norm-waby_decay-1",timetypes = ["word_mixed_fixed"] ,epoch = 12,do_prune_vocab =False)
    # yao_test(model_path="models/nyt_yao_tiny.txt.norm-waby_decay-1",timetypes = ["word_mixed_fixed"] ,epoch = 13,do_prune_vocab =True)
    # exit()
    # yao_test(model_path="models/nyt_yao_tiny.txt.norm-waby_amplitude-100_decay10",timetypes = ["word_mixed_amplitude"] ,do_prune_vocab =True)
    files = ["nyt_yao_tiny.txt.norm-waby_amplitude-100_decay100", "nyt_yao_tiny.txt.norm-waby_amplitude-100_decay10",
             "nyt_yao_tiny.txt.norm-waby_amplitude-100", "nyt_yao_tiny.txt.norm.train-waby_amplitude-100"]
    files = ["nyt_yao_tiny.txt.norm-word2fun"]
    # files = [  "nyt_yao_tiny.txt.norm.train-output20",
    # "nyt_yao_tiny.txt.norm.train-output20-50", "nyt_yao_tiny.txt.norm.train-waby-50","nyt_yao_tiny.txt.norm.train-waby-300","nyt_yao_tiny.txt.norm.train-waby-100","nyt_yao_tiny.txt.norm.train-waby-200"]
    # for file in files:
    #     # get_frequencies(model_path="models/{}".format(file),timetypes = ["word_mixed_amplitude"], do_prune_vocab =True)
    #     path = "models"  if not torch.cuda.is_available() else "."
    #     yao_test(model_path="{}/{}".format(path,file),timetypes = [ "word_linear", "word_mixed", "word_mixed_amplitude_shift", "word_mixed_fixed_no_amplitude", "word_mixed_fixed_shift", "word_mixed_shift", ], do_prune_vocab =True)#"time2vec_shift",
    #     continue
    files = ["nyt_yao_tiny.txt.norm-best", "nyt_yao_tiny.txt.norm-best2", "nyt_yao_tiny.txt.norm-best-10",
             "nyt_yao_tiny.txt.norm-best2-10"]
    # yao_test("coha", timetypes=["word_mixed"],   do_prune_vocab=True)
    # exit()
    files = [
        # "nyt_yao_tiny.txt.norm-best.0025-2",
        "nyt_yao_tiny.txt.norm-best.001-1",
        # "nyt_yao_tiny.txt.norm-best.0025-1",
        "nyt_yao_tiny.txt.norm-best.001-2"
    ]
    files = ["nyt_yao_tiny.txt.norm-best.0-100", "nyt_yao_tiny.txt.norm-best.1-100", "nyt_yao_tiny.txt.norm-best.2-100"]
    files = [
        # "nyt_yao_tiny.txt.norm-word2fun1_1000-50",
        # "nyt_yao_tiny.txt.norm-word2fun4_1000-50",
        # "nyt_yao_tiny.txt.norm-word2fun16_1000-50",
        # "nyt_yao_tiny.txt.norm-word2fun1_10000-50",
        # "nyt_yao_tiny.txt.norm-word2fun4_10000-50",
        # "nyt_yao_tiny.txt.norm-word2fun16_10000-50",
        "nyt_yao_tiny.txt.norm-word2fun1_1000",
        # "nyt_yao_tiny.txt.norm-word2fun4_1000",
        # "nyt_yao_tiny.txt.norm-word2fun16_1000",
        "nyt_yao_tiny.txt.norm-word2fun1_10000",
        # "nyt_yao_tiny.txt.norm-word2fun4_10000",
        # "nyt_yao_tiny.txt.norm-word2fun16_10000",
        # "nyt_yao_tiny.txt.norm-word2fun1_1000-200",
        # "nyt_yao_tiny.txt.norm-word2fun4_1000-200",
        # "nyt_yao_tiny.txt.norm-word2fun16_1000-200",
        # "nyt_yao_tiny.txt.norm-word2fun1_10000-200",
        # "nyt_yao_tiny.txt.norm-word2fun4_10000-200",
        # "nyt_yao_tiny.txt.norm-word2fun16_10000-200",
    ]
    files = ["nyt_yao_tiny.txt.norm-word2fun"]
    if not torch.cuda.is_available():
        files = ["models/{}".format(filename) for filename in files]

    # for filename in files:
    #     yao_test(filename, timetypes=["word_mixed_amplitude"],   do_prune_vocab=False)
    # print("***"*200)
    # exit()
    for filename in files:
        yao_test(filename, timetypes=["time2vec_shift"], epoch=4, do_prune_vocab=True)
        # continue
        exit()
        for epoch in range(5, 20):
            yao_test(filename, timetypes=["word_mixed_fixed"], epoch=epoch, do_prune_vocab=True)
        # # yao_test(model_path="models/nyt_yao_tiny.txt.norm-waby_decay-1",timetypes = ["word_mixed_fixed"] ,epoch = epoch,do_prune_vocab =False)
        # print("_______"*20)


def main(path="nyt_yao_tiny.txt.norm-word2fun1_10000-400-real", timetypes=None):
    if not torch.cuda.is_available():
        path = "models/{}".format(path)
    if timetypes is None:
        timetypes = [timetype for timetype in os.listdir(path)]

    yao_test(path, timetypes=timetypes, do_prune_vocab=True)

def load_yao_dynamic_embedding(input_path = "D:/codes/dynamicW2v/embeddings" ,output_file = "dyn.pkl"):
    import scipy.io as sio
    import os
    import pickle

    embeds = []
    path =  os.path.join (input_path, "embeddings")
    for i in range(27):   # the last embedding is fake, which is copied from ours
        filename = "embeddings_{}".format(i)
        filename = os.path.join(path, filename)
        print(filename)
        a = sio.loadmat(filename)["U"]
        print(a.shape)
        embeds.append(a)
    pickle.dump(embeds, open(output_file, "wb"))
    return output_file

# def main(path,timetypes):
#     # path ="nyt_yao_tiny.txt.norm"
#     # timetypes= None
#     ssd_test(path, timetypes=timetypes,use_yao=True)
#     # yao_test(path, timetypes=timetypes, do_prune_vocab=True,use_yao=True)


def alignment_quality():
    word2fun1 = Word2Fun(os.path.join("models","nyt_yao_tiny.txt.norm-waby_amplitude-100", "word_mixed_amplitude"),epoch=19)
    word2fun2 = Word2Fun(os.path.join("models","nyt_yao_tiny.txt.norm.train-waby-300", "word_mixed_fixed"),epoch=12)
    vocab_file = "./nyt.wordid,txt"
    dwe = DynamicWordEmbedding( corpus="yao_tiny.txt",path="results507-50-batchsize-512",filename="L10T50G50A1ngU_iter4.p",vocab_file=vocab_file)
    # vocab_file = "nyt.wordid,txt"  # loading  D:\codes\dynamicW2v/read_embedding.py
    dwe1 = DynamicWordEmbedding(path="./", corpus="nyt_yao_tiny.txt.norm", filename="dyn.pkl", vocab_file=vocab_file)
    compass = DynamicWordEmbeddingWithCompass(model_path="model-nyt")
    compass1 = DynamicWordEmbeddingWithCompass(model_path="model_nyt-1")

    static_compass = Compass(model_path="model-nyt")
    models = [compass1 , static_compass,word2fun1,word2fun2,dwe,dwe1,compass]
    # models = [static_compass]
    #
    for model in models:
        print(model)
        evaluator = Evaluator(model)
        evaluator.alignment_quality_driver()
        evaluator.check_ssd_driver()

def ssd():
    aw2v = DynamicWordEmbedding(corpus="coha.txt.norm", path="./",
                               filename="coha-aw2v.pkl")
    evaluator = Evaluator(aw2v)

    print(evaluator.check_ssd_driver())

def eval_yao_test(path = "nyt_yao_tiny.txt.norm-word2fun-42-1e-11" , time_type = "word_mixed_amplitude",debug_mode= False):
    for epoch in range(15,50):
        model = Word2Fun(os.path.join("models", path, time_type),
                         epoch=epoch, do_prune_vocab=False)
        evaluator = Evaluator(model, debug_mode=debug_mode)
        evaluator.alignment_quality_driver()
        for step in range(1,3):
            step = step*150-1
            model = Word2Fun(os.path.join("models", path, time_type),
                         epoch=epoch,step =step,do_prune_vocab =False)
            evaluator = Evaluator(model, debug_mode = debug_mode)
            evaluator.alignment_quality_driver()


def  yao_test_ours(path = "nyt_yao_tiny.txt.norm-word2fun-42-1e-11" , time_types = ["word_mixed_fixed", "time2vec", "word_mixed_amplitude", "word_mixed_amplitude_shift", ],debug_mode= False):
    for time_type in time_types:#
        for epoch in range(20,35):
            for step in range(1,2):
                step = step*200-1
                model = Word2Fun(os.path.join("models", path, time_type),
                             epoch=epoch,step =step,do_prune_vocab =False)
                evaluator = Evaluator(model, debug_mode = debug_mode)
                evaluator.alignment_quality_driver()

def demo():
    model = Word2Fun(os.path.join("models", "demo"), do_prune_vocab=False)
    evaluator = Evaluator(model, debug_mode=False)
    evaluator.alignment_quality_driver()

def test_compass():
    # compass = DynamicWordEmbeddingWithCompass(model_path="model-coha",corpus="coha.txt.raw.token.decade-output")
    #
    # evaluator = Evaluator(compass, debug_mode=True)
    #
    # print(evaluator.check_ssd_driver())

    compass = DynamicWordEmbeddingWithCompass(model_path="model-nyt-50")
    evaluator = Evaluator(compass, debug_mode=True,timer = Timer())
    # evaluator.semantic_sim_driver()
    evaluator.alignment_quality_driver()

def compare(debug_mode = False):
    compass = DynamicWordEmbeddingWithCompass(model_path="model-nyt")
    evaluator = Evaluator(compass, debug_mode=debug_mode)
    # evaluator.semantic_sim_driver()
    # evaluator.alignment_quality_driver(do_normarlization= False)
    evaluator.alignment_quality_driver(do_normarlization=True)

    word2fun1 = Word2Fun(os.path.join("models", "nyt_yao_tiny.txt.norm-waby_amplitude-100", "word_mixed_amplitude"),
                         epoch=19) #,do_prune_vocab=True
    evaluator = Evaluator(word2fun1, debug_mode=debug_mode)
    evaluator.semantic_sim_driver()
    # evaluator.alignment_quality_driver(do_normarlization=True)
    # evaluator.alignment_quality_driver(do_normarlization=False)


def test_align_in_compase():
    # use genral gensim could load it properly

    from gensim.models.word2vec import Word2Vec
    model1 =  Word2Vec.load("model-nyt/1990.model")
    model2 = Word2Vec.load("model-nyt/2015.model")
    print(sum(model1.syn1neg[model1.wv.vocab["software"].index][:10]))
    print(sum(model2.syn1neg[model2.wv.vocab["software"].index][:10]))
    print(model1.most_similar("apple"))
    print(model2.most_similar("apple"))

def test_performance(path = "nyt_yao_tiny.txt.norm-word2fun", predix = "./" ):
    for time_type in ["word_mixed_fixed", "time2vec", "word_mixed_amplitude_shift", "word_linear" ]:

        for epoch in range(15,19):
            word2fun1 = Word2Fun(
                os.path.join(predix, path, time_type), epoch=epoch)
            evaluator = Evaluator(word2fun1, debug_mode=True)
            # evaluator.semantic_sim_driver()
            evaluator.alignment_quality_driver(do_normarlization=True)

def parameter_word2fun():
    file = "coha.txt.raw.token-word2fun"
    file = "coha-word2fun"
    for epoch in [9]:
        model = Word2Fun(os.path.join(file,"word_mixed_amplitude"),do_prune_vocab=False,epoch=epoch )
        print(model)
        avg = model.model.time_encoder.para_embedding.weight.abs().mean(-1).cpu()
        values, indexs = avg.topk(k=100,largest =True)
        print([ model.id2word[i] for i in indexs.numpy() ])

        # avg = model.model.time_encoder.para_embedding.weight.abs().mean(-1).cpu()
        # avg = model.model.time_encoder.amplitude_embedding.weight.abs().mean(-1).cpu()
        avg = model.model.u_embeddings.weight.abs().mean(-1).cpu() *-1
        # avg = model.model.time_encoder.amplitude_embedding.weight.abs().mean(-1).cpu()/
        values, indexs = avg.topk(k=100, largest=False)
        print([model.id2word[i] for i in indexs.numpy()])

        words, scores =[],[]
        for line in open("data/grade.txt"):
            word, score = line.split()
            word = word.split("_")[0]
            if word in model.word2id:
                words.append(word)
                scores.append(float(score))
            else:
                print("{} not found".format(word) )


        indexs = [ model.word2id[str(word)] for word in words if word in model.word2id ]
        preds = avg[indexs].detach().numpy()
        from scipy.stats import spearmanr,pearsonr
        print(spearmanr(preds,scores))
        print(pearsonr(preds, scores))

import  math
def derivative():
    file = "coha.txt.raw.token-word2fun"
    file = "coha-word2fun"
    for epoch in [9]:
        model = Word2Fun(os.path.join(file,"word_mixed_amplitude"),do_prune_vocab=False,epoch=epoch )

        fre = model.model.time_encoder.para_embedding.weight.detach().numpy()
        amp = model.model.time_encoder.amplitude_embedding.weight.detach().numpy()
        bias = model.model.u_embeddings.weight.detach().numpy()





        avgs = []
        words, scores =[],[]
        for line in open("data/grade.txt"):
            word, score = line.split()
            word = word.split("_")[0]
            if word in model.word2id:
                words.append(word)
                scores.append(float(score))

                index  = model.word2id[word]

                acumulated = []
                for j in range(0, 19):
                    norm = []
                    for i in range(50):
                        if i < 25:
                            norm.append(math.fabs( amp[index][i] * fre[index][i] * math.sin( fre[index][i] * j)) ) #/bias[index][i]
                        else:
                            norm.append(math.fabs( -1*  amp[index][i] * fre[index][i] * math.cos( fre[index][i] * j)))  #/bias[index][i]
                    norm = np.linalg.norm(np.array(norm),1)
                    acumulated.append(norm)
                avgs.append( sum(acumulated)/ len(acumulated))
            else:
                print("{} not found".format(word) )

        print(avgs)

        from scipy.stats import spearmanr,pearsonr
        print(spearmanr(avgs,scores))
        print(pearsonr(avgs, scores))

def get_pos_neg(data_file = "data/grade.txt"):
    words, scores = [], []
    for line in open(data_file):
        word, score = line.split()
        word = word.split("_")[0]
        words.append(word)
        scores.append(float(score))
    scores = np.array(scores)
    median = np.median(scores)
    # neg = [word for word, score in zip(words, scores) if score <= median]
    # pos = [word for word, score in zip(words, scores) if score > median]
    pos = [word for word,score in zip(words,scores) if score > np.percentile(scores,90)  ]
    # print( [score for word, score in zip(words, scores) if score > np.percentile(scores, 90)])
    neg = [word for word, score in zip(words, scores) if score <= np.percentile(scores,10)]
    # print([score for word, score in zip(words, scores) if score <= np.percentile(scores,10)])
    return pos,neg



def case_study_word2fun():
    file = "coha.txt.raw.token-word2fun"
    for epoch in range(10):
        model = Word2Fun(os.path.join(file, "word_mixed_amplitude"), do_prune_vocab=False, epoch=epoch)
        pos,neg = get_pos_neg()
        index_pos = [model.word2id[str(word)] for word in pos if word in model.word2id]
        fre = model.model.time_encoder.para_embedding.weight[index_pos].abs().mean().cpu()
        amplitude = model.model.time_encoder.amplitude_embedding.weight[index_pos].abs().mean().cpu()
        bias = model.model.u_embeddings.weight[index_pos].abs().mean().cpu()

        print(fre,amplitude,bias)

        index_neg = [model.word2id[str(word)] for word in neg if word in model.word2id]
        fre_neg = model.model.time_encoder.para_embedding.weight[index_neg].abs().mean().cpu()
        amplitude_neg = model.model.time_encoder.amplitude_embedding.weight[index_neg].abs().mean().cpu()
        bias_neg = model.model.u_embeddings.weight[index_neg].abs().mean().cpu()

        print(fre_neg, amplitude_neg, bias_neg)
        print(model.model.u_embeddings.weight[index_neg].shape)


def draw_case():

    model = Word2Fun(os.path.join( "coha-word2fun", "word_mixed_amplitude"),
                     epoch=9,  do_prune_vocab=False)
    print(model)
    pos, neg = get_pos_neg()
    print(pos, neg)
    index_pos = [model.word2id[str(word)] for word in pos if word in model.word2id]
    fre = model.model.time_encoder.para_embedding.weight[index_pos].cpu().detach().numpy()
    amplitude = model.model.time_encoder.amplitude_embedding.weight[index_pos].cpu().detach().numpy()
    bias = model.model.u_embeddings.weight[index_pos].cpu().detach().numpy()



    index_neg = [model.word2id[str(word)] for word in neg if word in model.word2id]
    fre_neg = model.model.time_encoder.para_embedding.weight[index_neg].cpu().detach().numpy()
    amplitude_neg = model.model.time_encoder.amplitude_embedding.weight[index_neg].cpu().detach().numpy()
    bias_neg = model.model.u_embeddings.weight[index_neg].cpu().detach().numpy()

    # print(fre_neg, amplitude_neg, bias_neg)
    # print(model.model.u_embeddings.weight[index_neg].shape)
    from adjustText import adjust_text

    x = np.linspace(0,20,150)
    for i in range(50):
        # print(fre[:,i], amplitude[:,i], bias[:,i])
        # print(fre_neg[:,i], amplitude_neg[:,i], bias_neg[:,i])
        f = np.cos if i < 25 else np.sin
        # types = ["-","--","o","-."]
        types = ["", "", "", ""]*100

        for j in range(len(fre[:,i])):
            y = f(x * fre[j,i] ) *  amplitude[j,i] + bias[j,i]
            plt.plot(x,y ,'b' +types[j],label = "semantically-shifted",linewidth = 1 ) # pos[j]
            plt.text(x[0],y[0], pos[j])
            # plt.text(x[-1], y[-1], pos[j])
        # adjust_text([plt.text(x[0],y[0], pos[j]) for j in range(len(fre[:,i]))])
        for j in range(len(fre_neg[:, i])):
            print(fre_neg[j, i])
            y = f(x * fre_neg[j, i]) * amplitude_neg[j, i] + bias_neg[j, i]
            plt.plot(x, y,'g'+types[j],label = "semantically-unshifted",linewidth = 1)
            # plt.text(x[0], y[0], neg[j])
            plt.text(x[-1], y[-1], neg[j])
        # adjust_text([plt.text(x[0], y[0], neg[j]) for j in range(len(fre_neg[:, i]))])
        plt.xticks([0,4,14,19], ["1810s", "1860s","1960s","2010s" ] )
        # plt.show()
        # plt.legend()
        plt.savefig("figures/{}.pdf".format(i), bbox_inches="tight", pad_inches=0)
        plt.close()
        # exit()



def read_histogram():
    model = Word2Fun(os.path.join("coha-word2fun", "word_mixed_amplitude"),
                     epoch=9, do_prune_vocab=False)
    print(model)
    pos, neg = get_pos_neg()
    print(pos, neg)
    index_pos = [model.word2id[str(word)] for word in pos if word in model.word2id]
    fre = model.model.time_encoder.para_embedding.weight[index_pos].cpu().detach().numpy().flatten()
    amplitude = model.model.time_encoder.amplitude_embedding.weight[index_pos].cpu().detach().numpy().flatten()
    bias = model.model.u_embeddings.weight[index_pos].cpu().detach().numpy().flatten()

    index_neg = [model.word2id[str(word)] for word in neg if word in model.word2id]
    fre_neg = model.model.time_encoder.para_embedding.weight[index_neg].cpu().detach().numpy().flatten()
    amplitude_neg = model.model.time_encoder.amplitude_embedding.weight[index_neg].cpu().detach().numpy().flatten()
    bias_neg = model.model.u_embeddings.weight[index_neg].cpu().detach().numpy().flatten()
    import math
    scales = [
       [1/math.pi*1/20, 1/math.pi , 1],
     [0.01,   1  ],
        [0.05,1] ,
    ]
    names = ["frequency_r", "amplitude_r", "bias"]
    triples = [ [fre, fre_neg], [amplitude,amplitude_neg], [bias, bias_neg] ]
    for i, (x,y) in enumerate(triples):
        y = y [: len(x)]
        print(x.shape)
        print(y.shape)
        # print(x, y)
        alphab = [ "{0:.2f}".format(scale) for scale in scales[i]] + ["∞"]
        pos = np.arange(len(alphab))
        width = 1.0  # gives histogram aspect to the bar diagram
        ax = plt.axes()
        ax.set_xticks(pos + (width / 2))
        ax.set_xticklabels(alphab)

        frequencies = [0] * (len(alphab ))
        print(alphab)
        for num in x:
            for index, scale in enumerate(scales[i]):
                if num < scale:
                    frequencies[index] +=1
                    break
            else:
                frequencies[-1] += 1
        # frequencies = [23, 44, 12, 11, 2, 10]

        plt.bar(pos, np.array(frequencies)/ len(x), width, color='b', alpha = 0.4)

        frequencies = [0] * (len(alphab ))
        print(alphab)
        for num in y:
            for index, scale in enumerate(scales[i]):
                if num < scale:
                    frequencies[index] +=1
                    break
            else:
                frequencies[-1] += 1
        # frequencies = [23, 44, 12, 11, 2, 10]

        plt.bar(pos, np.array(frequencies)/ len(x), width, color='g', alpha = 0.4)


        plt.savefig("histogram/hist_{}.pdf".format(names[i]), bbox_inches="tight", pad_inches=0)
        plt.close()

    exit()



if __name__ == '__main__':
    for epoch in [0,1,2,3,4,5]:
        for step in [149,299,449,599,749]:
            word2fun1 = Word2Fun("re", epoch=epoch, step = step)
            evaluator = Evaluator(word2fun1, debug_mode=False)
            # evaluator.semantic_sim_driver()
            evaluator.alignment_quality_driver(do_normarlization=True)

    exit()
    read_histogram()
    exit()
    parameter_word2fun()
    exit()

if __name__ != '__main__':
    # ssd()
    # eval_yao_test(path = "nyt_yao_tiny.txt.norm-word2fun-42-1e-11",debug_mode= True)
    # eval_yao_tet(path = "nyt_yao_tiny.txt.norm-word2fun-42-1e-9",debug_mode= True)
    # eval_yao_test(path = "nyt_yao_tiny.txt.norm-word2fun-42-1e-7",debug_mode= True)
    # eval_yao_test(path = "nyt_yao_tiny.txt.norm-word2fun-42-lr25-wd10",debug_mode= False)
    # eval_yao_test(path="nyt_yao_tiny.txt.norm-word2fun-42-lr25-wd00", debug_mode=False)
    eval_yao_test(path="nyt_yao_tiny.txt.norm-king-42", time_type = "word_mixed_amplitude" ,debug_mode=True)
    eval_yao_test(path="nyt_yao_tiny.txt.norm-king-42", time_type="word_mixed_fixed", debug_mode=False)
    eval_yao_test(path="nyt_yao_tiny.txt.norm-king-42", time_type="word_mixed_amplitude_shift", debug_mode=False)
    eval_yao_test(path="nyt_yao_tiny.txt.norm-king-42-decay", time_type="word_mixed_amplitude", debug_mode=False)

    # yao_test_ours(path = "nyt_yao_tiny.txt.norm-word2fun50-42" )
    # eval_yao_test(path = "nyt_yao_tiny.txt.norm-word2fun-42-1e-11",debug_mode= False)
    # eval_yao_tet(path = "nyt_yao_tiny.txt.norm-word2fun-42-1e-9",debug_mode= True)
    # eval_yao_test(path = "nyt_yao_tiny.txt.norm-word2fun-42-1e-7",debug_mode= True)
    # eval_yao_test(path="nyt_yao_tiny.txt.norm-word2fun-42-1e-11", debug_mode=False)
    # test_performance()
    # compare()
    # demo("nyt_yao_tiny.txt.norm-word2fun-42-1e-11")
    # compare()
    # test_align_in_compase()
    # compare()
def unknown():
    file = "nyt_yao_tiny.txt.norm-king-42"
    # get_frequencies(model_path=file, timetypes=["word_mixed_amplitude"], epoch=49, do_prune_vocab=False)
    model = Word2Fun(os.path.join("models", file,  "word_mixed_amplitude"),
                     epoch=49, do_prune_vocab=False)
    avg = model.model.time_encoder.para_embedding.weight.abs().mean(-1)
    values, indexs = avg.topk( k = 100,largest=True)
    # print(model.id2word())
    print([model.id2word[index] for index in indexs.numpy()])

    avg = model.model.time_encoder.para_embedding.weight.abs().mean(-1)
    values, indexs = avg.topk(k=100, largest=False)
    # print(model.id2word())
    print([model.id2word[index] for index in indexs.numpy()])
    words,scores = [],[]
    for line in open("data/grade.txt"):
        word, score = line.split()
        word = word.split("_")[0]
        # print(word)
        if word in  model.word2id:
            words.append(word)
            scores.append(float(score))
    indexs = [model.word2id[str(word)] for word in words  if word in model.word2id]
    pred = avg[indexs]
    print(pred.detach().numpy())
    print(scores)

    print(pearsonr(pred.detach().numpy(), np.array(scores)))
    print(spearmanr(pred.detach().numpy(), np.array(scores)))
    # print(indexs,pred)


    print(words)


if __name__ == '__main__':

    pass

