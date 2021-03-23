import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SkipGramModel, TimestampedSkipGramModel
from data_reader import DataReader, Word2vecDataset, TimestampledWord2vecDataset
import pandas as pd
import os
import argparse
import pickle
import numpy as np
# from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sys import platform

from tqdm import tqdm

if platform != "darwin":
    plt.switch_backend('agg')

# coca 0 29  1990 - 2019
# coha 0 199  1810 2009
# arxiv 0 352 2007.4 - 2020.4
# nyt 1987- 2007
# nyt_yao 1986 - 2015

year_mapping = {
    "coha.txt.raw.token.decade-output": (
    [(i - 1810) // 10 for i in range(1810, 2020, 10)], [str(i) + "s" for i in range(1810, 2020, 10)]),
    "coca.txt.raw.token.decade-output": (
    [(i - 1990) // 10 for i in range(1990, 2020, 10)], [str(i) + "s" for i in range(1990, 2020, 10)]),
    "coca.txt.raw.token-output": ([i - 1990 for i in range(1990, 2020, 1)], [str(i) for i in range(1990, 2020, 1)]),
    "coha.txt.raw.token-output": ([i - 1810 for i in range(1810, 2009, 1)], [str(i) for i in range(1810, 2009, 1)]),
    "arxiv.txt.raw.token-output": (
    [i for i in range(0, 352, 1)], ["{}-{}".format(i // 12 + 1991, i % 12 + 1) for i in range(0, 352, 1)]),
    "nyt.txt.norm-output": ([i - 1987 for i in range(1987, 2007, 1)], [str(i) for i in range(1987, 2007, 1)]),
    "nyt_yao.txt-output": ([i - 1986 for i in range(1986, 2015, 1)], [str(i) for i in range(1986, 2015, 1)]),
}

# word_sin word_cos word_mixed word_linear word_mixed_fixed
parser = argparse.ArgumentParser(description='parameter information')
parser.add_argument('--time_type', dest='time_type', type=str, default="word_mixed_fixed",
                    help='sin cos  mixed others  linear, sin,  word_sin,word_cos,word_linear')
parser.add_argument('--text', dest='text', type=str, default="coha.txt", help='text dataset')
parser.add_argument('--use_time', dest='use_time', default=1, type=int, help='use_time or not')
parser.add_argument('--output', dest='output', default="coha", type=str, help='output dir to save embeddings')
parser.add_argument('--log_step', dest='log_step', default=1, type=int, help='log_step')
parser.add_argument('--from_scatch', dest='from_scatch', default=1, type=int, help='from_scatch or not')
parser.add_argument('--batch_size', dest='batch_size', default=64, type=int, help='batch_size')
parser.add_argument('--emb_dimension', dest='emb_dimension', default=50, type=int, help='emb_dimension')
parser.add_argument('--add_phase_shift', dest='add_phase_shift', default=0, type=int, help='add_phase_shift')
parser.add_argument('--verbose', dest='verbose', default=0, type=int, help='verbose')
parser.add_argument('--lr', dest='lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--do_eval', dest='do_eval', default=1, type=int, help='verbose')
parser.add_argument('--iterations', dest='iterations', default=5, type=int, help='iterations')
parser.add_argument('--years', dest='years', default=30, type=int, help='years')
parser.add_argument('--weight_decay', dest='weight_decay', default=0, type=float, help='verbose')
parser.add_argument('--time_scale', dest='time_scale', default=1, type=int, help='verbose')

args = parser.parse_args()
if not torch.cuda.is_available():
    args.verbose = 1

import numpy as np
import heapq
import scipy


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





class Word2VecChecker:
    def __init__(self, path="output", time_type="word_sin"):
        # for time_type in os.listdir(path):
        #     if ".DS_Store" in time_type:
        # continue
        self.path = path
        subpath = os.path.join(path, time_type)
        if args.add_phase_shift:
            subpath += "_shift"
        if not os.path.exists(os.path.join(subpath, "vectors.txt")):
            print("cannot find vectors.txt in {}, try to find {}-th iteration".format(subpath, args.iterations))
            subpath = os.path.join(subpath, str(args.iterations - 1))
            if not os.path.exists(subpath):
                print("cannot load model from {}".format(subpath))
                return
        self.embedding_dict = read_embeddings_from_file(os.path.join(subpath, "vectors.txt"))
        if args.use_time and "word2vec" not in time_type:
            self.skip_gram_model = TimestampedSkipGramModel(len(self.embedding_dict), args.emb_dimension,
                                                            time_type=time_type, add_phase_shift=args.add_phase_shift)
        else:
            self.skip_gram_model = SkipGramModel(len(self.embedding_dict), args.emb_dimension)

        self.id2word = pickle.load(open(os.path.join(subpath, "dict.pkl"), "rb"))
        self.skip_gram_model.load_embeddings(self.id2word, subpath)

        if torch.cuda.is_available():
            self.skip_gram_model.cuda()

        # print(embeddings)

    def get_similar_words(self, words, year, k=3, word2id=None):
        if word2id is None:
            word2id = {value: key for key, value in self.id2word.items()}
        embeddings_vectors = self.get_embedding_in_a_year(self.embedding_dict.keys(), word2id=word2id, year=year)

        # embeddings_vectors = np.array( [vector for word,vector in embeddings])
        # all_words = [word for word,vector in embeddings]
        not_found_words = [word for word in words if word not in word2id]
        if len(not_found_words) > 0:
            print("do not find {}".format(" ".join(not_found_words)))
        words_index = [word2id[word] for word in words if word in word2id]
        # print(words_index)

        selected_vectors = np.array([embeddings_vectors[word] for word in words_index])

        a = np.dot(selected_vectors, embeddings_vectors.T)  # /np.norm()
        # a = cosine_similarity(selected_vectors,embeddings_vectors)

        top_k = a.argsort()[:, -1 * k:]  # [::-1]
        # top_k = np.partition(a, -3)
        # print(top_k.shape)
        # print(top_k)

        words_str = [" ".join([self.id2word[word] for word in top_k_per_word[::-1]]) for top_k_per_word in top_k]
        return words_str

        # ranks = np.argsort(a,axis = 0)
        # print(ranks.argmax(0))
        # print(a.squeeze())
        # print(a.squeeze().argmax())
        # print(a.argmax(1))
        # print(a)
        # exit()

    def word_change_rate(self, words, years=30):
        vectors = []
        for year in range(years):
            word2id = {value: key for key, value in self.id2word.items()}
            embeddings_vectors = self.get_embedding_in_a_year(self.embedding_dict.keys(), word2id=word2id, year=year)

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
            embeddings = self.get_embedding_in_a_year(words, year, word2id)
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

    def get_sim_between_year(self, target, words=None, years=[i for i in range(1940, 2020, 1)], word2id=None,
                             name="nuclear"):
        name += "-" + target + "_".join(words)
        sims = []
        words.append(target)

        for year in years:
            embeddings = self.get_embedding_in_a_year(words, year)
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

    def check_ssd(self, helper):

        from scipy.spatial.distance import cosine  # cosine distance

        words = helper.words
        time_stamped_embeddings = []
        for timespan in helper.timespans:
            all_embeddings = [self.get_embedding_in_a_year(words, year) for year in timespan]
            mean_embedding = np.mean(np.array(all_embeddings), 0)
            time_stamped_embeddings.append(mean_embedding)
        assert len(time_stamped_embeddings) == 2, "more timespans than two"
        scores = [cosine(time_stamped_embeddings[0][i], time_stamped_embeddings[1][i]) for i, word in enumerate(words)]
        print(scores)
        print(helper.evaluate(scores))

    def get_embedding_in_a_year(self, words=None, year=0, word2id=None, return_known_index = False):
        if word2id is None:
            word2id = {value: key for key, value in self.id2word.items()}



        # print("___"*20)

        if type(year) != list:
            words_id = [word2id[word] for word in words]
            word_tensor = torch.LongTensor(words_id)
            time_tensor = torch.LongTensor([year] * len(words_id))
        else:
            word_tensor, time_tensor = [],[]

            known_index = []
            for index,(word,year) in enumerate(zip(words,year)):
                if word in word2id:
                    word_tensor.append(word2id[word])
                    time_tensor.append(year)
                    known_index.append(True)
                else:
                    # print("unknown word" + word)
                    known_index.append(False)


            word_tensor = torch.LongTensor(word_tensor)
            time_tensor = torch.LongTensor([int(y) for y in time_tensor])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            word_tensor = word_tensor.to(device)
            time_tensor = time_tensor.to(device)

        # print(time)
        # print(word)
        embeddings = self.skip_gram_model.forward_embedding(word_tensor, time_tensor).cpu().data.numpy()
        if return_known_index:
            return embeddings,np.array(known_index)
        return embeddings


year_mapping = {
    "coha.txt.raw.token.decade-output": ([(i-1810)//10 for i in range(1810, 2020, 10)],[str(i)+"s" for i in range(1810, 2020, 10)]),
    "coca.txt.raw.token.decade-output": ([(i-1990)//10 for i in range(1990, 2020, 10)],[str(i)+"s" for i in range(1990, 2020, 10)]),
    "coca.txt.raw.token-output": ([i-1990 for i in range(1990, 2020, 1)],[str(i) for i in range(1990, 2020, 1)]),
    "coha.txt.raw.token-output": ([i-1810 for i in range(1810, 2009, 1)],[str(i) for i in range(1810, 2009, 1)]),
    "arxiv.txt.raw.token-output": ([i for i in range(0, 352, 1)],["{}-{}".format( i//12 +1991, i%12+1 ) for i in range(0, 352, 1)]) ,
    "nyt.txt.norm-output": ([i-1987 for i in range(1987, 2007, 1)],[str(i) for i in range(1987, 2007, 1)]),
    "nyt_yao.txt.train-output": ([i-1986 for i in range(1986, 2016, 1)],[str(i) for i in range(1986, 2016, 1)]), #1986
    "nyt_yao_tiny.txt.norm.train-output": ([i-1990 for i in range(1990, 2017, 1)],[str(i) for i in range(1990, 2017, 1)]), #1986


}



def get_score(a,b):
    from sklearn.metrics import mutual_info_score
    from scipy.stats import entropy
    score = mutual_info_score(a,b)
    _, p1 = np.unique(a,return_counts=True)
    _, p2 = np.unique(a, return_counts=True)
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()
    return score/entropy(p1)/entropy(p2) *2

def get_score1(a,b):
    from sklearn.metrics import f1_score,fbeta_score
    x = [ i == j for i in a  for j in a]
    y = [ i == j for i in b for j in b]
    # print(x)
    # print(y)
    return fbeta_score(x,y,beta=5)




def semantic_sim_all(model_path, epoches=10, dataset="none",year_mapping =None):

    df = pd.read_csv("eval/yao/testset_1.csv")

    for time_type in ["word_mixed_fixed"]:  # "word_cos", , "word_cos"
        epoches = 10 if "mixed_fixed" in time_type else 5

        for epoch in range(1, epoches, 1):
            save_filename = "{}-{}-{}-sim_word_log.txt".format(dataset, epoch, time_type)
            print("save log in {}".format(save_filename))
            with open(save_filename, "w", encoding="utf-8") as f:
                args.iterations = epoch
                checker = Word2VecChecker(path=model_path, time_type=time_type)
                try :
                    df.real_year = df.year.apply(lambda x: int(year_mapping[str(x)] ))
                except Exception as  e:
                    print(e)
                    print(year_mapping.keys())
                    print(df.year.unique())
                    df.real_year = df.year.apply(lambda x: int(year_mapping[ str(x//10*10) +"s" ]))
                log_text = semantic_sim(checker, df)
                print(log_text)
                f.write(log_text + "\n")
                # exit()


def alignment_quality(checker,df):

    lines = ["{} ".format(checker.path)]
    # df = df.reset_index()
    # print(df)
    # print(len(df))
    embeddings, known_index = checker.get_embedding_in_a_year(df.w1, df.y1.tolist(), return_known_index=True)
    # print(max(known_index))
    years = [i-1986 for i in range(1990, 2016, 1)]
    # df = df[np.array(known_index)].reset_index()
    print(embeddings.shape)
    raw_len = len(df)

    df = df[known_index].reset_index()
    print("original len {} and finally {}".format( raw_len, len(df)))

    p1, mr, p3,p5, p10 = [],[],[],[],[]


    for i,row in tqdm(df.iterrows()):
        embedding = embeddings[i]

        candicate, known_index = checker.get_embedding_in_a_year( [row["w2"]] * len(years) , years,
                                                                  return_known_index=True)

        ranking_scores = np.dot(embedding, candicate.transpose())
        ranking_indexes = np.argsort(ranking_scores)[::-1]
        # print(ranking_scores)
        # print(ranking_indexes)

        gold_year = int(row.y2)-1986
        # print(gold_year)
        did_find = False
        for rank, index in enumerate(ranking_indexes):
            if index == gold_year:
                ranked_index = rank
                # print("bingo")
                did_find = True
        if not did_find:
            ranked_index = 100
        # print(ranked_index,gold_year)


        p1.append(1 if ranked_index<1  else 0)
        p3.append(1 if ranked_index < 3 else 0)
        p5.append(1 if ranked_index < 5 else 0)
        p10.append(1 if ranked_index < 10 else 0)
        mr.append(1/(ranked_index+1) if ranked_index!=100 else 0)
    scores= [ np.mean(s) for s in (mr,p1, p3,p5, p10)]
    # print(scores)
    # exit()


    # exit()


    return "\t".join(["{0:.4f}".format(s) for s  in scores])


def read_alignment(filename = "eval/yao/testset_2(1).csv"):
    results = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            w1, w2 = line.split(",")
            w1,y1 = w1.split("-")
            w2, y2 = w2.split("-")
            results.append((w1,y1,w2,y2))
    return pd.DataFrame(results, columns= ["w1","y1","w2","y2"])


def alignment_quality_all(model_path, epoches=10, dataset="none",year_mapping =None):

    df1 = read_alignment("eval/yao/testset_2(1).csv")
    df2 = read_alignment("eval/yao/testset_2(2).csv")

    for time_type in ["word_mixed_fixed"]:  # "word_cos", , "word_cos"
        epoches = 100 if "mixed_fixed" in time_type else 5

        for epoch in range(1, epoches, 1):
            save_filename = "{}-{}-{}-sim_word_log.txt".format(dataset, epoch, time_type)
            print("save log in {}".format(save_filename))
            with open(save_filename, "w", encoding="utf-8") as f:

                args.iterations = epoch
                print("load model in {}".format(model_path))
                checker = Word2VecChecker(path=model_path, time_type=time_type)
                for df in [df2 , df1 ]:
                    log_text = alignment_quality(checker, df)
                    print(log_text)

                    f.write(log_text + "\n")
                # exit()



if __name__ == '__main__':

    for model_path, (years, real_years) in year_mapping.items():
        d =  { year: real_year for real_year,year in zip(years,real_years)}
        print(d)
        # semantic_sim_all(model_path, dataset=model_path.split("-")[0], year_mapping = d)
        alignment_quality_all(model_path, dataset=model_path.split("-")[0], year_mapping=d)

    # embeddings = checker.get_embedding_in_a_year(words = "network", year =1990)
