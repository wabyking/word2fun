import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

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

if platform != "darwin":
    plt.switch_backend('agg')

# coca 0 29  1990 - 2019
# coha 0 199  1810 2009
# arxiv 0 352 2007.4 - 2020.4
# nyt 1987- 2007
# nyt_yao 1986 - 2015


year_mapping = {
    "coha.txt.raw.token.decade-output": ([(i-1810)//10 for i in range(1810, 2020, 10)],[str(i)+"s" for i in range(1810, 2020, 10)]),
    "coca.txt.raw.token.decade-output": ([(i-1990)//10 for i in range(1990, 2020, 10)],[str(i)+"s" for i in range(1990, 2020, 10)]),
    "coca.txt.raw.token-output": ([i-1990 for i in range(1990, 2020, 1)],[str(i) for i in range(1990, 2020, 1)]),
    "coha.txt.raw.token-output": ([i-1810 for i in range(1810, 2009, 1)],[str(i) for i in range(1810, 2009, 1)]),
    "arxiv.txt.raw.token-output": ([i for i in range(0, 352, 1)],["{}-{}".format( i//12 +1991, i%12+1 ) for i in range(0, 352, 1)]) ,
    "nyt.txt.norm-output": ([i-1987 for i in range(1987, 2008, 1)],[str(i) for i in range(1987, 2008, 1)]),
    "nyt_yao.txt.train-output": ([i-1986 for i in range(1986, 2016, 1)],[str(i) for i in range(1986, 2016, 1)]), #1986
    "nyt_yao_tiny.txt.norm.train-output": ([i-1990 for i in range(1990, 2017, 1)],[str(i) for i in range(1990, 2017, 1)]), #1986
}

# word_sin word_cos word_mixed word_linear word_mixed_fixed
parser = argparse.ArgumentParser(description='parameter information')
parser.add_argument('--model_path', dest='time_type', type=str, default="coha",
                    help='model path with log.txt, vocab.txt and pytorch.bin')
args = parser.parse_args()

def get_score(a,b):
    from sklearn.metrics import mutual_info_score
    from scipy.stats import entropy
    score = mutual_info_score(a,b)
    _, p1 = np.unique(a,return_counts=True)
    _, p2 = np.unique(b, return_counts=True)
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()
    return score/(entropy(p1) + entropy(p2)) *2

def get_score1(a,b):
    from sklearn.metrics import f1_score,fbeta_score
    x = [ i == j for i in a  for j in a]
    y = [ i == j for i in b for j in b]
    # print(x)
    # print(y)
    return fbeta_score(x,y,beta=5)


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


def load_model(model,filename = "pytorch.bin"):  # currently not used

    state_dict = torch.load(filename)
    print(filename)
    print(state_dict.keys())
    print(state_dict.__class__.__name__)
    exit()
    missing_keys, unexpected_keys, error_msgs = [], [], []
    prefix = ""
    metadata = getattr(state_dict,"_metadata","None")
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix = ''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1],{})
        module._load_from_state_dict(state_dict, prefix,local_metadata,True,missing_keys,unexpected_keys,error_msgs)
        for name,child in module._modules.items():
            if child is not None:
                load(child,prefix + name + ".")
    start_prefix = ""
    load(model,prefix=start_prefix)

    if len(missing_keys) > 0:
        print("weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__,missing_keys))
    if len(unexpected_keys) > 0:
        print("weights of {} not used pretrained model: {}".format(model.__class__.__name__,unexpected_keys))
    if len(error_msgs) > 0:
        print("errors in loading state_dict  for  {}  :  \n{}".format(model.__class__.__name__,error_msgs))
    return model

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

    def __init__(self,start_year =None, save_path = None, scale = None):
        if start_year is not None:
            self.start_year = start_year
        elif save_path is not None:
            for name,year in Timer.mapping.items():
                # print(name,year)
                if name in save_path:
                    print("bingo, found the first year of {} for {}".format(save_path,year))
                    self.start_year = year
        else:
            print("error for input")
        if scale is None :
            if save_path is not None and "decade"  in save_path:
                self.scale = 10
            else:
                self.scale = 1
        else:
            self.scale = scale


    def get_index(self, year):
        return (year - self.start_year)//self.scale

    def get_index_in_batch(self, years):
        return [self.get_index(year) for year in years]



class Word2VecChecker:
    def __init__(self, path="coha",epoch = None):
        self.id2word = self.load_vocab(os.path.join(path, "vocab.txt"))
        self.word2id = {value: int(key) for key, value in self.id2word.items()}
        self.model_path = path
        if epoch is not None:   # load indivusual epoch, or the last epoch
            path = os.path.join(path,str(epoch))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load( os.path.join(path,"pytorch.bin")).to(self.device)

        self.timer = Timer(save_path=path)

        # self.skip_gram_model,self.id2word = load_old(path,time_type)

    def load_vocab(self,vocob_file):
        id2word = dict()
        with open(vocob_file) as f:
            for line in f:
                _id, word = line.strip().split()
                id2word[int(_id)] = word
                id2word[int(_id)] = word
        return id2word

    def load_old(self,path,time_type):
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
            skip_gram_model = TimestampedSkipGramModel(len(self.embedding_dict), args.emb_dimension,
                                                            time_type=time_type, add_phase_shift=args.add_phase_shift)
        else:
            skip_gram_model = SkipGramModel(len(self.embedding_dict), args.emb_dimension)

        id2word = pickle.load(open(os.path.join(subpath, "dict.pkl"), "rb"))
        skip_gram_model.load_embeddings(self.id2word, subpath)
        return skip_gram_model,id2word

    def get_top_k_most_similar_words_in_specific_time(self, words, year, k=3):


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

    def get_tempory_embedding(self, words=None, year=0, return_unknow_words = False ):
        words_id = [self.word2id[word] for word in words]
        unknow_words = [i for i,word in enumerate(words) if word not in self.word2id]


        word, time = torch.LongTensor(words_id).to(self.device), torch.LongTensor([year] * len(words_id)).to(self.device)
        embeddings = self.model.forward_embedding(word, time).cpu().data.numpy()
        if return_unknow_words:
            return embeddings,unknow_words
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
            embedding = self.model.forward_embedding(word_tensor, time_tensor).cpu().data.numpy()
            return embedding


    def get_embedding_in_a_year(self, words=None, year=0, word2id=None, return_known_index = False):
        if word2id is None:
            word2id = {value: key for key, value in self.id2word.items()}

        if type(year) != list:
            words_id = [word2id[word] for word in words]
            word_tensor = torch.LongTensor(words_id).to(self.device)
            time_tensor = torch.LongTensor([year] * len(words_id)).to(self.device)
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


            word_tensor = torch.LongTensor(word_tensor).to(self.device)
            time_tensor = torch.LongTensor([int(y) for y in time_tensor]).to(self.device)

        # print(time)
        # print(word)
        embeddings = self.model.forward_embedding(word_tensor, time_tensor).cpu().data.numpy()
        if return_known_index:
            return embeddings,np.array(known_index)
        return embeddings


    def check_ssd_driver(self):
        from data.ssd import Helper
        helper = Helper("data/grade.txt", tims_scale= 10 if "decade" in self.model_path else 1)

        from scipy.spatial.distance import cosine  # cosine distance

        helper.adapt(self.word2id.keys())
        words = helper.words


        time_stamped_embeddings = []
        for timespan in helper.timespans:
            all_embeddings = [self.get_tempory_embedding(words, year) for year in timespan]
            mean_embedding = np.mean(np.array(all_embeddings), 0)
            time_stamped_embeddings.append(mean_embedding)

        assert len(time_stamped_embeddings) == 2, "more timespans than two"
        scores = [cosine(time_stamped_embeddings[0][i], time_stamped_embeddings[1][i]) for i, word in enumerate(words)]

        return helper.evaluate(scores)




    def get_sim_words_diver(self, words, years,  real_years, k=100, log_filename = "sim_log.txt"):
        simwords = []
        for year in years:
            simwords.append(self.get_top_k_most_similar_words_in_specific_time(words=words, year=year, k=k))

        lines=[]
        with open(log_filename, "w", encoding="utf-8") as f:
            for row in range(len(simwords[0])):
                line = [real_years[i] + " : " + simword[row] for i, simword in enumerate(simwords)]
                print(line)
                lines.extend(line)
            f.write("\n".join(lines))
        return

    def semantic_sim_driver(self,log_filename = "yao_test1.txt"): # time_mapping

        df = pd.read_csv("eval/yao/testset_1.csv")

        df.real_year = df.year.apply(lambda x: int(self.timer.get_index(int(x)) ))
        # print(df.real_year.unique(),df.year.unique())

        labels = set(df.label.unique())
        labels_mapping =  { label : index  for index,label in enumerate(labels) }
        df.label_id = df.label.apply(lambda  x: labels_mapping[x])
        # print(df.label_id)

        embeddings,known_index = self.get_embedding_in_a_year(df.word,df.real_year.tolist(),return_known_index =True)

        from spherecluster import SphericalKMeans

        scores = []
        for n in [10,15,20]:
            skm = SphericalKMeans(n_clusters = n)
            skm.fit(embeddings)
            # print(skm.labels_.shape)
            # print(len(df.label_id[known_index]))
            # print(sum(known_index))
            score = get_score(skm.labels_,df.label_id[known_index])
            score1 = get_score1(skm.labels_,df.label_id[known_index])
            scores.append(score)
            scores.append(score1)

        # print(scores)

        with open(log_filename, "w", encoding="utf-8") as f:
            line = "\t&".join(["{0:.4f}".format(s) for s  in scores]) + "\n"
            print(line)
            f.write(line)

        return None

    def alignment_quality_driver(self, log_filename= "alignment_quality.log"):
        df1 = read_alignment("eval/yao/testset_2(1).csv")
        df2 = read_alignment("eval/yao/testset_2(2).csv")

        with open(log_filename, "w", encoding="utf-8") as f:
            for df in [df1,df2]:

                # df = df.reset_index()
                # print(df)
                # print(len(df))
                df.y1 = df.y1.apply(lambda x: int(self.timer.get_index(int(x))))
                # embeddings, known_index = self.get_embedding_in_a_year(df.w1, df.real_year.tolist(), return_known_index=True)
                # print(max(known_index))
                years = [self.timer.get_index(i) for i in range(1990, 2017, 1)]
                # df = df[np.array(known_index)].reset_index()
                # print("candidate years {} ".format(years))
                raw_len = len(df)

                # df = df[known_index].reset_index()
                # print("original len {} and finally {}".format(raw_len, len(df)))
                # print(df[ ~ known_index])

                p1, mr, p3, p5, p10 = [], [], [], [], []
                count = 0
                for (w1,y1,w2), group in tqdm(df.groupby(["w1","y1", "w2"])):
                # for i, row in tqdm(df.iterrows()):
                #     print(group)
                #     item = group[0]
                #     w1,y1,w2 = group.w1.unique()[0], group.y1.unique()[0], group.w2.unique()[0]
                    # print(w1,y1,w2)


                    if w1 not in self.word2id or w2 not in self.word2id:
                        continue

                    count += 1
                    gold_years = [self.timer.get_index(int(y)) for y in group.y2.tolist()]
                    embedding = self.get_embedding_by_year(w1,y1).squeeze()
                    # print(embedding.shape)


                    candicate = self.get_embedding_in_a_year([w2] * len(years), years,
                                                                             return_known_index=False)

                    ranking_scores = np.dot(embedding, candicate.transpose())

                    ranking_indexes = np.argsort(ranking_scores)[::-1]
                    ranking_indexes = np.array(years)[ranking_indexes]

                    answers =  np.array( [1 if index in  gold_years else 0 for rank, index in enumerate(ranking_indexes) ])

                    first_index = -1
                    for index,ans in enumerate(answers):
                        if ans ==1:
                            first_index = index
                    assert first_index != -1, "wrong for calculating MRR"

                    p1.append(answers[0])
                    p3.append(answers[:3].mean())
                    p5.append(answers[:5].mean())
                    p10.append(answers[:10].mean())
                    mr.append(1/(first_index+1))

                print(" {} triples include {}".format(len(df.groupby(["w1","y1", "w2"])), count ))
                scores = [np.mean(s) for s in (mr, p1, p3, p5, p10)]
                print(scores)
                # exit()

                line = "\t&".join(["{0:.4f}".format(s) for s in scores])
                print(line )
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

words = ["apple", "amazon", "dna", "innovation", "data", "app", "twitter", "ranking", "quantum", "nuclear",
              "weapon", "president", "chairman", "soviet", "reagan", "trump", "biden", "obama", "olympic", "olympics",
              "china", "america", "ai", "artificial", "intelligence", "neural", "network", "language", "model",
              "information", "retrieval"] + ["iphone", "mp3"]



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









def yao_test( model_path = "nyt_yao.txt.train-output", timetypes = [],epoch = None):


    for time_type in timetypes:  # "word_cos", , "word_cos"
        # for epoch in range(1, 10, 1):

        save_filename = "sim_word_{}_{}_{}\n".format(model_path,time_type,epoch)
        model_save_path = os.path.join(model_path,time_type)
        checker = Word2VecChecker(model_save_path,epoch=epoch)

        # time_mapping = { real:year for  year,real in zip(years, real_years)}
        checker.alignment_quality_driver()
        if torch.cuda.is_available():
            print(checker.semantic_sim_driver()) #time_mapping=time_mapping


def ssd_test(model_path = "coha", timetypes = [],epoch = None):

    for time_type in timetypes:  # "word_cos", , "word_cos"
        # for epoch in range(1, 10, 1):

        save_filename = "sim_word_{}_{}_{}\n".format(model_path, time_type, epoch)
        model_save_path = os.path.join(model_path, time_type)
        checker = Word2VecChecker(model_save_path, epoch=epoch)
        # checker.get_sim_words_diver(words, years, real_years, log_filename=save_filename)
        print(checker.check_ssd_driver())


if __name__ == '__main__':
    timetypes = [ "word_mixed_fixed"      ] # "word_cos",  "word_linear", "word_mixed","word_mixed_fixed","word_sin"
    files = ["nyt_yao_tiny.txt-20-nodecay-output", "nyt_yao_tiny.txt-20-100dim-output", "nyt_yao_tiny.txt-20-half-lr-output", "nyt_yao_tiny.txt-20-half-batchsize-output"]#, "nyt_yao_tiny.txt-20-phase-output"
    # for file in files:
    #     yao_test(model_path=file, timetypes=["word_mixed_fixed"])
    yao_test("coha",timetypes=["word_mixed"] )
    exit()
    for file in files:
        for epoch in range(20):
            # ssd_test("coha.txt.raw.token.train-decade-output",timetypes=timetypes,epoch=epoch)
            yao_test(model_path=file, timetypes=["word_mixed_fixed"], epoch=epoch)
    # yao_test(model_path="nyt_yao.txt.train-output", timetypes=["word_mixed_fixed"])

    # for epoch in range(5):
    #     yao_test(model_path="nyt_yao_tiny.txt.norm.train-output",timetypes=["word_mixed_fixed"], epoch=epoch)
    #  #, "word_mixed"
    # for epoch in range(5):
    #     yao_test(model_path="nyt_yao.txt.train-output",timetypes=["word_mixed_fixed"], epoch=epoch)
    #





                    # exit()

