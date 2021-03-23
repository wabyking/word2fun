from scipy.spatial import distance
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
print(dir(sklearn))
import numpy as np
a = np.random.randn(8,5)
b = np.random.randn(7,5)

c =a[np.newaxis,:,:]
print(c.squeeze().shape)
# print(distance.cosine(a,b))
print(cosine_similarity(a,b).shape)
print([i for i in range(2000,2020,1)])


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

# words = ["dna", "innovazione", "invecchiamento", "anziano", "vaccino", "spaziale", "coronavirus", "pandemia", "mascherina", "vaccino", "test", "respiratore"]
year_mapping = {
    # "coha.txt.raw.token.decade-output": ([(i-1810)//10 for i in range(1810, 2020, 10)],[str(i)+"s" for i in range(1810, 2020, 10)]),
    # "coca.txt.raw.token.decade-output": ([(i-1990)//10 for i in range(1990, 2020, 10)],[str(i)+"s" for i in range(1990, 2020, 10)]),
    # "coca.txt.raw.token-output": ([i-1990 for i in range(1990, 2020, 1)],[str(i) for i in range(1990, 2020, 1)]),
    # "coha.txt.raw.token-output": ([i-1810 for i in range(1810, 2009, 1)],[str(i) for i in range(1810, 2009, 1)]),
    # "arxiv.txt.raw.token-output": ([i for i in range(0, 352, 1)],["{}-{}".format( i//12 +1991, i%12+1 ) for i in range(0, 352, 1)]) ,
    # "nyt.txt.norm-output": ([i-1987 for i in range(1987, 2007, 1)],[str(i) for i in range(1987, 2007, 1)]),
    # "nyt_yao.txt-output": ([i-1986 for i in range(1986, 2015, 1)],[str(i) for i in range(1986, 2015, 1)]),
    "newsit.txt.norm-output": ([i - 2007 for i in range(2007, 2019, 1)], [str(i) for i in range(2007, 2019, 1)]),
    "repubblica.txt.norm-output": ([i - 1984 for i in range(1984, 2019, 1)], [str(i) for i in range(1984, 2019, 1)]),
    "nyt_yao.txt.train-output": ([i - 1986 for i in range(1986, 2015, 1)], [str(i) for i in range(1986, 2015, 1)]),

}
