import os
import json



from nltk.tokenize import word_tokenize

def read_text(filename = "parapraphs.txt", id_to_year =None):
    docs = json.load(open(filename))
    # print(docs[:5])
    texts = []
    for  doc in docs:

        paragraphs = " ".join(doc["paragraphs"])
        # for d in paragraphs:
        words = " ".join(word_tokenize(paragraphs))

        if id_to_year is not None:
            texts.append (  id_to_year[doc["id"]][:4]  + " " + words  )
        else:
            texts.append(words)
    return texts

def process(path = "/nfsd/quartz/benyou/dataset/nyt"):
    with open("nyt_yao.txt","w") as f:
        for raw_filename in os.listdir(path):
            filename = os.path.join(path,raw_filename)
            if raw_filename.startswith("paragraphs") and raw_filename.endswith(".json"):
                year = str( int(raw_filename[11:15]) -1986)
                texts = read_text(filename)
                texts = "\n".join([" ".join([year,text]) for text in texts])
                f.write(texts+"\n")


def get_id_to_year( path= "/nfsd/quartz/benyou/dataset/tinynyt/Data_NYT/articles-search-1990-2016.json"):
    docs = json.load(open(path))
    id_to_year = dict()
    for doc in docs:
        id_to_year[doc["id"]] = doc["date"]
    return id_to_year



def process_tiny(path = "/nfsd/quartz/benyou/dataset/tinynyt/Data_NYT"):

    id_to_year = get_id_to_year()

    with open("nyt_yao_tiny.txt","w") as f:
        for raw_filename in os.listdir(path):
            filename = os.path.join(path,raw_filename)
            if raw_filename.startswith("paragraphs") and raw_filename.endswith(".json"):

                texts = read_text(filename, id_to_year)
                texts = "\n".join( texts)
                f.write(texts+"\n")
if __name__ == "__main__":
    # process()
    process_tiny()
    # process_tiny()