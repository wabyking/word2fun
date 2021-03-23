import os
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import sys

mate_filename = "meta_start_years.txt"
def tokenlize(text):
    # text =text.lower()
    # print(" ".join(word_tokenize(text))) word_tokenize(text)
    return word_tokenize(text)



def clean_text(filename,new_filename, do_token = False):

	years = set()
	with open(filename, encoding="utf-8") as f:  # open(filename+".norm",encoding= "utf-8") as f
		for line in f:
			tokens = line.split()
			year = tokens[0]
			try:
				years.add(int(year))
			except:
				print(year,)
	a = np.array(list(years))
	print(a)
	print(a.min())
	with open(mate_filename,"a") as f:
		f.write("{} \t {}".format(filename,a.min()))
	smallest = a.min()
	with open(filename, encoding="utf-8") as f, open(new_filename, "w", encoding="utf-8") as newf:  #
		for line in tqdm(f):
			tokens = tokenlize(line) if do_token else line.split()
			try:
				year = int(tokens[0]) - smallest
			except:
				continue
			newline = "{} {}\n".format(year, " ".join(tokens[1:]))
			newf.write(newline)


def clear_all():
	for datatset in ["arxiv", "coha", "coca"]: # -101 1810 1990
		filename = "{}.txt.raw".format(datatset)
		print(filename)
		clean_text(filename,filename + ".token")

if len(sys.args) == 3:
	clean_text(sys.args[1],sys.args[2])
elif len(sys.args) == 2:
	clean_text(sys.args[1], "{}.token.norm".format(sys.args[1]))
# clean_text("repubblica_1984_2019.txt", "repubblica.txt.norm")
# clean_text("newsit.txt", "newsit.txt.norm")
# clean_text("newsit.txt", "newsit.txt.norm")

# clean_text("nyt_yao_tiny.txt", "nyt_yao_tiny.txt.norm",do_token =False)

# clean_text("nyt.txt", "nyt.txt.norm")




