import os
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import os
import tarfile
import zipfile

def extract_file(path, to_directory='.'):
    if path.endswith('.zip'):
        opener, mode = zipfile.ZipFile, 'r'
    elif path.endswith('.tar.gz') or path.endswith('.tgz'):
        opener, mode = tarfile.open, 'r:gz'
    elif path.endswith('.tar.bz2') or path.endswith('.tbz'):
        opener, mode = tarfile.open, 'r:bz2'
    else: 
        print("Could not extract, as no appropriate extractor is found: " + path  )

    cwd = os.getcwd()
    os.chdir(to_directory)

    try:
        file = opener(path, mode)
        try: file.extractall()
        finally: file.close()
    finally:
        os.chdir(cwd)

from nltk.tokenize import word_tokenize
from xml.dom import minidom


def tokenlize(text):
    # text =text.lower()
    # print(" ".join(word_tokenize(text))) word_tokenize(text)
    return word_tokenize(text)

def read_xml(filename = "1815777.xml"):
    doc = minidom.parse(filename)
    items = doc.getElementsByTagName("block")

    # print(dir(items[0].getElementsByTagName("p")[0]))
    # print([ item.p.text for item  in items])

    # assert len(items) >1, "xml error, do not have exactly three blocks, but with {} blocks".format(len(items))
    # if len(items) <2:
    #     print(filename, len(items),"!!!!!")
    #     return None
    try:
        text = " ".join([p.firstChild.data for p in items[-1].getElementsByTagName("p")])
        text = text.replace("\n"," ")
        text = text.replace("\r", " ")



        return " ".join(tokenlize(text))
    except Exception as e:
        print(e)
        return None



#. /nfsd/quartz/benyou/codes/wordwave
from pathlib import Path

demo_path = "/nfsd/quartz/datasets/nyt/data/"
new_path = "nyt"
def unzip():

        if not os.path.exists(new_path):
            os.mkdir(new_path)

        for year in os.listdir(demo_path):
            path = os.path.join(demo_path,year)
            print("processing year: "+year)

            new_path_year = os.path.join(new_path,year)
            if not os.path.exists(new_path_year):
                os.mkdir(new_path_year)

            for filename in os.listdir(path):
                zipped_file = os.path.join(path,filename)
                print("unzip {} to {} ".format(zipped_file,new_path_year))
                if os.path.isfile(zipped_file):
                    extract_file(zipped_file,new_path_year)


            # exit()

def convert_raw():
    with open("nyt.txt", "w", encoding="utf-8") as f:
        for year in [str(i) for i in range (1987,2008,1)]:
            new_path_year = os.path.join(new_path, year)
            for group in os.listdir(new_path_year):
                unzipped = os.path.join(new_path_year, group)
                print("read unzipped path: " + unzipped)
                if os.path.isdir(unzipped):
                    filenames = list(Path(unzipped).glob("**/*.xml"))
                    for xml_filename in filenames:
                        # print( xml_filename)
                        text = read_xml(xml_filename.as_posix())
                        if text is not None:
                            f.write("{} {}\n".format(year, text))

if __name__ == "__main__":
    # unzip()
    convert_raw()
    # read_xml()

		