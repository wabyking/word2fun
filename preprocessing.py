import os
import random
filenames = ["arxiv.txt","coca.txt","coha.txt"]
filenames = [os.path.join("/nfsd/quartz/benyou/dataset/temporal",filename) for filename in filenames ]
print(filenames)

for filename in filenames:
	years = dict()
	path,raw_filename = os.path.split(filename)
	with open(filename,encoding= "utf-8") as f, open(raw_filename+".raw","w",encoding="utf-8") as newf:
		for line in f:
			tokens = line.split()
			year = tokens[0]
			year = int(year)
			if "arxiv" in filename and year > 90*12:
				ori_year = year
				year, month = year //12 , year%12
				
				year = (year-100) *12 + month
				if random.random() < 0.01:
					print(ori_year,year)
			newline = "{} {}\n".format(year," ".join(tokens[1:]))
			newf.write(newline )
			years.setdefault(year,0)
			years[year]+=1

	print(filename)
	print(years)