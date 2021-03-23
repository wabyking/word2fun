import sys


if len(sys.argv) ==2:
	years = set()
	with open(sys.argv[1]) as f:
		for line in f:
			year = line.split()[0]
			years.add(year)
	print(" ".join(years))
	print(max([int(year) for year in years]))
	print(min([int(year) for year in years]))