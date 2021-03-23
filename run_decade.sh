# dataset="coha.txt.raw.norm coca.txt.raw.norm  arxiv.txt.raw.norm"

#coca 0 29  1990 - 2019
#coha 0 199  1810 2009
#arxiv 0 352 2007.4 - 2020.4

# dataset="input.txt" -101 1810 1990


dataset="coha.txt.raw.token coca.txt.raw.token"
# dataset="input.txt" # coha.txt.raw.token
for text in ${dataset}
do
	 mkdir ${text}-decade-output/
	 # python trainer.py --use_time 0 --output  ${text}-output   --text  ${text} --do_eval 0 --iterations 10
	 # python trainer.py --use_time 1 --time_type  word_mixed_fixed   --output  ${text}-output   --text  ${text} --add_phase_shif 1 --do_eval 0
	 python trainer.py --use_time 1 --time_type  word_mixed_fixed   --output  ${text}-decade-output   --text  ${text} --add_phase_shif 0 --do_eval 0 --iterations 10 --time_scale 10

done

#for text in ${dataset}
#do
#	python trainer.py --use_time 0 --output  ${text}-output   --text  ${text} --do_eval 0 --iterations 10
#done


