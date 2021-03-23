# dataset="coha.txt.raw.norm coca.txt.raw.norm  arxiv.txt.raw.norm"

#coca 0 29  1990 - 2019
#coha 0 199  1810 2009
#arxiv 0 352 2007.4 - 2020.4
# nyt 1987- 2007
# nyt_yao 1986 - 2015
# dataset="input.txt" -101 1810 1990
#python normalized.py
#sh split.sh




text="coha.txt.raw.token.train"
mkdir ${text}-decade-output/
python3 trainer.py --use_time 1 --time_type  word_mixed   --output  ${text}-decade-output   --text  ${text} --add_phase_shif 0 --do_eval 0  --iterations 5  --batch_size  512 --lr 0.0025  --emb_dimension  50 --weight_decay 0.00000000001  --time_scale 10

python3 trainer.py --use_time 1 --time_type  word_mixed_fixed   --output  ${text}-decade-output   --text  ${text} --add_phase_shif 0 --do_eval 0 --iterations 5 --lr 0.0025  --emb_dimension  50 --weight_decay 0.00000000001 --batch_size 512 --time_scale 10

python3 trainer.py --use_time 1 --time_type  word_sin  --output  ${text}-decade-output   --text  ${text}  --add_phase_shif 0 --do_eval 0  --iterations 5  --batch_size  512 --lr 0.0025  --emb_dimension  50 --weight_decay 0.00000000001 --time_scale 10

python3 trainer.py --use_time 1 --time_type  word_cos   --output  ${text}-decade-output   --text  ${text} --add_phase_shif 0 --do_eval 0  --iterations 5  --batch_size  512 --lr 0.0025  --emb_dimension  50 --weight_decay 0.00000000001 --time_scale 10

#    python trainer.py --use_time 0 --output  ${text}-output   --text  ${text}  --iterations 5  --batch_size  256 --lr 0.0025  --emb_dimension  50 --weight_decay 0.00000000001

#	  python trainer.py --use_time 1 --time_type  word_mixed   --output  ${text}-output   --text  ${text} --add_phase_shif 0 --do_eval 0
python3 trainer.py --use_time 1 --time_type word_linear  --output  ${text}-decade-output  --text  ${text} --add_phase_shif 0 --do_eval 0   --iterations 5  --batch_size  512 --lr 0.0025  --emb_dimension  50 --weight_decay 0.00000000001 --time_scale 10


#repubblica.txt.norm nyt_yao_tiny.txt.norm
#nyt.txt.norm  nyt_yao.txt  coha.txt.raw.token coca.txt.raw.token  arxiv.txt.raw.token nyt_yao_tiny.txt.norm
dataset="nyt_yao.txt.train   coca.txt.raw.token.train  nyt.txt.norm.train  nyt_yao_tiny.txt.norm.train "
# dataset="input.txt" # coha.txt.raw.token



for text in ${dataset}
do
	 mkdir ${text}-output/
    python3 trainer.py --use_time 1 --time_type  word_mixed   --output  ${text}-output   --text  ${text} --add_phase_shif 0 --do_eval 0  --iterations 5  --batch_size  256 --lr 0.0025  --emb_dimension  50 --weight_decay 0.00000000001

    python3 trainer.py --use_time 1 --time_type  word_mixed_fixed   --output  ${text}-output   --text  ${text} --add_phase_shif 0 --do_eval 0 --iterations 5 --lr 0.0025  --emb_dimension  50 --weight_decay 0.00000000001 # --batch_size 128

    python3 trainer.py --use_time 1 --time_type  word_sin  --output  ${text}-output   --text  ${text}  --add_phase_shif 0 --do_eval 0  --iterations 5  --batch_size  256 --lr 0.0025  --emb_dimension  50 --weight_decay 0.00000000001

    python3 trainer.py --use_time 1 --time_type  word_cos   --output  ${text}-output   --text  ${text} --add_phase_shif 0 --do_eval 0  --iterations 5  --batch_size  256 --lr 0.0025  --emb_dimension  50 --weight_decay 0.00000000001

#    python trainer.py --use_time 0 --output  ${text}-output   --text  ${text}  --iterations 5  --batch_size  256 --lr 0.0025  --emb_dimension  50 --weight_decay 0.00000000001

#	  python trainer.py --use_time 1 --time_type  word_mixed   --output  ${text}-output   --text  ${text} --add_phase_shif 0 --do_eval 0
	  python3 trainer.py --use_time 1 --time_type word_linear  --output  ${text}-output  --text  ${text} --add_phase_shif 0 --do_eval 0   --iterations 5  --batch_size  256 --lr 0.0025  --emb_dimension  50 --weight_decay 0.00000000001

#--iterations 5  --batch_size  256 --lr 0.0025  --emb_dimension  50 --weight_decay 0.00000000001

#	 python trainer.py --use_time 1 --time_type  word_mixed_fixed   --output  ${text}-output   --text  ${text} --add_phase_shif 0 --do_eval 0 --iterations 10
#	  python trainer.py --use_time 1 --time_type  word_sin  --output  ${text}-output  --add_phase_shif 1  --text  ${text} --do_eval 0

	  #	 	python trainer.py --use_time 1 --time_type  word_cos   --output  ${text}-output   --add_phase_shif 1  --text  ${text} --do_eval 0



#	  python trainer.py --use_time 1 --time_type cos   --output  ${text}-output   --text  ${text} --add_phase_shif 1 --do_eval 0
#	  python trainer.py --use_time 1 --time_type mixed   --output  ${text}-output   --text  ${text} --add_phase_shif 1 --do_eval 0
#	  python trainer.py --use_time 1 --time_type sin   --output  ${text}-output   --text  ${text} --add_phase_shif 1 --do_eval 0
#	  python trainer.py --use_time 1 --time_type cos   --output  ${text}-output   --text  ${text} --add_phase_shif 0 --do_eval 0
#	  python trainer.py --use_time 1 --time_type mixed   --output  ${text}-output   --text  ${text} --add_phase_shif 0 --do_eval 0
#	  python trainer.py --use_time 1 --time_type sin   --output  ${text}-output   --text  ${text} --add_phase_shif 0 --do_eval 0
#	  python trainer.py --use_time 1 --time_type linear  --output  ${text}-output   --text  ${text} --do_eval 0
#	  python trainer.py --use_time 1 --time_type others  --output  ${text}-output  --text  ${text} --do_eval 0
done

#python yao_test.py >  yao_results_wd8.log
#dataset=" repubblica.txt.norm "
#for text in ${dataset}
#do

#    python trainer.py --use_time 1 --time_type  word_mixed_fixed   --output  ${text}-output   --text  ${text} --add_phase_shif 0 --do_eval 0

#    python trainer.py --use_time 1 --time_type  word_sin  --output  ${text}-output   --text  ${text}  --add_phase_shif 0 --do_eval 0
#
#    python trainer.py --use_time 1 --time_type  word_cos   --output  ${text}-output   --text  ${text} --add_phase_shif 0 --do_eval 0
#
#    python trainer.py --use_time 0 --output  ${text}-output   --text  ${text} --do_eval 0 --iterations 10
#
#	  python trainer.py --use_time 1 --time_type  word_mixed   --output  ${text}-output   --text  ${text} --add_phase_shif 0 --do_eval 0
#	  #    python trainer.py --use_time 1 --time_type  word_mixed   --output  ${text}-output   --text  ${text} --add_phase_shif 1 --do_eval 0
#
#	  python trainer.py --use_time 1 --time_type word_linear  --output  ${text}-output  --text  ${text} --do_eval 0

#	python trainer.py --use_time 0 --output  ${text}-output   --text  ${text} --do_eval 0 --iterations 10
#done


# echo python trainer.py --use_time 0 --output  shit/word2vec   --text  arxiv.txt.raw.norm

# python trainer.py --use_time 1 --time_type  word_sin  --output  shit/word_sin  --add_phase_shif 0  --text  arxiv.txt.raw.norm  --log_step 1


#    python trainer.py --use_time 1 --time_type  word_mixed_fixed   --output  ${text}-output   --text  ${text} --add_phase_shif 1 --do_eval 0
#        python trainer.py --use_time 1 --time_type  word_mixed_fixed   --output  ${text}-output100   --text  ${text} --add_phase_shif 0 --do_eval 0 --iterations 200
