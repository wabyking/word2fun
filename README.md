# word2fun


To crawl the data, you may run

```
sh crawl_dta.sh
```

to run the code

```
text="nyt_yao.txt.norm.train"
python3 trainer.py --use_time 1 --time_type  word_mixed   --output  ${text}-output   --text  ${text} --add_phase_shif 0 --do_eval 0  --iterations 5  --batch_size  512 --lr 0.0025  --emb_dimension  50 --weight_decay 0.00000000001  --time_scale 1
```

"Time type" could be one of "word_cos, word_sin, word_mixed, word_linear, word_mixed_fixed, ...". time_scale indicates how many years belongs to a group during training.


to run the evalution of two tests from Yao et.al WSDM 2017 and semantic shift detection.
```
python3  evaluation.py
```
