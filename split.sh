# dataset="coha.txt.raw.norm coca.txt.raw.norm  arxiv.txt.raw.norm"

#coca 0 29  1990 - 2019
#coha 0 199  1810 2009
#arxiv 0 352 2007.4 - 2020.4
# nyt 1987- 2007
# nyt_yao 1986 - 2015
# dataset="input.txt" -101 1810 1990
#python normalized.py


datasets="  nyt_yao_tiny.txt.norm  nyt.txt.norm  nyt_yao.txt  coha.txt.raw.token coca.txt.raw.token  arxiv.txt.raw.token repubblica.txt.norm  "

#datasets="  coha.txt  "

ratio=" 9/10 "


for dataset in ${datasets}
do
   shuf  ${dataset} > ${dataset}.shuf
   length=$(wc -l < ${dataset}.shuf)
   echo $length
    top=$(($length * $ratio ))
    echo $top
   head -n $top ${dataset}.shuf > ${dataset}.train
   tail -n $(($length-$top)) ${dataset}.shuf > ${dataset}.test
   rm ${dataset}.shuf

done

ratio=" 1/2 "


for dataset in ${datasets}
do
   shuf  ${dataset}.test > ${dataset}.shuf
   length=$(wc -l < ${dataset}.shuf)
   echo $length
    top=$(($length * $ratio ))
    echo $top
   head -n $top ${dataset}.shuf > ${dataset}.dev
   tail -n $(($length-$top)) ${dataset}.shuf > ${dataset}.test
   rm ${dataset}.shuf

done


