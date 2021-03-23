wget https://ucfac3cb37989621a3b76cbe3bde.dl.dropboxusercontent.com/cd/0/get/BLIQTQbLGf5gN6BRWn4xP2sO2QBtGcoNnZ6wo_iI20RRxyXEy9L2nFG-6SnC-A_lTOncGkOoo93g1iZxxHlnGCZkcYv7-05oD1nQxT-40S0CdN_3LfPnGBIT5gfjhg9be6eo9n8vOAaAC-dsoPTKJEWH/file?_download_id=05060270695208491462522788794749496212071779540399256396800566855&_notify_domain=www.dropbox.com&dl=1
unzip  NYT_archive.zip -d nyt_yao
python process_nyt.py
python normalized.py nyt_yao.txt nyt_yao.txt.norm



ratio=" 9/10 "

dataset="nyt_yao.txt.norm"

shuf  ${dataset} > ${dataset}.shuf
length=$(wc -l < ${dataset}.shuf)
echo $length
top=$(($length * $ratio ))
echo $top
head -n $top ${dataset}.shuf > ${dataset}.train
tail -n $(($length-$top)) ${dataset}.shuf > ${dataset}.test
rm ${dataset}.shuf


ratio=" 1/2 "

shuf  ${dataset}.test > ${dataset}.shuf
length=$(wc -l < ${dataset}.shuf)
echo $length
top=$(($length * $ratio ))
echo $top
head -n $top ${dataset}.shuf > ${dataset}.dev
tail -n $(($length-$top)) ${dataset}.shuf > ${dataset}.test
rm ${dataset}.shuf
