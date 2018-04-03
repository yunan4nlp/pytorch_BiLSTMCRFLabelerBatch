export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1
basepath=$(cd `dirname $0`; pwd)
echo $basepath
nohup python -u train.py --train ./conll2000/train.txt --dev ./conll2000/test.txt --test ./conll2000/test.txt > log 2>&1 &
tail -f log
