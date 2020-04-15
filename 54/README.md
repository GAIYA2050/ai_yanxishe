# 111
[竞赛链接](https://god.yanxishe.com/54)
## score
|model|score|note|
|:---:|:---:|:---:|
|resnet|98.1308||


## script
nohup python main.py -m='resnet' -b=64 -e=100 -mode=2 > nohup/resnet.out 2>&1 &

python main.py -o=predict -m='resnet' -b=100