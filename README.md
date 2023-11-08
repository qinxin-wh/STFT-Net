# STFT-Net
## 1. Prerequisites
### Environment
Our experiments are conducted with Ubuntu 18.04, Pytorch 1.7.0 and CUDA 10.2. Kindly ensure that the prerequisite libraries are installed prior to executing this code:
~~~
pip install -r requirements.txt
~~~
### Data Preparation
- [LSOTB-TIR](https://github.com/QiaoLiuHit/LSOTB-TIR)
- [LASOT](https://github.com/HengLan/LaSOT_Evaluation_Toolkit)
- [GOT-10K](http://got-10k.aitestunion.com/downloads)
- [PTB-TIR](https://github.com/QiaoLiuHit/PTB-TIR_Evaluation_toolkit)

## 2. Train
- Training with multiple GPUs by DDP
~~~
python tracking/train.py --script stft --config baseline --save_dir . --mode multiple --nproc_per_node 4
~~~
- Training with single GPU (too slow, not recommended for training)
~~~
python tracking/train.py --script stft --config baseline --save_dir . --mode single
~~~

## 3. Evaluation
We incorporate [PySOT](https://github.com/STVIR/pysot) for evaluation. 
- ### Test
~~~
python -u pysot_toolkit/test.py --dataset_name <name of dataset> --tracker_name stft
~~~
- ### Eval
~~~
python pysot_toolkit/eval.py --dataset <name of dataset> --tracker_prefix stft
~~~
## Raw Results
The raw tracking results are provided in the [Raw results](https://pan.baidu.com/s/14tJ9gl1HaxJBhS2Ac98Xhg) (Baidu Driver: 7hoa). 

## 💖Acknowledgement
The realization of our ideas is based on the following approach. We are deeply grateful for the remarkable contributions they have made, as they have provided us with the opportunity to ascend upon the shoulders of pioneers and make our own humble yet noteworthy accomplishments.
- STARK ([Paper](https://arxiv.org/abs/2103.17154)) ([Code](https://github.com/researchmm/Stark))
- SAM-Detr ([Paper](https://arxiv.org/abs/2203.06883)) ([Code](https://github.com/ZhangGongjie/SAM-DETR))
- Pysot ([Code](https://github.com/STVIR/pysot))
- Pytracking ([Code](https://github.com/visionml/pytracking()))
