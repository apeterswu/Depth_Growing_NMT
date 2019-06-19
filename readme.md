# Depth Growing for Neural Machine Translation
This repository is the code for ACL 2019 short paper: Depth Growing for Neural Machine Translation.

The project is based on the [fairseq](https://github.com/facebookresearch/fairseq)
(Please get familar with the fairseq project first)


```
@inproceedings{zhu2019soft,
  title={Depth Growing for Neural Machine Translation},
  author={Wu, Lijun and Wang, Yiren and Xia, Yingce and Tian, Fei and Gao, Fei and Qin, Tao and Lai, Jianhuang and Liu, Tie-Yan},
  booktitle={ACL 2019},
  year={2019}
}
```


# Requirements and Installation
* A [PyTorch installation](http://pytorch.org/)
And install fairseq with:
```
pip install -r ./deepNMT/requirements.txt
python ./deepNMT/setup.py build develop
```

### Data
Please refer to [WMT14_EN_DE](https://github.com/pytorch/fairseq/blob/v0.6.0/examples/translation/prepare-wmt14en2de.sh) for data processing.

### Training
The detaied training procedure is:
* Train shallow model with six layers 
```
train_fairseq_en2de.sh
```
* Train first several steps of the deep model with eight layers. For example, train only 10 steps.
```
train_fairseq_en2de_deep.sh
```
* Prepare the deep model. Initialize the deep model with the parameters from the shallow model in last step.
```
build_initial_ckpt_for_deep.sh
```
* Reload the initialized deep model and train deep model with eight layers.
```
train_fairseq_en2de_deep.sh
```

### Inference
The detailed inference procedure is:
```
bash infer_deepNMT.sh 0 <shallow_model_ckpt_path>  <deep_model_ckpt_path>
```
