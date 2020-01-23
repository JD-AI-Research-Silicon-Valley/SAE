# SAE
Code for AAAI 2020 paper "[Select, Answer and Explain: Interpretable Multi-hop Reading Comprehension over Multiple Documents](https://arxiv.org/abs/1911.00484)"

Based on PyTorch

## Overview
Evaluation code for SAE-large on [HotpotQA leaderboad](https://hotpotqa.github.io/) with pretrained models. 

## Installation
1. git clone

2. Install PyTorch. The code has been tested with PyTorch >= 1.1

2. Install the requirements

3. `python -m spacy download en_core_web_sm`

4. Download [pretrained models](https://drive.google.com/open?id=1Eqgi0SYB9XRHkuMyjFpeYI_MWZ7GCNTt). Put zip file into the same folder with `main.py`, and unzip it.

## Running
Create a directory `output` in the same folder with `main.py` and then run

```
python main.py input_file
```

`input_file` can be HotpotQA dev file or other data sets organized in the same format with HotpotQA.

By default, the code uses the 0th GPU but you can change it the `main.py`.

The final prediction `pred.json` will be in the `output` folder.

## Citation
```
@inproceedings{tu2020sae,
  title={Select, Answer and Explain: Interpretable Multi-hop Reading Comprehension over Multiple Documents},
  author={Tu, Ming and Huang, Kevin and Wang, Guangtao and Huang, Jing and He, Xiaodong and Zhou, Bowen},
  booktitle={{AAAI 2020 (accepted)}},
  year={2020}
}
```
