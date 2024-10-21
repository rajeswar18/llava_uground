# llava_uground

- [🏠Homepage](https://osu-nlp-group.github.io/UGround)
- [📖Paper](https://arxiv.org/abs/2410.05243)
- [😊Model Weights](https://huggingface.co/osunlp/UGround)
- [😊Live Demo](https://huggingface.co/spaces/orby-osu/UGround) (Try it out yourself!)
- [Main Code Repo](https://github.com/OSU-NLP-Group/UGround)

Note: This is the codebase to inference and train [UGround-v1](https://github.com/OSU-NLP-Group/UGround). It is modified from [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA), with changes mainly on image processing. We may use other codebases or architectures for later versions.

## Install

1. Create a python environment


```bash
conda create -n llava_uground python=3.11 -y
conda activate llava_uground
pip install --upgrade pip  # enable PEP 660 support
```


2. Install the dependencies

There are several ways to install the package:

```bash
# Install from Github
pip install git+https://github.com/boyugou/llava_uground.git
```

```bash
# Install locally
git clone https://github.com/boyugou/llava_uground.git
pip install -e .
```

```bash
# Install from pypi (Not uploaded yet)
pip install to_be_uploaded
```

