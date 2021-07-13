# BETH Dataset Analysis

This repository contains the corresponding code for "BETH Dataset: Real Cybersecurity Data for Anomaly Detection Research" by **Kate Highnam*** (@jinxmirror13), **Kai Arulkumaran*** (@kaixhin), **Zachary Hanif*** (@zhanif3), and **Nicholas R. Jennings**. This paper was published in the [ICML](https://icml.cc/) Workshop on [Uncertainty and Robustness in Deep Learning 2021](https://sites.google.com/view/udlworkshop2021/home).

(* = equal contributions)

For the dataset, [please download it from Kaggle](www.kaggle.com).

## Organisation of the Repository

**notebooks**: Jupyter notebooks containing simple data analysis of some hosts within the dataset to demonstrate potential applications of research

**data**: Repository where data is expected to be stored locally

**notes**: Presentations, papers, and other relevant notes regarding the dataset or the analysis of it is stored here.

***.py**: Code to run the benchmarks mentioned in the paper. See the `Usage` section below.

## Setup/Installation

After initialising a clean virtual environment with Python 3.8, use `requirements.txt` to install the necessary packages for our code to run:

```
pip install -r requirements.txt
```

Stay at the highest level of the directory when running any of the following commands (see the `Usage` section below.


## Usage

### Run a Benchmark



```
python run_benchmark.py --train --benchmark vae

# vae = VAE + DoSE-SVM
# rcov = Robust Covariance
# svm = One-Class SVM
# ifor = Isolation Forest
```

### Run a Jupyter Notebook

```


```





## Citing

If you user our dataset or our code in your work, please cite us using:

```
@article{highnam2021bethdata,
	title={BETH Dataset: Real Cybersecurity Data for Anomaly Detection Research},
	author={Highnam, Kate and Arulkumaran, Kai and Hanif, Zachary and Jennings, Nicholas R.},
	booktitle={ICML Workshop on Uncertainty and Robustness in Deep Learning},
	year={2021}
}
```