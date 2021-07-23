# BETH Dataset Analysis



This repository contains the corresponding code for ["BETH Dataset: Real Cybersecurity Data for Anomaly Detection Research"](http://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-033.pdf) by **Kate Highnam*** (@jinxmirror13), **Kai Arulkumaran*** (@kaixhin), **Zachary Hanif*** (@zhanif3), and **Nicholas R. Jennings**. This paper was published in the [ICML](https://icml.cc/) Workshop on [Uncertainty and Robustness in Deep Learning 2021](https://sites.google.com/view/udlworkshop2021/home).

(* = equal contributions)

For the dataset, [please download it from Kaggle](https://www.kaggle.com/katehighnam/beth-dataset).


### **THIS CODE IS STILL BEING UPDATED**: This code is a slightly refactored version of what was used for the results in the paper for ease of use. 



## Organisation of the Repository

**notebooks/**: Jupyter notebooks containing simple data analysis of some hosts within the dataset to demonstrate potential applications of research

**data/**: Repository where data is expected to be stored locally

**notes/**: Presentations, papers, and other relevant notes regarding the dataset or the analysis of it is stored here.

***.py**: Code to run the benchmarks mentioned in the paper. See the `Usage` section below.

**results/**: Results from running the benchmarks will go here, e.g. visualisations of the VAE's latent space, plotting of the loss function.

**stats/**: After training the VAE+DoSE-SVM benchmark, we store the summary statistics for each trial in this directory. When testing, the VAE+DoSE-SVM benchmark expects five versions of the statistics to be recorded (seeds 1-5) in this directory to then calculated the anomaly predictions (using DoSE-SVM)



## Setup/Installation

After initialising a clean virtual environment with Python 3.8, use `requirements.txt` to install the necessary packages for our code to run:

```
pip install -r requirements.txt
```

Stay at the highest level of the directory when running any of the following commands (see the `Usage` section below).


## Usage

This is by no means a production level of code, but we hope the models and benchmark suite may be of use when utilising the BETH Dataset for your anomaly detection method or use case. 

For each of the following commands, the BETH dataset is expected to be stored in `data/`. 

### Run a Benchmark

To train each of the anomaly detection benchmarks, described in [our paper]((http://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-033.pdf)), use the following command: 

```
python run_benchmark.py --train --benchmark dose

# dose = VAE + DoSE-SVM
# rcov = Robust Covariance
# svm = One-Class SVM
# ifor = Isolation Forest
```

To test each benchmark, swap the `--train` flag for `--test` and re-run. We also provided a `train_ensemble.sh` script that will run the specified benchmark five times before testing, stating the average anomaly prediction accuracy in the end.

 ⚠️ **WARNING** ⚠️ dose takes over 24 hours to run on the BETH dataset without a GPU (and currently takes ~9GB of memory on the GPU). To test out the models beforehand to ensure your environment can execute them, we have also included a simple Gaussian dataset with three clusters. Please add the flag `--dataset gaussian` to run this!


### Run a Jupyter Notebook

For reproducibility, we included the UMAP visualisation code (takes about 4-hours on my Macbook Pro) and some data analysis notebooks in the `notebooks/` directory.

```
jupyter notebook
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
