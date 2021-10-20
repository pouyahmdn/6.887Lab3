# 6.887 Machine Learning for Systems
## Lab 3: Reinforcement Learning

###### Due: Monday, Oct 25 at 12:00 PM

### Instructions:

### Step 1

Install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) if you don't have it and clone this repository with this command:
```
git clone https://github.com/pouyahmdn/6.887Lab3.git
cd 6.887Lab3
```

### Step 2

We recommend using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) (or the smaller [miniconda](https://docs.conda.io/en/latest/miniconda.html)) to manage your packages.
* If using Conda:

Create a new environment:
```
conda create --name lab3_mlforsys python=3.7
```
Activate the environment
```
conda activate lab3_mlforsys
```
Install the required packages:
```
conda install pytorch torchvision torchaudio -c pytorch
conda install -c conda-forge python-wget
conda install scipy numpy matplotlib pandas scikit-learn jupyter ipykernel nb_conda_kernels tqdm tensorboard
```
* If not using Conda:

Make sure you have python (>=3.5) installed on your system. Now install the required packages:
```
python3 -m pip install torch torchvision torchaudio tensorboard wget scipy numpy matplotlib pandas scikit-learn jupyter ipykernel
```

### Step 3
Launch jupyter notebook:
```
jupyter notebook
```

### Step 4
Once completed, upload your notebook through Canvas.
