# Description
The source code is for paper #2882.

# Package Introduction
- datasets: the code for config and dataset class.
- models: the code for neural network in Pytorch.
- prepocess: the code for processing dataset.
- utils: the code for evalution and other tools.
- data: created by download.sh, containing all runtime files.

# Prepare Data
First download the dataset.

```shell
chmod -R 700 download.sh
./download.sh
``` 

Afer that, the complete package structure is:
```
|-- datasets
|-- prepocess
|-- models
|-- utils
`-- data
    |-- train
        |-- behaviors.tsv
        |-- news.tsv
    |-- dev
        |-- behaviors.tsv
        |-- news.tsv
    |-- raw 
    `-- processed
```

# Set Environments
We first need to create a *python=3.7* virtualenv and activate it.

Then, we should intall some dependencies.
```shell
pip install -r requirements.txt
``` 

Our pytorch version is 1.6.0 and torch-geometric version is 1.6.1. If there is problem on building torch-sparse or torch-scatter, go for this [webpage](https://pytorch-geometric.com/whl/) to download whl file directly. They are two package reqiured for using torch-geometric.

# prepare the runnable dataset
The dataset downloaded from website cannot be used currently, and need to be processed first.

```shell
chmod -R 700 run.sh
./run.sh
``` 

run.sh assumes 4 GPUs will be used, if the GPU number is different, change the processes argument in the code:

```
python -m prepocess.convert_train --processes=#GPUs
```
Number of GPUs can be 1, 2, 4, 8. 

# Training and testing
To train the model, run:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --gpus=4 --epoch=10
```

Based on the validation results, choose parameters from one epoch (i.e. epoch 3) to do the test:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python validate.py --gpus=4 --epoch=3 --filenum=20
```
In these instructions, argument gpus means the number of GPUs. 

# Experiment Environment
This model was trained and tested by using 4 Tesla P40 GPUs. The memory of the device was 128GB.
