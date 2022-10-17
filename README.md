# Learning From Data Assignment 3: Multi-Class Classification for Text Reviews


### Installation 

Note: python 3.10 is required

```shell
python -m venv .venv
source .venv/bin/activate
python -m pip install -U -r requirements.txt 
```


### Dataset split 

Split default dataset (`datasets` folder `reviews.txt` file) with 0.7/0.15/0.15 as train/val/test sets as `csv` files.
```shell
python dataset_split.py
```

For more information and additional parameters please refer to the script help
```shell
python dataset_split.py --help
```

The split dataset used for our experiments is uploaded to the git and can be found in `datasets` folder. 



### Experiments

Notebooks with experiments are available in `experiments` folder. 
They are optimized to be used with collab and requre dataset files to be uploaded. 


### Training model from scratch 

Training best model with default dataset (`datasets` folder `train.csv`, `val.csv` and `test.csv` files)
```shell
python train.py
```

For more information and additional parameters please refer to the script help
```shell
python train.py --help
```

The best model we trained can be found at [HuggingFace Model Hub](https://huggingface.co/k4black/distilbert-base-uncased-reviews-finetuned).


### Predict with trained model 

Download and run best model on provided dataset file (`datasets/test.csv` by default).
```shell
python predict.py
```

For more information and additional parameters please refer to the script help
```shell
python predict.py --help
```



