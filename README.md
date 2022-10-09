# Learning From Data Assignment 3: TBA




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


### Training model from scratch 

Model with default dataset (`datasets` folder `train.csv` and `val.csv` files)
```shell
python train.py
```

For more information and additional parameters please refer to the script help
```shell
python train.py --help
```

The best model we trained can be found in the `models` folder.


### Testing trained model 

Test trained model with default dataset (`datasets` folder `test.csv` files) and default model (`models` folder `model` file)
```shell
python test.py
```

For more information and additional parameters please refer to the script help
```shell
python test.py --help
```



