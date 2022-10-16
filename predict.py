"""Finetune pretrained model"""

from pathlib import Path

import pandas as pd
import typer
import torch
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

from utils import load_hf_dataset


IS_CUDA_AVAILABLE = torch.cuda.is_available()


app = typer.Typer(add_completion=False)


@app.command()
def main(
        dataset_file: Path = typer.Argument('datasets/test.csv', file_okay=True, dir_okay=False, writable=True, help='Dataset to predict on'),
        hf_model: str = typer.Option('k4black/distilbert-base-uncased-reviews-finetuned', help='Trained model available at HF Model Hub'),
        batch_size: int = typer.Option(4, help='Batch Size'),
        save_to_file: Path = typer.Option('prediction.csv', file_okay=True, dir_okay=False, writable=True, help='File to write predictions'),
):
    """Load HF model and predict"""

    # loading dataset
    df_input = pd.read_csv(dataset_file)
    ds, cl = load_hf_dataset({'test': dataset_file})

    # create classification pipeline
    pipe = pipeline(model=hf_model)

    # predict
    predictions = []
    for out in tqdm(pipe(KeyDataset(ds['test'], 'text'), batch_size=batch_size, truncation=True), total=len(ds['test'])):
        predictions.append(out)
    df = pd.DataFrame(predictions)
    df['id'] = df_input['id']

    # save prediction
    df.to_csv(save_to_file, index=False)


if __name__ == '__main__':
    app()
