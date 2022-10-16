"""Dataset train/test split script"""

from pathlib import Path

import pandas as pd
import typer
from sklearn.model_selection import train_test_split


app = typer.Typer(add_completion=False)


def _line_to_dict(line: str) -> dict[str, str]:
    split_line = line.split()
    return {
        'label': split_line[0],
        'label_sentiment': split_line[1],
        'id': split_line[2],
        'text': ' '.join(split_line[3:]),
    }


@app.command()
def main(
        dataset_file: Path = typer.Option('datasets/reviews.txt', exists=True, file_okay=True, dir_okay=False, readable=True, help='Dataset file to split'),
        train_file: Path = typer.Option('datasets/train.csv', file_okay=True, dir_okay=False, writable=True, help='Train file write part of dataset'),
        val_file: Path = typer.Option('datasets/val.csv', file_okay=True, dir_okay=False, writable=True, help='Validation file write part of dataset'),
        test_file: Path = typer.Option('datasets/test.csv', file_okay=True, dir_okay=False, writable=True, help='Test file write part of dataset'),
        val_size: float = typer.Option(0.15, help='Validation data size [0, 1]'),
        test_size: float = typer.Option(0.15, help='Test data size [0, 1]'),
):
    """Split data on train, validation and test subsets."""

    # read file
    with open(dataset_file, encoding='utf-8') as f:
        lines = list(f)

    # split train/val/test set with given ratio
    # stratify by label and shuffle
    X_val_train, X_test = train_test_split(
        lines,
        test_size=test_size,
        shuffle=True,
        stratify=[i.split()[0] for i in lines],
        random_state=42,
    )
    X_train, X_val = train_test_split(
        X_val_train,
        test_size=val_size*(1+test_size),
        shuffle=True,
        stratify=[i.split()[0] for i in X_val_train],
        random_state=42,
    )

    # save to the given files
    for file, data in [(train_file, X_train), (val_file, X_val), (test_file, X_test)]:
        # save as csv file
        df_ = pd.DataFrame([_line_to_dict(i) for i in data])
        df_.to_csv(file, index=False)

        # print stats
        print(f'{file}:')
        for k, v in df_['label'].value_counts().items():
            print(f'    {k}: {v}')
        print(f'  total: {len(data)}')


if __name__ == '__main__':
    app()
