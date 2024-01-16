
import click 
import os
from pathlib import Path
from typing import Optional
import zipfile

from config_loader import ProjectConfig


def find_zip_files(directory: str) -> list[str]:
    path = Path(directory)
    zip_files = [file.name for file in path.glob('*.zip')]
    return zip_files


@click.command()
@click.option('--dataset_type', default='competition')
@click.option('--dataset_name', default=None)
def load_raw_kaggle_data(dataset_type: str = 'competition',
                         dataset_name: Optional[str] = None) -> None:
    config = ProjectConfig()
    config.set_kaggle_credentials()

    import kaggle
    data_config = config.get_data_config()
    kaggle_config = config.get_kaggle_config()

    zip_destination_folder = data_config['zip_destination_folder']
    raw_destination_folder = data_config['raw_data_folder']

    if dataset_type == 'competition':
        kaggle.api.competition_download_files(
            competition=kaggle_config['competition_name'],
            path=zip_destination_folder,
            quiet=False
        )

    elif dataset_type == 'dataset':
        if not dataset_name:
            raise ValueError(
                "You must provide a dataset name if dataset_type is 'dataset'"
            )
        kaggle.api.dataset_download_files(
            dataset=dataset_name,
            path=zip_destination_folder,
            quiet=False
        )
    else:
        raise ValueError(
            "dataset_type must be either 'competition'"
            f" or 'dataset', not {dataset_type}"
        )

    if not os.path.exists(raw_destination_folder):
        os.makedirs(raw_destination_folder)

    zip_files = find_zip_files(zip_destination_folder)

    for file in zip_files:
        print(f"Extracting {file} to {raw_destination_folder}")
        zip_name = os.path.join(
            zip_destination_folder, file
        )

        with zipfile.ZipFile(zip_name, "r") as zip_ref:
            zip_ref.extractall(raw_destination_folder)
    return


if __name__ == '__main__':
    load_raw_kaggle_data()
