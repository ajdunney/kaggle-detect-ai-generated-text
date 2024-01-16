import os
import pandas as pd

from config_loader import ProjectConfig
from sklearn.model_selection import StratifiedKFold


def add_k_fold_column(df: pd.DataFrame,
                      k: int = 5,
                      strat_col: str = 'stratify') -> pd.DataFrame:
    print(f"Adding {k}-fold column to dataframe")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    df = df.reset_index(drop=True)
    df["fold"] = -1

    for fold, (_, val_idx) in enumerate(skf.split(df, df[strat_col])):
        df.loc[val_idx, 'fold'] = fold

    df = df.drop(columns=[strat_col, 'source'])
    return df


def preprocess_data():
    config = ProjectConfig()
    data_config = config.get_data_config() 
    raw_data_folder = data_config['raw_data_folder']
    processed_data_folder = data_config['processed_data_folder']

    if not os.path.exists(processed_data_folder):
        os.makedirs(processed_data_folder)

    essays_train = pd.read_csv(
        os.path.join(raw_data_folder, 'train_essays.csv')
        )
    drcat02 = pd.read_csv(
        os.path.join(raw_data_folder, 'train_v3_drcat_02.csv')
        )

    essays_train['source'] = 'train_data'
    essays_train = essays_train[
        ['text', 'generated', 'source']
        ].rename(
            columns={'generated': 'label'}
            )
    drcat02 = drcat02[['text', 'label', 'source']]

    df = pd.concat([essays_train, drcat02], axis=0)
    df['stratify'] = df.label.astype(str)+df.source.astype(str)
    df = add_k_fold_column(
        df, k=config.get_training_config()['k_folds']
    )

    df.to_csv(os.path.join(
        processed_data_folder, 'train.csv'), index=False
        )


if __name__ == '__main__':
    preprocess_data()
