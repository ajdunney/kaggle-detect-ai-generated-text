import click
import mlflow
import torch
from torch.utils.data import DataLoader
from typing import Union

from pathlib import Path
import os
import pandas as pd
from typing import Optional

from config_loader import ProjectConfig
from dataset import Essays
from bert import BertModel

import sys


def log_metrics(params: Optional[dict] = None,
                metrics: Optional[dict] = None,
                tags: Optional[dict] = None) -> None:
    if params:
        for key, value in params.items():
            mlflow.log_param(key, value)
    if metrics:
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
    if tags:
        for key, value in tags.items():
            mlflow.set_tag(key, value)

    command_line_command = ' '.join(sys.argv)
    mlflow.set_tag("command_line", command_line_command)


def get_loss_function(loss_function_name: str) -> object:
    if loss_function_name == 'BCEWithLogitsLoss':
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(
            "loss_function_name must be 'BCEWithLogitsLoss',"
            f" not {loss_function_name}"
            )


def load_processed_data(data_path: str) -> pd.DataFrame:
    data_path = Path(data_path)
    if data_path.suffix == '.csv':
        return pd.read_csv(data_path)
    elif data_path.suffix == '.parquet':
        return pd.read_parquet(data_path)
    else:
        raise ValueError(
            "data_path must be a .csv or .parquet file, not {data_path.suffix}"
        )


def prepare_datasets(df: pd.DataFrame,
                     fold: Optional[int] = None,
                     tokenizer: object = None,
                     max_len: int = 512) -> tuple[torch.utils.data.Dataset,
                                                  torch.utils.data.Dataset]:
    train_dataset = df[df['fold'] != fold].reset_index(drop=True)
    val_dataset = df[df['fold'] == fold].reset_index(drop=True)

    training_set = Essays(train_dataset, tokenizer, max_len)
    validation_set = Essays(val_dataset, tokenizer, max_len)

    return training_set, validation_set


def create_data_loaders(training_set: torch.utils.data.Dataset,
                        validation_set: torch.utils.data.Dataset,
                        train_params: dict,
                        eval_params: dict):
    train_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(validation_set, **eval_params)
    return train_loader, val_loader


def get_tokenizer(tokenizer_name: str = 'distilbert-base-cased') -> object:
    if tokenizer_name == 'distilbert-base-cased':
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer

    elif tokenizer_name == 'roberta-base':
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer
    
    else:
        raise NotImplementedError(
            "tokenizer_name must be 'distilbert-base-cased',"
            f" not {tokenizer_name}"
            )

    return tokenizer.from_pretrained(
            tokenizer_name, do_lower_case=False
            )


def _train_and_val_epoch(
        epoch_num: int,
        model,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.modules.loss,
        optimizer,
        device: Union[str, torch.device] = 'cuda',
        phase: str = 'train'
        ) -> tuple[float, float]:

    if phase == 'train':
        model.train()
    else:
        model.eval()

    tr_loss, n_correct, nb_tr_steps, nb_tr_examples = 0, 0, 0, 0
    total_samples = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    total_steps = total_samples // batch_size + (1 if total_samples % batch_size != 0 else 0)

    for _, data in enumerate(data_loader):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float32)

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(ids, mask).squeeze(1)
            loss = loss_fn(outputs, targets)
            tr_loss += loss.item()

            probabilities = torch.sigmoid(outputs.data)
            predictions = (probabilities > 0.5).int()
            n_correct += (predictions == targets).sum().item()

        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        loss_step = tr_loss / nb_tr_steps
        accu_step = (n_correct * 100) / nb_tr_examples
        samples_done = nb_tr_examples
        samples_remaining = total_samples - samples_done
        log_message = f"Step {_}/{total_steps} - Loss: {loss_step:.4f}, Accuracy: {accu_step:.2f}%, Processed: {samples_done}, Remaining: {samples_remaining}"
        print(log_message.center(80, " "))

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    log_string = 'Training' if phase == 'train' else 'Validation'
    print(f"{log_string} Loss Epoch {epoch_num}: {epoch_loss}")
    print(f"{log_string} Accuracy Epoch {epoch_num}: {epoch_accu}")

    return epoch_loss, epoch_accu


def infer(model, test_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for _, data in enumerate(test_loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            outputs = model(ids, mask)
            probabilities = torch.sigmoid(outputs).squeeze(1).tolist()
            predictions.extend(probabilities)

    return predictions


def set_up_dataloader_params(train_config: ProjectConfig,
                             splits: list[str]) -> dict:
    if splits == str:
        splits = [splits]
    params = {}
    for split in splits:
        params[split] = {
            'batch_size': train_config[split + '_batch_size'],
            'shuffle': train_config[split + '_shuffle'],
            'num_workers': train_config[split + '_num_workers']
        }
    return params


def _train_model(train_config: ProjectConfig,
                 model: object,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 loss_function: torch.nn.modules.loss,
                 optimizer: torch.optim.Optimizer,
                 device: Union[str, torch.device] = 'cuda') -> None:

    for epoch in range(train_config['num_epochs']):
        train_loss, train_acc = _train_and_val_epoch(
            epoch, model, train_loader, loss_function, optimizer, device,
            'train')
        eval_loss, eval_acc = _train_and_val_epoch(
            epoch, model, val_loader, loss_function, optimizer, device, 'eval')

        mlflow.log_metric("Training Loss", train_loss)
        mlflow.log_metric("Training Accuracy", train_acc)
        mlflow.log_metric("Validation Loss", eval_loss)
        mlflow.log_metric("Validation Accuracy", eval_acc)


def _train_bert_model(config: ProjectConfig,
                      bert_model: str = 'distilbert') -> None:
    model_config = config.get_config('model')[bert_model]
    train_config = config.get_config('training')
    data_config = config.get_config('data')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = get_tokenizer(model_config['tokenizer_name'])
    dl_params = set_up_dataloader_params(train_config, splits=['train', 'eval'])

    df = load_processed_data(
        os.path.join(
            data_config['processed_data_folder'],
            'train.csv')
    )

    test_dataframe = pd.read_csv(
        os.path.join(
            data_config['raw_data_folder'],
            'test_essays.csv')
    )

    test_data = Essays(test_dataframe, tokenizer, model_config['max_len'])
    test_loader = DataLoader(
        test_data,
        batch_size=train_config['eval_batch_size'],
        shuffle=False)

    all_fold_predictions = []
    loss_function = get_loss_function(train_config['loss_function'])
    num_folds = df['fold'].nunique()

    with mlflow.start_run():
        for fold in range(num_folds):
            print(f"| Fold {fold+1} |".center(80, "-"))

            training_set, validation_set = prepare_datasets(
                df, fold, tokenizer, model_config['max_len']
                )

            train_loader, val_loader = create_data_loaders(
                training_set, validation_set,
                dl_params['train'], dl_params['eval']
            )

            model = BertModel(model_config).to(device)
            optimizer = torch.optim.Adam(params=model.parameters(),
                                         lr=float(train_config['learning_rate']))

            for epoch in range(train_config['num_epochs']):
                _train_model(train_config, model, train_loader, val_loader,
                             loss_function, optimizer, device)

            fold_predictions = infer(model, test_loader, device)
            print(f'Made predictions for fold {fold+1}')
            all_fold_predictions.append(fold_predictions)

        log_metrics(params=model_config.update(train_config))

    average_predictions = [
        sum(x) / num_folds for x in zip(*all_fold_predictions)]

    output_df = pd.DataFrame(
        {
            'id': test_dataframe['id'],
            'generated': average_predictions
        }
    )

    output_df.to_csv('submission.csv', index=False)


@click.command()
@click.option('--model_name', type=str)
@click.option('--config_path', type=str, default='proj_config.yaml')
def train(model_name: str = 'distilbert',
          config_path: str = 'proj_config.yaml'):

    config = ProjectConfig(config_path)
    if model_name == 'distilbert' or model_name == 'roberta-base':
        _train_bert_model(config=config, bert_model=model_name)
    else:
        raise NotImplementedError(
            "model_name must be 'distilbert' or 'roberta-base, not {model_name}"
        )


if __name__ == '__main__':
    train()
