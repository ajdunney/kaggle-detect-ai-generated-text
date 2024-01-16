import pandas as pd
import torch

from torch.utils.data import Dataset


class Essays(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 tokenizer, max_len: int):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = str(self.data.text[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        if 'label' in self.data.columns:
            targets = torch.tensor(self.data.label[index], dtype=torch.long)
        else:
            print('No label column found in dataframe')
            targets = torch.tensor(0, dtype=torch.long)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': targets
        }

    def __len__(self):
        return self.len
