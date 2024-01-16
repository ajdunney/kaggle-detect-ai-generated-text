import torch
from transformers import DistilBertModel, RobertaModel

model_dict = {
    "distilbert-base-cased": DistilBertModel,
    "roberta-base": RobertaModel
}


class BertModel(torch.nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.l1 = model_dict[config['model_name']].from_pretrained(
            config["model_name"])
        self.pre_classifier = torch.nn.Linear(
            config["hidden_size"], config["pre_classifier_dim"]
            )
        self.dropout = torch.nn.Dropout(config["dropout_rate"])
        self.classifier = torch.nn.Linear(
            config["pre_classifier_dim"], 1
            )

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output