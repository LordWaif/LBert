import torch
from transformers import BertModel


class CustomBertClassifier(torch.nn.Module):
    def __init__(self, model_name, predicted_pooler, num_classes):
        super(CustomBertClassifier, self).__init__()
        self.model: BertModel = BertModel.from_pretrained(model_name)  # type: ignore
        self.num_classes = num_classes
        self.drop3 = torch.nn.Dropout(p=0.3)
        self.classifier = torch.nn.Linear(
            self.model.config.hidden_size, self.num_classes
        )
        self.predicted_pooler = predicted_pooler

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        batch_size, mini_batch, seg_length = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        output_model = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output_model.pooler_output
        pooled_output = pooled_output.view(batch_size, mini_batch, -1)
        output = self.drop3(pooled_output)
        pooler_output = self.pooler_predict_fn(output)
        logits = self.classifier(pooler_output)
        return logits

    def pooler_predict_fn(self, logits: torch.Tensor) -> torch.Tensor:  # type: ignore
        if self.predicted_pooler == "mean":
            return logits.mean(dim=1)
        elif self.predicted_pooler == "max":
            return logits.max(dim=1)[0]
