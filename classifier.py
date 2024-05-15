import torch
from transformers import BertModel


class CustomBertClassifier(torch.nn.Module):
    def __init__(
        self,
        model_name,
        predicted_agregation,
        logit_agregation,
        logit_pooler,
        hidden_layer,
        num_classes,
    ):
        super(CustomBertClassifier, self).__init__()
        self.model: BertModel = BertModel.from_pretrained(model_name, output_hidden_states=True)  # type: ignore
        self.num_classes = num_classes
        self.drop3 = torch.nn.Dropout(p=0.3)

        self.predicted_agregation = predicted_agregation
        self.logit_agregation = logit_agregation
        self.logit_pooler = logit_pooler
        self.hidden_layer = hidden_layer

        self.num_parameters = sum(p.numel() for p in self.parameters())

        if self.predicted_agregation == "mean_max":
            input_linear = self.model.config.hidden_size * 2
        else:
            input_linear = self.model.config.hidden_size
        self.classifier = torch.nn.Linear(input_linear, self.num_classes)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        batch_size, mini_batch, _ = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        output_model = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if self.logit_pooler == "hidden_state":
            output = output_model.hidden_states[self.hidden_layer - 1]
            output = self.pooler_predict_fn(output, self.logit_agregation)
        elif self.logit_pooler == "pooler_output":
            output = output_model.pooler_output
        else:
            raise ValueError("Invalid logit_pooler")
        output = output.view(batch_size, mini_batch, -1)
        output = self.drop3(output)
        pooler_output = self.pooler_predict_fn(output, self.predicted_agregation)
        logits = self.classifier(pooler_output)
        return logits

    def pooler_predict_fn(self, logits: torch.Tensor, agregation) -> torch.Tensor:  # type: ignore
        if agregation == "mean":
            return logits.mean(dim=1)
        elif agregation == "max":
            return logits.max(dim=1)[0]
        elif agregation == "first":
            return logits[:, 0, :]
        elif agregation == "median":
            return logits.median(dim=1)[0]
        elif agregation == "mean_max":
            return torch.cat([logits.mean(dim=1), logits.max(dim=1)[0]], dim=1)
