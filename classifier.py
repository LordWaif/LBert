import torch
from transformers import BertModel


class CustomBertClassifier(torch.nn.Module):
    def __init__(
        self,
        model_name,
        predicted_agregation,
        token_agregation,
        layer_agregation,
        logit_pooler,
        hidden_layer,
        num_classes,
        second_level,
        conv,
        conv_args={},
        second_level_args={},
    ):
        super(CustomBertClassifier, self).__init__()
        self.model: BertModel = BertModel.from_pretrained(model_name, output_hidden_states=True)  # type: ignore
        self.num_classes = num_classes
        self.drop3 = torch.nn.Dropout(p=0.3)

        self.predicted_agregation = predicted_agregation
        self.token_agregation = token_agregation
        self.layer_agregation = layer_agregation
        self.logit_pooler = logit_pooler
        self.hidden_layer = hidden_layer

        if self.predicted_agregation in ["mean_max", "median_mean"]:
            input_linear = self.model.config.hidden_size * 2
        else:
            input_linear = self.model.config.hidden_size

        if self.logit_pooler == "hidden_state" and self.layer_agregation == "concat":
            input_linear = self.model.config.hidden_size * len(self.hidden_layer)

        self.conv = conv
        if self.conv:
            input_linear = self.model.config.hidden_size
            self.convolutional = Convolutional(
                input_dim=input_linear,
                num_classes=self.num_classes,
                **conv_args,
            )
        else:
            self.second_level = second_level
            if self.second_level:
                self.second_level_transformer = SecondLevelTransformer(
                    input_linear, **second_level_args
                )
            self.classifier = torch.nn.Linear(input_linear, self.num_classes)
        self.num_parameters = sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        batch_size, mini_batch, _ = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        output_model = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # print(output_model.hidden_states)
        if self.logit_pooler == "hidden_state":
            output = torch.stack(
                [output_model.hidden_states[layer] for layer in self.hidden_layer],
                dim=2,
            )
            # Layer Pooler
            output = self.pooler_predict_fn(output, self.layer_agregation, dim=2)
            if self.conv:
                output = self.drop3(output)
                output = self.convolutional(output)
                output = self.pooler_predict_fn(output, self.token_agregation, dim=2)
                output = output.view(batch_size, mini_batch, -1)
                output = self.pooler_predict_fn(output, self.predicted_agregation)
                return output
            else:
                # Token Pooler
                output = self.pooler_predict_fn(output, self.token_agregation, dim=1)
        elif self.logit_pooler == "pooler_output":
            output = output_model.pooler_output
        else:
            raise ValueError("Invalid logit_pooler")
        output = output.view(batch_size, mini_batch, -1)
        output = self.drop3(output)

        if self.second_level:
            output = self.second_level_transformer(output)

        output = self.pooler_predict_fn(output, self.predicted_agregation)

        logits = self.classifier(output)
        return logits

    def pooler_predict_fn(self, logits: torch.Tensor, agregation, dim=1) -> torch.Tensor:  # type: ignore
        """
        Applies the specified aggregation method to the logits tensor.

        Args:
            logits (torch.Tensor): The input tensor of logits.
            agregation (str): The aggregation method to apply. Options are: "mean", "max", "first", "median", "mean_max",
                              "median_mean", "concat", "sum", "cls".
            dim (int, optional): The dimension along which to perform the aggregation. Defaults to 1.

        Returns:
            torch.Tensor: The aggregated tensor.

        Raises:
            ValueError: If the specified aggregation method is not supported.

        Examples:
            >>> logits = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> model = Classifier()
            >>> model.pooler_predict_fn(logits, "mean")
            tensor([2.0000, 5.0000])

        """
        if agregation == "mean":
            return logits.mean(dim=dim)
        elif agregation == "max":
            return logits.max(dim=dim)[0]
        elif agregation == "first":
            return logits[:, 0, :]
        elif agregation == "median":
            return logits.median(dim=dim)[0]
        elif agregation == "mean_max":
            return torch.cat([logits.mean(dim=dim), logits.max(dim=dim)[0]], dim=1)
        elif agregation == "median_mean":
            return torch.cat([logits.median(dim=dim)[0], logits.mean(dim=dim)], dim=1)
        elif agregation == "concat":
            return logits.view(logits.size(0), logits.size(1), -1)
        elif agregation == "sum":
            return logits.sum(dim=dim)
        elif agregation == "cls":
            return logits[:, 0, :]
        else:
            raise ValueError("Invalid agregation method")


class SecondLevelTransformer(torch.nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(SecondLevelTransformer, self).__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, paragraph_representations):
        batch_size, segment_size, embed_dim = paragraph_representations.size()
        paragraph_representations = paragraph_representations.view(-1, embed_dim)
        # Adding a batch dimension
        context_aware_representations = self.transformer_encoder(
            paragraph_representations
        )
        context_aware_representations = context_aware_representations.view(
            batch_size, segment_size, -1
        )
        return context_aware_representations


class Convolutional(torch.nn.Module):
    def __init__(self, input_dim, kernel_size, num_filters, num_classes):
        super(Convolutional, self).__init__()
        self.conv1 = torch.nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=1,
        )
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(
            in_channels=num_filters, out_channels=num_classes, kernel_size=2, padding=1
        )

    def forward(self, data):
        data = data.permute(0, 2, 1)
        x = self.conv1(data)
        x = self.relu(x)
        x = self.conv2(x)
        return x
