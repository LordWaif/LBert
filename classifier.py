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
        lwan,
        second_level,
        lwan_args={},
        second_level_args={},
    ):
        super(CustomBertClassifier, self).__init__()
        self.model: BertModel = BertModel.from_pretrained(model_name, output_hidden_states=True)  # type: ignore
        self.num_classes = num_classes
        self.drop3 = torch.nn.Dropout(p=0.3)

        self.predicted_agregation = predicted_agregation
        self.logit_agregation = logit_agregation
        self.logit_pooler = logit_pooler
        self.hidden_layer = hidden_layer

        if self.predicted_agregation == "mean_max":
            input_linear = self.model.config.hidden_size * 2
        else:
            input_linear = self.model.config.hidden_size

        self.lwan = lwan
        if self.lwan:
            self.lwan_classifier = LWAN(
                self.model.config.hidden_size, num_classes, **lwan_args
            )
            input_linear = self.model.config.hidden_size * num_classes

        self.second_level = second_level
        if self.second_level:
            self.second_level_transformer = SecondLevelTransformer(
                self.model.config.hidden_size, **second_level_args
            )
            input_linear = self.model.config.hidden_size

        self.classifier = torch.nn.Linear(input_linear, self.num_classes)
        self.num_parameters = sum(p.numel() for p in self.parameters())

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

        if self.lwan:
            output = self.lwan_classifier(output)
        if self.second_level:
            output = self.second_level_transformer(output)

        output = self.pooler_predict_fn(output, self.predicted_agregation)

        logits = self.classifier(output)
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


class LWAN(torch.nn.Module):
    def __init__(self, embedding_dim, num_classes, num_heads):
        super(LWAN, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.attention = torch.nn.MultiheadAttention(
            embedding_dim, num_heads=num_heads, batch_first=True
        )
        # Vetores de atenção específicos para cada classe
        self.class_attention_vectors = torch.nn.Parameter(
            torch.randn(num_classes, embedding_dim)
        )

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        # Aplicar atenção para cada classe
        attention_outputs = []
        for i in range(self.num_classes):
            class_vector = (
                self.class_attention_vectors[i].unsqueeze(0).unsqueeze(1)
            )  # Shape: [1, 1, embed_dim]
            class_vector = class_vector.expand(
                batch_size, seq_len, -1
            )  # Shape: [batch_size, 1, embed_dim]
            class_attn_output, _ = self.attention(class_vector, x, x)
            attention_outputs.append(
                class_attn_output
            )  # Shape: [batch_size, embed_dim]

        # Concatenar as representações de cada classe
        concatenated_output = torch.cat(
            attention_outputs, dim=2
        )  # Shape: [batch_size, num_classes * embed_dim]
        return concatenated_output


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
