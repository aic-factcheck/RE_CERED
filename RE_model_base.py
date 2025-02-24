import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel, AutoConfig, AutoTokenizer

class consts:
    XLMR = 'xlm-roberta-large'
    mBERT = 'bert-base-multilingual-cased'
    RoBeCzech = 'ufal/robeczech-base'

    ADDITIONAL_SPECIAL_TOKENS = ['[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]']

    E1_START_TOKEN: str = '[unused1]'
    E1_END_TOKEN: str = '[unused2]'
    E2_START_TOKEN: str = '[unused3]'
    E2_END_TOKEN: str = '[unused4]'
    BLANK_TOKEN: str = '[unused5]'

class RelationExtractionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, model_config: AutoConfig, tokenizer: AutoTokenizer):
        super().__init__(model_config)
        self.model: AutoModel = AutoModel.from_pretrained(consts.XLMR, config=model_config)
        self.model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(model_config.hidden_size * 3, model_config.num_labels)

        self.e1_start_id = tokenizer.convert_tokens_to_ids(consts.E1_START_TOKEN)
        self.e2_start_id = tokenizer.convert_tokens_to_ids(consts.E2_START_TOKEN)
        self.cls_token_id = tokenizer.cls_token_id

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        e1_mask = (input_ids == self.e1_start_id).unsqueeze(-1).expand(sequence_output.size())
        entity_a = torch.sum(sequence_output * e1_mask, dim=1)

        e2_mask = (input_ids == self.e2_start_id).unsqueeze(-1).expand(sequence_output.size())
        entity_b = torch.sum(sequence_output * e2_mask, dim=1)

        cls_mask = (input_ids == self.cls_token_id).unsqueeze(-1).expand(sequence_output.size())
        cls_embedding = torch.sum(sequence_output * cls_mask, dim=1)

        embedding = torch.cat([entity_a, entity_b, cls_embedding], dim=1)
        embedding = self.dropout(embedding)

        logits = self.classifier(embedding)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits} if labels is not None else {"logits": logits}