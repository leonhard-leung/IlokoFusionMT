"""
model.py

This module contains two types of sequence-to-sequence transformer models:

1. BaseNMT: vanilla Seq2Seq model using T5
2. LexiconPointerNMT: Seq2Seq model with a lexicon pointer for OOV words

LexiconPointerNMT:
- Adds a pointer layer that predicts whether to copy a token from the lexicon.
- Includes decode_with_lexicon() for lexicon replacement during inference.
"""

import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration

class BaseNMT(nn.Module):
    def __init__(self, model_name="t5-base"):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        return outputs

class LexiconPointerNMT(nn.Module):
    def __init__(self, base_model, lexicon: dict, tokenizer: T5Tokenizer):
        super().__init__()
        self.model = base_model
        self.lexicon = lexicon
        self.tokenizer = tokenizer

        self.pointer_layer = nn.Linear(self.model.model.config.d_model, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        hidden_states = outputs.decoder_hidden_states[-1]
        pointer_prob = torch.sigmoid(self.pointer_layer(hidden_states))

        return outputs, pointer_prob

    def decode_with_lexicon(self, input_sentence, max_length=50, pointer_threshold=0.5):
        self.model.eval()

        input_ids = self.tokenizer(input_sentence, return_tensors="pt").input_ids

        outputs = self.model.model.generate(
            input_ids,
            max_length=max_length,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_ids = outputs.sequences

        final_tokens = []
        for i, token_id in enumerate(generated_ids[0]):
            token = self.tokenizer.decode([token_id], skip_special_tokens=True)

            decoder_hidden = self.model.model(
                input_ids=input_ids,
                decoder_input_ids=generated_ids[:, :i+1],
                output_hidden_states=True,
            ).decoder_hidden_states[-1][:, -1, :]

            pointer_prob = torch.sigmoid(self.pointer_layer(decoder_hidden)).item()

            if token in self.lexicon and pointer_prob > pointer_threshold:
                final_tokens.append(self.lexicon[token])
            else:
                final_tokens.append(token)

        return " ".join(final_tokens)