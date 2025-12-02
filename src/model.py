"""
model.py

This module contains two types of sequence-to-sequence transformer models:

1. BaseNMT: Standard Seq2Seq model using T5
2. LexiconPointerNMT: Extends BaseNMT with a threshold-based lexicon pointer for handling
out-of-vocabulary (OOV) words during inference.

LexiconPointerNMT:
- Adds a non-trainable pointer layer that predicts whether to replace a token using a lexicon.
- Replacement is based on a threshold probability.
- The pointer is only used during inference.
"""

import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

class BaseNMT(nn.Module):
    """
        A wrapper class around the T5ForConditionalGeneration model from HuggingFace Transformers.
    """
    def __init__(self, model_name="t5-base", **configs):
        """
        Initializes the model.

        :param model_name: Name of the pretrained T5 model (default: "t5-base")
        :param configs: Additional configuration arguments to override T5Config defaults.
        """
        super().__init__()
        config = T5Config.from_pretrained(model_name, **configs)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.

        :param input_ids: Input token IDs
        :param attention_mask: Attention mask
        :param labels: Target token IDs for training
        :return: HuggingFace model output containing loss, logits, and hidden states
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        return outputs

    def freeze_encoder(self):
        """
        Freezes the encoder parameters to prevent them from updating during training.

        :return: None
        """
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    @property
    def d_model(self):
        return self.model.config.d_model

class LexiconPointerNMT(nn.Module):
    """
    Extends the BaseNMT model by adding a Threshold-based lexicon pointer for handling
    out-of-vocabulary (OOV) words during inference.

    The pointer layer produces a probability for each generated token. Tokens that
    exceed the threshold probability and exist in the lexicon are replaced with the
    lexicon translation.

    Note:
        - The pointer is not trained, only used during inference.
    """
    def __init__(self, base_model, lexicon: dict, tokenizer: T5Tokenizer):
        """
        Initializes the model.

        :param base_model: An instance of BaseNMT
        :param lexicon: Dictionary mapping source tokens to target translations
        :param tokenizer: Tokenizer for input preprocessing
        """
        super().__init__()
        self.model = base_model
        self.lexicon = lexicon
        self.tokenizer = tokenizer

        # fixed pointer layer for scoring tokens (no training)
        self.pointer_layer = nn.Linear(self.model.d_model, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass simply delegates to the underlying BaseNMT model.

        :param input_ids: Input token IDs
        :param attention_mask: Attention mask
        :param labels: Target token IDs for training
        :return: Output of the BaseNMT model which includes loss, logits, and hidden states
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def decode_with_lexicon(self, input_sentence, max_length=128, pointer_threshold=0.5):
        """
        Generates translation and optionally replaces tokens using the lexicon based on
        the pointer probability threshold.

        :param input_sentence: Source sentence to translate
        :param max_length:Maximum length of the generated sentence
        :param pointer_threshold: Minimum pointer probability to trigger lexicon replacement
        :return: Translated sentence with OOV tokens replaced if applicable
        """
        self.model.eval()
        self.pointer_layer.eval()

        inputs = self.tokenizer(input_sentence, return_tensors="pt").input_ids

        outputs = self.model.generate(
            input_ids=inputs,
            max_length=max_length,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_ids = outputs.sequences
        decoded_tokens = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        pointer_probs = []
        generated_ids_no_eos = generated_ids[:, :-1]
        for i in range(generated_ids_no_eos.size(1)):
            decoder_out = self.model(
                input_ids=inputs,
                decoder_input_ids=generated_ids_no_eos[:, :i + 1],
                output_hidden_states=True,
            )

            hidden = decoder_out.hidden_states[-1][:, -1, :]
            prob = torch.sigmoid(self.pointer_layer(hidden)).item()
            pointer_probs.append(prob)

        tokens = decoded_tokens.split()
        for i, token in enumerate(tokens):
            if token in self.lexicon and i < len(pointer_probs):
                if pointer_probs[i] > pointer_threshold:
                    tokens[i] = self.lexicon[token]

        return " ".join(tokens)

    # Future Work:
    # - Replace the fixed threshold pointer with a trainable pointer when dataset size allows.
    # - Explore contextual scoring for lexicon replacements instead of simple linear layer.
    # - Optimize inference to avoid multiple forward passes over the decoder sequence.