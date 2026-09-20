"""Sequence-classification head attached to a chosen hidden state instead of the
final one.

Measured motivation (docs/FINDINGS.md, layer probe): with a frozen gemma-3-270m
the best decision-relevant representation is at layer 8 of 18, and the shipped
Gemma3TextForSequenceClassification head attaches at the final layer (0.0497
worse). This subclass reads a configurable layer. Layer index follows
layer_probe.py: 0 = embeddings, L = hidden state after block L-1, so the final
layer == config.num_hidden_layers.
"""

import torch
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextForSequenceClassification
from transformers.utils import logging as hf_logging

logger = hf_logging.get_logger(__name__)


class Gemma3TextForSequenceClassificationHeadAtLayer(Gemma3TextForSequenceClassification):
    DEFAULT_HEAD_LAYER = 8  # measured best layer (docs/FINDINGS.md)

    def __init__(self, config, head_layer=None):
        super().__init__(config)  # builds self.model + self.score over config.num_labels
        # default to the measured best layer; set after from_pretrained for a
        # custom value (from_pretrained would otherwise forward head_layer into
        # the submodel __init__ and raise TypeError)
        self.head_layer = self.DEFAULT_HEAD_LAYER if head_layer is None else head_layer

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        labels=None,
        **kwargs,
    ):
        kwargs.pop("output_hidden_states", None)  # peft forwards it; we force it on below
        transformer_outputs = getattr(self, self.base_model_prefix)(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=True,
            **kwargs,
        )
        hidden_states = transformer_outputs.hidden_states
        # hidden_states[head_layer] with head_layer=0 embeddings; L = after block L-1
        hidden = hidden_states[self.head_layer]
        logits = self.score(hidden)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.get_text_config().pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.get_text_config().pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            non_pad_mask = (input_ids != self.config.get_text_config().pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )