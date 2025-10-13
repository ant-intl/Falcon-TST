"""
Time Series Generation Mixin for PatchMoE

This module provides generation capabilities specifically designed for time series
forecasting tasks. It extends the standard Transformers GenerationMixin to handle
time series data with proper input/output reshaping and autoregressive generation.
"""

from typing import List, Optional, Union, Callable
import torch
from transformers import GenerationMixin, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerationConfig,
    GenerateOutput,
)


class PatchMoEGenerationMixin(GenerationMixin):
    """
    Generation mixin class for PatchMoE time series forecasting.

    This class extends the standard Transformers GenerationMixin to provide
    specialized generation capabilities for time series data, including proper
    handling of multi-channel inputs and autoregressive forecasting.
    """

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        revin: Optional[bool] = True,
        num_samples: Optional[int] = 1,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Generate time series forecasts using the PatchMoE model.

        This method handles the generation of time series forecasts with proper
        input preprocessing and output postprocessing for multi-channel data.

        Args:
            inputs (torch.Tensor): Input time series data of shape:
                - [batch_size, seq_len] for single-channel
                - [batch_size, seq_len, channels] for multi-channel
            generation_config (GenerationConfig, optional): Generation configuration
            logits_processor (LogitsProcessorList, optional): Logits processors
            stopping_criteria (StoppingCriteriaList, optional): Stopping criteria
            prefix_allowed_tokens_fn (Callable, optional): Prefix token function
            synced_gpus (bool, optional): Whether to sync GPUs
            assistant_model (PreTrainedModel, optional): Assistant model
            streamer (BaseStreamer, optional): Output streamer
            negative_prompt_ids (torch.Tensor, optional): Negative prompt IDs
            negative_prompt_attention_mask (torch.Tensor, optional): Negative attention mask
            revin (bool, optional): Whether to apply RevIN normalization
            num_samples (int, optional): Number of samples to generate
            **kwargs: Additional keyword arguments

        Returns:
            torch.Tensor: Generated forecasts of shape [batch_size, pred_len, channels]

        Raises:
            ValueError: If input shape is not supported
        """
        # Extract input dimensions
        batch_size = inputs.shape[0]
        length = inputs.shape[1]
        channel = 1

        # Handle multi-channel inputs
        if len(inputs.shape) == 3:
            channel = inputs.shape[2]
            # Reshape to [batch_size * channels, seq_len] for processing
            inputs = inputs.reshape(batch_size * channel, length)
        elif len(inputs.shape) > 3:
            raise ValueError("Input shape must be [batch, seq_len, channel] or [batch, seq_len]")

        # Call parent generation method
        outputs = super().generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            revin=revin,
            **kwargs,
        )

        # Reshape outputs back to [batch_size, pred_len, channels]
        pred_len = outputs.shape[1]
        outputs = outputs.reshape(batch_size, channel, pred_len)
        outputs = outputs.transpose(1, 2).contiguous()
        return outputs

    def _greedy_search(
        self,
        input_ids: torch.Tensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.Tensor]:
        """
        Perform greedy search generation for time series forecasting.

        This method implements greedy decoding specifically for time series data,
        where the model generates forecasts autoregressively.

        Args:
            input_ids (torch.Tensor): Input time series data
            logits_processor (LogitsProcessorList, optional): Logits processors
            stopping_criteria (StoppingCriteriaList, optional): Stopping criteria
            max_length (int, optional): Maximum generation length
            pad_token_id (int, optional): Padding token ID (not used for time series)
            eos_token_id (int or List[int], optional): End-of-sequence token ID
            output_attentions (bool, optional): Whether to output attentions
            output_hidden_states (bool, optional): Whether to output hidden states
            output_scores (bool, optional): Whether to output scores
            output_logits (bool, optional): Whether to output logits
            return_dict_in_generate (bool, optional): Whether to return dict
            synced_gpus (bool): Whether to sync GPUs
            streamer (BaseStreamer, optional): Output streamer
            **model_kwargs: Additional model arguments

        Returns:
            torch.Tensor: Generated time series forecasts
        """
        # Move inputs to model device
        input_ids = input_ids.to(self.device)
        batch_size, cur_len = input_ids.shape

        # Initialize processors and criteria if not provided
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        )

        # Prepare model inputs for generation
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # Generate forecasts with specified output length
        outputs = self(
            **model_inputs,
            return_dict=True,
            max_output_length=stopping_criteria.max_length - cur_len,
        )
        return outputs
