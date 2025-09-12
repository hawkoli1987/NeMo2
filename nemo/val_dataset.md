
# NeMO

Back to NeMo, the following calculates the loss of one microbatch of val samples:

1. 
```python
def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
    return self.forward_step(batch)  # Line 699
```

2. De
```python
def forward_step(self, batch) -> torch.Tensor:
    return self.config.forward_step_fn(self, batch)  # Line 672
```

3. The forward_step_fn is set to gpt_forward_step in the config:
```python
def gpt_forward_step(model, batch) -> torch.Tensor:
    # Prepare forward arguments from batch
    forward_args = {
        "input_ids": batch["tokens"],           # Token IDs
        "position_ids": batch["position_ids"],  # Position encodings
        "labels": batch["labels"],              # Target labels for loss
    }
    
    # Add attention mask if available
    if "attention_mask" not in batch:
        assert HAVE_TE, "Transformer Engine required for fused attention"
    else:
        forward_args["attention_mask"] = batch["attention_mask"]
    
    # Handle packed sequences for efficiency
    if "cu_seqlens" in batch:
        forward_args["packed_seq_params"] = get_packed_seq_params(batch)
    
    # Call the model's forward method
    return model(**forward_args)  # Line 145
```

4. GPT forward step:
```python
def forward(self, input_ids, position_ids, attention_mask=None, labels=None, 
           decoder_input=None, inference_context=None, packed_seq_params=None):
    # Prepare extra kwargs for packed sequences
    extra_kwargs = {"packed_seq_params": packed_seq_params} if packed_seq_params is not None else {}
    
    # Delegate to the underlying Megatron Core model
    output_tensor = self.module(  # Line 640
        input_ids,
        position_ids,
        attention_mask,
        decoder_input=decoder_input,
        labels=labels,
        inference_context=inference_context,
        **extra_kwargs,
    )
    return output_tensor
```

5. Underlining mCore model 

```python
# Created in configure_model method (line 395-414)
model = MCoreGPTModel(
    self,  # GPTConfig
    transformer_layer_spec=transformer_layer_spec,
    vocab_size=vocab_size,
    max_sequence_length=self.seq_length,
    fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
    parallel_output=self.parallel_output,
    share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
    position_embedding_type=self.position_embedding_type,
    rotary_percent=self.rotary_percent,
    rotary_base=self.rotary_base,
    seq_len_interpolation_factor=self.seq_len_interpolation_factor,
    pre_process=pre_process or parallel_state.is_pipeline_first_stage(),
    post_process=post_process or parallel_state.is_pipeline_last_stage(),
    scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
    vp_stage=vp_stage,
    **kwargs,
)
```

6. Overall call stack
validation_step(batch) 
    ↓
forward_step(batch)
    ↓
gpt_forward_step(model, batch)
    ↓
model.forward(**forward_args)
    ↓
self.module(input_ids, position_ids, attention_mask, labels, ...)
    ↓
MCoreGPTModel.forward() [Megatron Core]
    ↓
[Embedding Layer] → [Transformer Layers] → [Output Layer]
    ↓
Returns: Loss tensor (if labels provided) or Logits