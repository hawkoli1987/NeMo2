# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Union

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


class IndividualDatasetValidationCallback(Callback):
    """
    Custom callback to track validation loss for individual datasets.
    This works by intercepting validation batches and tracking loss per dataset.
    
    This is particularly useful when using multiple validation datasets to understand
    how the model performs on each dataset individually.
    """
    
    def __init__(self, dataset_names: Optional[List[str]] = None, batch_ratio: Optional[List[float]] = None):
        """
        Args:
            dataset_names: List of dataset names to track. If None, will auto-detect from data module.
            batch_ratio: List of ratios for each dataset (e.g., [0.5, 0.5] for equal split).
                        If None, assumes equal distribution.
        """
        self.dataset_names = dataset_names or []
        self.batch_ratio = batch_ratio or [1.0/len(self.dataset_names)] * len(self.dataset_names) if self.dataset_names else [1.0]
        self.dataset_losses: Dict[str, float] = {}
        self.dataset_counts: Dict[str, int] = {}
        self.current_epoch = 0
        self.total_batches = 0
        
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset tracking at the start of each validation epoch."""
        self.dataset_losses = {}
        self.dataset_counts = {}
        self.current_epoch = trainer.current_epoch
        self.total_batches = 0
        
    def on_validation_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Any, 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        """
        Track loss for each validation batch.
        For multiple datasets in a single dataloader, we estimate dataset assignment based on batch ratios.
        """
        if outputs is None:
            return
            
        # Extract loss from outputs
        loss = self._extract_loss(outputs)
        if loss is None:
            return
            
        self.total_batches += 1
        
        # Determine dataset name
        dataset_name = self._get_dataset_name(batch_idx, dataloader_idx)
        
        # Debug: Print batch information
        if self._is_rank_zero(trainer) and batch_idx < 5:  # Only print first 5 batches
            print(f"DEBUG: Batch {batch_idx}, dataloader_idx {dataloader_idx}, dataset_name: {dataset_name}, loss: {loss:.6f}")
        
        # Accumulate loss and count
        if dataset_name not in self.dataset_losses:
            self.dataset_losses[dataset_name] = 0.0
            self.dataset_counts[dataset_name] = 0
            
        self.dataset_losses[dataset_name] += loss
        self.dataset_counts[dataset_name] += 1
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log individual dataset losses at the end of validation."""
        if not self.dataset_losses:
            return
            
        # Calculate average loss per dataset
        for dataset_name, total_loss in self.dataset_losses.items():
            count = self.dataset_counts[dataset_name]
            avg_loss = total_loss / count if count > 0 else 0.0
            
            # Log to wandb and console
            metric_name = f"val_loss_{dataset_name}"
            if hasattr(trainer.logger, 'log_metrics'):
                trainer.logger.log_metrics({metric_name: avg_loss}, step=trainer.global_step)
            
            # Print to console (only on rank 0)
            if self._is_rank_zero(trainer):
                print(f"Validation Loss - {dataset_name}: {avg_loss:.6f} (over {count} batches)")
    
    def _extract_loss(self, outputs: Any) -> Optional[float]:
        """Extract loss value from model outputs."""
        if isinstance(outputs, dict) and 'loss' in outputs:
            return outputs['loss'].item()
        elif hasattr(outputs, 'loss'):
            return outputs.loss.item()
        return None
    
    def _get_dataset_name(self, batch_idx: int, dataloader_idx: int) -> str:
        """Determine dataset name based on dataloader_idx or batch ratios."""
        # If we have multiple dataloaders, use dataloader_idx
        if len(self.dataset_names) > dataloader_idx:
            return self.dataset_names[dataloader_idx]
        else:
            # Estimate dataset assignment based on batch ratios
            cumulative_ratio = 0
            dataset_name = self.dataset_names[0] if self.dataset_names else "val_dataset_0"  # default
            
            for i, ratio in enumerate(self.batch_ratio):
                cumulative_ratio += ratio
                if batch_idx / max(1, self.total_batches) <= cumulative_ratio:
                    dataset_name = self.dataset_names[i] if i < len(self.dataset_names) else f"val_dataset_{i}"
                    break
            return dataset_name
    
    def _is_rank_zero(self, trainer: pl.Trainer) -> bool:
        """Check if this is the rank zero process."""
        if hasattr(trainer, 'is_global_zero') and trainer.is_global_zero:
            return True
        elif hasattr(trainer, 'local_rank') and trainer.local_rank == 0:
            return True
        return False
