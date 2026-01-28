"""
Masking Utilities: Prevent gradient leakage from padding slots

Author: FOenv Team
Date: 2026-01-13
"""

import numpy as np
import torch
from typing import Union


def masked_sum(
    x: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
    dim: int = -1,
    keepdim: bool = False
) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute sum over masked dimensions.
    
    Args:
        x: Input tensor/array
        mask: Binary mask (1.0 for valid, 0.0 for invalid)
        dim: Dimension to sum over
        keepdim: Whether to keep the reduced dimension
        
    Returns:
        Masked sum
    """
    if isinstance(x, torch.Tensor):
        masked_x = x * mask
        return masked_x.sum(dim=dim, keepdim=keepdim)
    else:
        # NumPy
        masked_x = x * mask
        return masked_x.sum(axis=dim, keepdims=keepdim)


def masked_mean(
    x: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
    dim: int = -1,
    keepdim: bool = False,
    eps: float = 1e-8
) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute mean over masked dimensions.
    
    Args:
        x: Input tensor/array
        mask: Binary mask (1.0 for valid, 0.0 for invalid)
        dim: Dimension to average over
        keepdim: Whether to keep the reduced dimension
        eps: Small value to prevent division by zero
        
    Returns:
        Masked mean
    """
    if isinstance(x, torch.Tensor):
        masked_x = x * mask
        sum_val = masked_x.sum(dim=dim, keepdim=keepdim)
        count = mask.sum(dim=dim, keepdim=keepdim).clamp(min=eps)
        return sum_val / count
    else:
        # NumPy
        masked_x = x * mask
        sum_val = masked_x.sum(axis=dim, keepdims=keepdim)
        count = np.maximum(mask.sum(axis=dim, keepdims=keepdim), eps)
        return sum_val / count


def repeat_mask_for_action(
    device_mask: Union[torch.Tensor, np.ndarray],
    p: int
) -> Union[torch.Tensor, np.ndarray]:
    """
    Repeat device mask for action dimensions.
    
    For flattened actions [N_max * p], we need an action mask that repeats
    each device mask element p times.
    
    Args:
        device_mask: Device mask [N_max] or [batch, N_max]
        p: Action dimension per device
        
    Returns:
        action_mask: [N_max * p] or [batch, N_max * p]
        
    Example:
        device_mask = [1, 0, 1]  # 3 devices, 2 active
        p = 2
        action_mask = [1, 1, 0, 0, 1, 1]  # 6 action dims
    """
    if isinstance(device_mask, torch.Tensor):
        if device_mask.ndim == 1:
            # [N_max] -> [N_max, p] -> [N_max * p]
            action_mask = device_mask.unsqueeze(-1).repeat(1, p).reshape(-1)
        else:
            # [batch, N_max] -> [batch, N_max, p] -> [batch, N_max * p]
            action_mask = device_mask.unsqueeze(-1).repeat(1, 1, p).reshape(device_mask.shape[0], -1)
        return action_mask
    else:
        # NumPy
        if device_mask.ndim == 1:
            action_mask = np.repeat(device_mask, p)
        else:
            action_mask = np.repeat(device_mask, p, axis=-1)
        return action_mask


def expand_mask_for_features(
    mask: Union[torch.Tensor, np.ndarray],
    feature_dim: int
) -> Union[torch.Tensor, np.ndarray]:
    """
    Expand mask to broadcast with feature dimension.
    
    Args:
        mask: Device mask [N_max] or [batch, N_max]
        feature_dim: Feature dimension per device
        
    Returns:
        expanded_mask: [N_max, feature_dim] or [batch, N_max, feature_dim]
    """
    if isinstance(mask, torch.Tensor):
        return mask.unsqueeze(-1).expand(*mask.shape, feature_dim)
    else:
        # NumPy
        return np.expand_dims(mask, axis=-1)


def apply_action_mask(
    actions: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
    p: int
) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply mask to actions (zero out inactive device actions).
    
    Args:
        actions: Actions [N_max, p] or [batch, N_max, p] or flattened
        mask: Device mask [N_max] or [batch, N_max]
        p: Action dimension per device
        
    Returns:
        Masked actions (same shape as input)
    """
    if isinstance(actions, torch.Tensor):
        if actions.ndim == 2 and actions.shape[-1] == p:
            # [N_max, p]
            mask_exp = mask.unsqueeze(-1)  # [N_max, 1]
            return actions * mask_exp
        elif actions.ndim == 3 and actions.shape[-1] == p:
            # [batch, N_max, p]
            mask_exp = mask.unsqueeze(-1)  # [batch, N_max, 1]
            return actions * mask_exp
        elif actions.ndim == 1:
            # Flattened [N_max * p]
            action_mask = repeat_mask_for_action(mask, p)
            return actions * action_mask
        elif actions.ndim == 2 and actions.shape[-1] != p:
            # Flattened [batch, N_max * p]
            action_mask = repeat_mask_for_action(mask, p)
            return actions * action_mask
        else:
            raise ValueError(f"Unsupported action shape: {actions.shape}")
    else:
        # NumPy - same logic
        if actions.ndim == 2 and actions.shape[-1] == p:
            mask_exp = mask[:, np.newaxis]
            return actions * mask_exp
        elif actions.ndim == 3 and actions.shape[-1] == p:
            mask_exp = mask[:, :, np.newaxis]
            return actions * mask_exp
        else:
            action_mask = repeat_mask_for_action(mask, p)
            return actions * action_mask
