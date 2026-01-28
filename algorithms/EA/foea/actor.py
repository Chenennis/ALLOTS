"""
Set-to-Set Actor Network for EA Algorithm

This module implements the actor network that processes variable-size device sets
using token encoding, pooling, and refinement mechanisms.

Author: FOenv Team
Date: 2026-01-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, keepdim: bool = False) -> torch.Tensor:
    """
    Compute masked mean along specified dimension
    
    Args:
        tensor: Input tensor [B, N, D]
        mask: Binary mask [B, N]
        dim: Dimension to reduce
        keepdim: Keep reduced dimension
    
    Returns:
        Masked mean tensor
    """
    # Expand mask to match tensor dimensions
    mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
    
    # Apply mask
    masked_tensor = tensor * mask_expanded
    
    # Sum and divide by number of active elements
    sum_tensor = torch.sum(masked_tensor, dim=dim, keepdim=keepdim)
    count = torch.sum(mask_expanded, dim=dim, keepdim=keepdim).clamp(min=1.0)  # Avoid division by zero
    
    return sum_tensor / count


class SetToSetActor(nn.Module):
    """
    Set-to-Set Actor Network
    
    Architecture:
        1. Token Encoder: φ_k([x_j, g, emb(manager_id)]) → u_j
        2. Pooling: u_bar = masked_mean(u, mask)
        3. Refiner: φ_r([u_j, u_bar]) → h_j
        4. Action Head: head(h_j) → a_j
        5. Mask Application: a_j *= mask
    
    Input:
        - g: [B, g_dim] Global features
        - X: [B, N_max, x_dim] Device states (padded)
        - mask: [B, N_max] Active device mask
        - manager_id: [B] or int Manager identifiers
    
    Output:
        - A: [B, N_max, p] Actions (masked, inactive slots are zero)
    """
    
    def __init__(
        self,
        x_dim: int = 6,
        g_dim: int = 50,
        p: int = 5,
        N_max: int = 60,
        num_managers: int = 10,
        emb_dim: int = 16,
        token_dim: int = 128,
        hidden_dim: int = 256,
        activation: str = 'relu',
    ):
        """
        Initialize Set-to-Set Actor
        
        Args:
            x_dim: Device state dimension
            g_dim: Global feature dimension
            p: Action dimension per device
            N_max: Maximum number of devices
            num_managers: Number of managers (for embedding)
            emb_dim: Manager ID embedding dimension
            token_dim: Token dimension
            hidden_dim: Hidden layer dimension
            activation: Activation function ('relu', 'tanh')
        """
        super(SetToSetActor, self).__init__()
        
        self.x_dim = x_dim
        self.g_dim = g_dim
        self.p = p
        self.N_max = N_max
        self.num_managers = num_managers
        self.emb_dim = emb_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Manager ID embedding
        self.manager_embedding = nn.Embedding(num_managers, emb_dim)
        
        # Token Encoder φ_k: [x_j, g, emb] → u_j
        encoder_input_dim = x_dim + g_dim + emb_dim
        self.token_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, token_dim),
            self.activation,
        )
        
        # Refiner φ_r: [u_j, u_bar] → h_j
        refiner_input_dim = token_dim * 2
        self.refiner = nn.Sequential(
            nn.Linear(refiner_input_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
        )
        
        # Action Head: h_j → a_j
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, p),
            nn.Tanh(),  # Bounded actions in [-1, 1]
        )
    
    def forward(
        self,
        g: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor,
        manager_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            g: [B, g_dim] Global features
            X: [B, N_max, x_dim] Device states
            mask: [B, N_max] Active device mask (0 or 1)
            manager_id: [B] Manager IDs
        
        Returns:
            A: [B, N_max, p] Actions (masked)
        """
        B = g.size(0)
        N_max = X.size(1)
        
        # Get manager embedding
        manager_emb = self.manager_embedding(manager_id)  # [B, emb_dim]
        
        # Expand g and manager_emb to all devices
        g_expanded = g.unsqueeze(1).expand(B, N_max, self.g_dim)  # [B, N_max, g_dim]
        manager_emb_expanded = manager_emb.unsqueeze(1).expand(B, N_max, self.emb_dim)  # [B, N_max, emb_dim]
        
        # Token Encoder: concatenate [x_j, g, emb]
        encoder_input = torch.cat([X, g_expanded, manager_emb_expanded], dim=-1)  # [B, N_max, x_dim + g_dim + emb_dim]
        u = self.token_encoder(encoder_input)  # [B, N_max, token_dim]
        
        # Pooling: u_bar = masked_mean(u, mask)
        u_bar = masked_mean(u, mask, dim=1, keepdim=False)  # [B, token_dim]
        
        # Expand u_bar to all devices
        u_bar_expanded = u_bar.unsqueeze(1).expand(B, N_max, self.token_dim)  # [B, N_max, token_dim]
        
        # Refiner: concatenate [u_j, u_bar]
        refiner_input = torch.cat([u, u_bar_expanded], dim=-1)  # [B, N_max, token_dim * 2]
        h = self.refiner(refiner_input)  # [B, N_max, hidden_dim]
        
        # Action Head
        a = self.action_head(h)  # [B, N_max, p]
        
        # Apply mask (zero out inactive slots)
        a = a * mask.unsqueeze(-1)  # [B, N_max, p]
        
        return a
    
    def get_actions(
        self,
        g: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor,
        manager_id: torch.Tensor,
        noise_scale: float = 0.0
    ) -> torch.Tensor:
        """
        Get actions with optional exploration noise
        
        Args:
            g: [B, g_dim] Global features
            X: [B, N_max, x_dim] Device states
            mask: [B, N_max] Active device mask
            manager_id: [B] Manager IDs
            noise_scale: Exploration noise scale
        
        Returns:
            A: [B, N_max, p] Actions with noise (masked and clipped to [-1, 1])
        """
        a = self.forward(g, X, mask, manager_id)
        
        if noise_scale > 0.0:
            # Add Gaussian noise only to active slots
            noise = torch.randn_like(a) * noise_scale
            noise = noise * mask.unsqueeze(-1)  # Mask the noise
            a = a + noise
            
            # Clip to [-1, 1]
            a = torch.clamp(a, -1.0, 1.0)
            
            # Re-apply mask (in case clipping affected masked slots)
            a = a * mask.unsqueeze(-1)
        
        return a


def test_actor():
    """Test Set-to-Set Actor"""
    print("=== Testing Set-to-Set Actor ===\n")
    
    # Parameters
    B = 4  # Batch size
    x_dim = 6
    g_dim = 50
    p = 5
    N_max = 60
    num_managers = 4
    
    # Create actor
    actor = SetToSetActor(
        x_dim=x_dim,
        g_dim=g_dim,
        p=p,
        N_max=N_max,
        num_managers=num_managers,
    )
    
    print(f"Actor created with {sum(p.numel() for p in actor.parameters())} parameters")
    
    # Test forward pass
    g = torch.randn(B, g_dim)
    X = torch.randn(B, N_max, x_dim)
    
    # Create mask with varying active devices
    mask = torch.zeros(B, N_max)
    mask[0, :20] = 1  # 20 active devices
    mask[1, :30] = 1  # 30 active devices
    mask[2, :15] = 1  # 15 active devices
    mask[3, :40] = 1  # 40 active devices
    
    manager_id = torch.tensor([0, 1, 2, 3])
    
    # Forward pass
    A = actor(g, X, mask, manager_id)
    
    print(f"\nInput shapes:")
    print(f"  g: {g.shape}")
    print(f"  X: {X.shape}")
    print(f"  mask: {mask.shape}")
    print(f"  manager_id: {manager_id.shape}")
    
    print(f"\nOutput shape: {A.shape}")
    print(f"Output range: [{A.min():.3f}, {A.max():.3f}]")
    
    # Check masking
    print(f"\nMasking check:")
    for i in range(B):
        n_active = int(mask[i].sum().item())
        active_actions = A[i, :n_active]
        inactive_actions = A[i, n_active:]
        
        print(f"  Batch {i}: {n_active} active devices")
        print(f"    Active actions norm: {active_actions.norm():.3f}")
        print(f"    Inactive actions norm: {inactive_actions.norm():.6f} (should be ~0)")
    
    # Test with noise
    print(f"\nTesting with exploration noise...")
    A_noisy = actor.get_actions(g, X, mask, manager_id, noise_scale=0.1)
    print(f"  Noisy actions shape: {A_noisy.shape}")
    print(f"  Noisy actions range: [{A_noisy.min():.3f}, {A_noisy.max():.3f}]")
    print(f"  Difference from deterministic: {(A - A_noisy).abs().mean():.4f}")
    
    print("\n=== Actor test passed ===")


if __name__ == "__main__":
    test_actor()
