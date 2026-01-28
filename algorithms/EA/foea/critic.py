"""
Pair-Set Critic Network for EA Algorithm

This module implements twin critics (Q1, Q2) that process state-action pairs
using pair token encoding and pooling mechanisms.

Author: FOenv Team
Date: 2026-01-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from algorithms.EA.foea.actor import masked_mean


class PairSetCritic(nn.Module):
    """
    Pair-Set Critic Network (Single Q Network)
    
    Architecture:
        1. Pair Token Encoder: ψ_Q([x_j, a_j, g, emb]) → v_j
        2. Pooling: v_bar = masked_mean(v, mask)
        3. Per-device Q Head: ρ_dev(v_j) → q_j
        4. Global Q Head: ρ_glob([v_bar, g]) → q_glob
        5. Final Q: masked_mean(q_j, mask) + q_glob
    
    Input:
        - g: [B, g_dim] Global features
        - X: [B, N_max, x_dim] Device states
        - A: [B, N_max, p] Actions
        - mask: [B, N_max] Active device mask
        - manager_id: [B] Manager IDs
    
    Output:
        - Q: [B, 1] Q value
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
        Initialize Pair-Set Critic
        
        Args:
            x_dim: Device state dimension
            g_dim: Global feature dimension
            p: Action dimension per device
            N_max: Maximum number of devices
            num_managers: Number of managers
            emb_dim: Manager ID embedding dimension
            token_dim: Token dimension
            hidden_dim: Hidden layer dimension
            activation: Activation function
        """
        super(PairSetCritic, self).__init__()
        
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
        
        # Pair Token Encoder ψ_Q: [x_j, a_j, g, emb] → v_j
        # Enhanced capacity: 3-layer MLP with doubled hidden dim for better expressiveness
        encoder_input_dim = x_dim + p + g_dim + emb_dim
        encoder_hidden_dim = hidden_dim * 2  # 256 → 512 for enhanced capacity
        self.pair_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, encoder_hidden_dim),
            self.activation,
            nn.Linear(encoder_hidden_dim, encoder_hidden_dim),  # Added 3rd layer
            self.activation,
            nn.Linear(encoder_hidden_dim, token_dim),
            self.activation,
        )
        
        # Per-device Q Head ρ_dev: v_j → q_j (enhanced capacity)
        q_head_hidden = hidden_dim * 2  # 256 → 512 for enhanced capacity
        self.device_q_head = nn.Sequential(
            nn.Linear(token_dim, q_head_hidden),
            self.activation,
            nn.Linear(q_head_hidden, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1),
        )
        
        # Global Q Head ρ_glob: [v_bar, g] → q_glob (enhanced capacity)
        glob_input_dim = token_dim + g_dim
        self.global_q_head = nn.Sequential(
            nn.Linear(glob_input_dim, q_head_hidden),
            self.activation,
            nn.Linear(q_head_hidden, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        g: torch.Tensor,
        X: torch.Tensor,
        A: torch.Tensor,
        mask: torch.Tensor,
        manager_id: torch.Tensor,
        return_per_device: bool = False
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            g: [B, g_dim] Global features
            X: [B, N_max, x_dim] Device states
            A: [B, N_max, p] Actions
            mask: [B, N_max] Active device mask
            manager_id: [B] Manager IDs
            return_per_device: If True, also return per-device Q values
        
        Returns:
            Q: [B, 1] Q value
            (optional) q_dev: [B, N_max] Per-device Q values (if return_per_device=True)
        """
        B = g.size(0)
        N_max = X.size(1)
        
        # Get manager embedding
        manager_emb = self.manager_embedding(manager_id)  # [B, emb_dim]
        
        # Expand g and manager_emb to all devices
        g_expanded = g.unsqueeze(1).expand(B, N_max, self.g_dim)  # [B, N_max, g_dim]
        manager_emb_expanded = manager_emb.unsqueeze(1).expand(B, N_max, self.emb_dim)  # [B, N_max, emb_dim]
        
        # Pair Token Encoder: concatenate [x_j, a_j, g, emb]
        pair_input = torch.cat([X, A, g_expanded, manager_emb_expanded], dim=-1)  # [B, N_max, x_dim + p + g_dim + emb_dim]
        v = self.pair_encoder(pair_input)  # [B, N_max, token_dim]
        
        # Pooling: v_bar = masked_mean(v, mask)
        v_bar = masked_mean(v, mask, dim=1, keepdim=False)  # [B, token_dim]
        
        # Per-device Q values
        q_dev = self.device_q_head(v)  # [B, N_max, 1]
        q_dev = q_dev.squeeze(-1)  # [B, N_max]
        
        # Masked mean of per-device Q values
        q_dev_expanded = q_dev.unsqueeze(-1)  # [B, N_max, 1]
        q_dev_mean = masked_mean(q_dev_expanded, mask, dim=1, keepdim=True)  # [B, 1]
        
        # Global Q value
        glob_input = torch.cat([v_bar, g], dim=-1)  # [B, token_dim + g_dim]
        q_glob = self.global_q_head(glob_input)  # [B, 1]
        
        # Final Q value
        Q = q_dev_mean + q_glob  # [B, 1]
        
        if return_per_device:
            return Q, q_dev
        return Q


class TwinCritics(nn.Module):
    """
    Twin Critics (Q1 and Q2) for MATD3-style training
    
    Uses two independent Pair-Set Critics and returns both Q values
    or their minimum for target computation.
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
        Initialize Twin Critics
        
        Args:
            Same as PairSetCritic
        """
        super(TwinCritics, self).__init__()
        
        # Q1 network
        self.Q1 = PairSetCritic(
            x_dim=x_dim,
            g_dim=g_dim,
            p=p,
            N_max=N_max,
            num_managers=num_managers,
            emb_dim=emb_dim,
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            activation=activation,
        )
        
        # Q2 network
        self.Q2 = PairSetCritic(
            x_dim=x_dim,
            g_dim=g_dim,
            p=p,
            N_max=N_max,
            num_managers=num_managers,
            emb_dim=emb_dim,
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            activation=activation,
        )
    
    def forward(
        self,
        g: torch.Tensor,
        X: torch.Tensor,
        A: torch.Tensor,
        mask: torch.Tensor,
        manager_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both critics
        
        Args:
            g: [B, g_dim] Global features
            X: [B, N_max, x_dim] Device states
            A: [B, N_max, p] Actions
            mask: [B, N_max] Active device mask
            manager_id: [B] Manager IDs
        
        Returns:
            Q1, Q2: Both Q values [B, 1]
        """
        Q1 = self.Q1(g, X, A, mask, manager_id)
        Q2 = self.Q2(g, X, A, mask, manager_id)
        return Q1, Q2
    
    def Q1_forward(
        self,
        g: torch.Tensor,
        X: torch.Tensor,
        A: torch.Tensor,
        mask: torch.Tensor,
        manager_id: torch.Tensor,
        return_per_device: bool = False
    ) -> torch.Tensor:
        """Forward pass through Q1 only (for actor update)
        
        Args:
            g: [B, g_dim] Global features
            X: [B, N_max, x_dim] Device states
            A: [B, N_max, p] Actions
            mask: [B, N_max] Active device mask
            manager_id: [B] Manager IDs
            return_per_device: If True, also return per-device Q values
        
        Returns:
            Q1: [B, 1] Q1 value
            (optional) q_dev: [B, N_max] Per-device Q values (if return_per_device=True)
        """
        return self.Q1(g, X, A, mask, manager_id, return_per_device=return_per_device)
    
    def min_Q(
        self,
        g: torch.Tensor,
        X: torch.Tensor,
        A: torch.Tensor,
        mask: torch.Tensor,
        manager_id: torch.Tensor
    ) -> torch.Tensor:
        """Return minimum of Q1 and Q2 (for target computation)"""
        Q1, Q2 = self.forward(g, X, A, mask, manager_id)
        return torch.min(Q1, Q2)


def test_critic():
    """Test Pair-Set Critic and Twin Critics"""
    print("=== Testing Pair-Set Critic ===\n")
    
    # Parameters
    B = 4  # Batch size
    x_dim = 6
    g_dim = 50
    p = 5
    N_max = 60
    num_managers = 4
    
    # Create single critic
    critic = PairSetCritic(
        x_dim=x_dim,
        g_dim=g_dim,
        p=p,
        N_max=N_max,
        num_managers=num_managers,
    )
    
    print(f"Single Critic created with {sum(p.numel() for p in critic.parameters())} parameters")
    
    # Test inputs
    g = torch.randn(B, g_dim)
    X = torch.randn(B, N_max, x_dim)
    A = torch.randn(B, N_max, p)
    
    # Create mask
    mask = torch.zeros(B, N_max)
    mask[0, :20] = 1
    mask[1, :30] = 1
    mask[2, :15] = 1
    mask[3, :40] = 1
    
    # Apply mask to actions
    A = A * mask.unsqueeze(-1)
    
    manager_id = torch.tensor([0, 1, 2, 3])
    
    # Forward pass
    Q = critic(g, X, A, mask, manager_id)
    
    print(f"\nInput shapes:")
    print(f"  g: {g.shape}")
    print(f"  X: {X.shape}")
    print(f"  A: {A.shape}")
    print(f"  mask: {mask.shape}")
    
    print(f"\nOutput shape: {Q.shape}")
    print(f"Q values: {Q.squeeze().tolist()}")
    
    # Test Twin Critics
    print(f"\n=== Testing Twin Critics ===\n")
    
    twin_critics = TwinCritics(
        x_dim=x_dim,
        g_dim=g_dim,
        p=p,
        N_max=N_max,
        num_managers=num_managers,
    )
    
    print(f"Twin Critics created with {sum(p.numel() for p in twin_critics.parameters())} parameters")
    
    # Forward pass
    Q1, Q2 = twin_critics(g, X, A, mask, manager_id)
    Q_min = twin_critics.min_Q(g, X, A, mask, manager_id)
    Q1_only = twin_critics.Q1_forward(g, X, A, mask, manager_id)
    
    print(f"\nOutput shapes:")
    print(f"  Q1: {Q1.shape}")
    print(f"  Q2: {Q2.shape}")
    print(f"  Q_min: {Q_min.shape}")
    print(f"  Q1_only: {Q1_only.shape}")
    
    print(f"\nQ values:")
    for i in range(B):
        print(f"  Batch {i}: Q1={Q1[i].item():.3f}, Q2={Q2[i].item():.3f}, min={Q_min[i].item():.3f}")
    
    # Verify Q_min is actually minimum
    Q_min_manual = torch.min(Q1, Q2)
    assert torch.allclose(Q_min, Q_min_manual), "Q_min computation error"
    print(f"\n✓ Q_min computation correct")
    
    # Verify Q1_only matches Q1
    assert torch.allclose(Q1_only, Q1), "Q1_forward computation error"
    print(f"✓ Q1_forward computation correct")
    
    print("\n=== Critic tests passed ===")


if __name__ == "__main__":
    test_critic()
