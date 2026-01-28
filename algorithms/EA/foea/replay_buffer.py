"""
Churn-Aware Replay Buffer for EA Algorithm

This module implements a replay buffer that stores transitions with masks,
supporting device churn by tracking mask and mask_next separately.

Author: FOenv Team
Date: 2026-01-12
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from collections import deque
import random


class ChurnAwareReplayBuffer:
    """
    Replay Buffer for EA Algorithm with Churn Support
    
    Stores transitions with the following structure:
        - manager_id: int
        - g: [g_dim] Global features
        - X: [N_max, x_dim] Device states (padded)
        - mask: [N_max] Active device mask
        - A: [N_max, p] Actions (padded)
        - r: float Reward
        - g_next: [g_dim] Next global features
        - X_next: [N_max, x_dim] Next device states
        - mask_next: [N_max] Next active mask (IMPORTANT for churn!)
        - done: bool Terminal flag
    
    Key Feature: Stores mask_next separately to support churn-consistent TD targets
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        x_dim: int = 6,
        g_dim: int = 50,
        p: int = 5,
        N_max: int = 60,
    ):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum buffer size
            x_dim: Device state dimension
            g_dim: Global feature dimension
            p: Action dimension per device
            N_max: Maximum number of devices
        """
        self.capacity = capacity
        self.x_dim = x_dim
        self.g_dim = g_dim
        self.p = p
        self.N_max = N_max
        
        # Use deque for automatic capacity management
        self.buffer = deque(maxlen=capacity)
        
        self.ptr = 0
        self.size = 0
    
    def add(
        self,
        manager_id: int,
        g: np.ndarray,
        X: np.ndarray,
        mask: np.ndarray,
        A: np.ndarray,
        r: float,
        g_next: np.ndarray,
        X_next: np.ndarray,
        mask_next: np.ndarray,
        done: bool
    ):
        """
        Add a transition to the buffer
        
        Args:
            manager_id: Manager identifier
            g: [g_dim] Global features
            X: [N_max, x_dim] Device states
            mask: [N_max] Active device mask
            A: [N_max, p] Actions
            r: Reward
            g_next: [g_dim] Next global features
            X_next: [N_max, x_dim] Next device states
            mask_next: [N_max] Next active mask
            done: Terminal flag
        """
        transition = {
            'manager_id': manager_id,
            'g': g.copy(),
            'X': X.copy(),
            'mask': mask.copy(),
            'A': A.copy(),
            'r': float(r),
            'g_next': g_next.copy(),
            'X_next': X_next.copy(),
            'mask_next': mask_next.copy(),
            'done': bool(done),
        }
        
        self.buffer.append(transition)
        self.size = len(self.buffer)
    
    def sample(self, batch_size: int, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
            device: PyTorch device ('cpu' or 'cuda')
        
        Returns:
            Dictionary of batched tensors
        """
        if self.size < batch_size:
            batch_size = self.size
        
        # Random sampling
        indices = random.sample(range(self.size), batch_size)
        batch = [self.buffer[i] for i in indices]
        
        # Stack into tensors
        manager_ids = torch.tensor([t['manager_id'] for t in batch], dtype=torch.long, device=device)
        g = torch.tensor(np.stack([t['g'] for t in batch]), dtype=torch.float32, device=device)
        X = torch.tensor(np.stack([t['X'] for t in batch]), dtype=torch.float32, device=device)
        mask = torch.tensor(np.stack([t['mask'] for t in batch]), dtype=torch.float32, device=device)
        A = torch.tensor(np.stack([t['A'] for t in batch]), dtype=torch.float32, device=device)
        r = torch.tensor([t['r'] for t in batch], dtype=torch.float32, device=device).unsqueeze(-1)
        g_next = torch.tensor(np.stack([t['g_next'] for t in batch]), dtype=torch.float32, device=device)
        X_next = torch.tensor(np.stack([t['X_next'] for t in batch]), dtype=torch.float32, device=device)
        mask_next = torch.tensor(np.stack([t['mask_next'] for t in batch]), dtype=torch.float32, device=device)
        done = torch.tensor([t['done'] for t in batch], dtype=torch.float32, device=device).unsqueeze(-1)
        
        return {
            'manager_id': manager_ids,
            'g': g,
            'X': X,
            'mask': mask,
            'A': A,
            'r': r,
            'g_next': g_next,
            'X_next': X_next,
            'mask_next': mask_next,
            'done': done,
        }
    
    def __len__(self) -> int:
        """Return current buffer size"""
        return self.size
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.size = 0
        self.ptr = 0
    
    def get_statistics(self) -> Dict:
        """Get buffer statistics"""
        if self.size == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'avg_reward': 0.0,
                'std_reward': 0.0,
                'done_ratio': 0.0,
                'avg_active_devices': 0.0,
                'avg_active_devices_next': 0.0,
                'churn_events': 0,
                'churn_ratio': 0.0,
            }
        
        # Compute statistics
        rewards = [t['r'] for t in self.buffer]
        done_count = sum(t['done'] for t in self.buffer)
        
        # Count active devices
        active_counts = [t['mask'].sum() for t in self.buffer]
        active_counts_next = [t['mask_next'].sum() for t in self.buffer]
        
        # Churn statistics
        churn_events = sum(
            (t['mask'].sum() != t['mask_next'].sum()) for t in self.buffer
        )
        
        return {
            'size': self.size,
            'capacity': self.capacity,
            'utilization': self.size / self.capacity,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'done_ratio': done_count / self.size,
            'avg_active_devices': np.mean(active_counts),
            'avg_active_devices_next': np.mean(active_counts_next),
            'churn_events': churn_events,
            'churn_ratio': churn_events / self.size,
        }


def test_replay_buffer():
    """Test Churn-Aware Replay Buffer"""
    print("=== Testing Churn-Aware Replay Buffer ===\n")
    
    # Parameters
    capacity = 1000
    x_dim = 6
    g_dim = 50
    p = 5
    N_max = 60
    
    # Create buffer
    buffer = ChurnAwareReplayBuffer(
        capacity=capacity,
        x_dim=x_dim,
        g_dim=g_dim,
        p=p,
        N_max=N_max,
    )
    
    print(f"Buffer created with capacity {capacity}")
    print(f"Initial size: {len(buffer)}\n")
    
    # Add some transitions
    print("Adding 500 transitions...")
    for i in range(500):
        manager_id = i % 4
        g = np.random.randn(g_dim)
        X = np.random.randn(N_max, x_dim)
        
        # Random mask
        n_active = np.random.randint(10, 50)
        mask = np.zeros(N_max)
        mask[:n_active] = 1
        
        A = np.random.randn(N_max, p)
        A = A * mask[:, None]  # Mask actions
        
        r = np.random.randn()
        g_next = np.random.randn(g_dim)
        X_next = np.random.randn(N_max, x_dim)
        
        # Simulate churn: 10% chance of device count change
        if np.random.rand() < 0.1:
            n_active_next = np.random.randint(10, 50)
        else:
            n_active_next = n_active
        
        mask_next = np.zeros(N_max)
        mask_next[:n_active_next] = 1
        
        done = (i % 24 == 23)  # Episode ends every 24 steps
        
        buffer.add(manager_id, g, X, mask, A, r, g_next, X_next, mask_next, done)
    
    print(f"Buffer size after adding: {len(buffer)}")
    
    # Get statistics
    stats = buffer.get_statistics()
    print(f"\nBuffer Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test sampling
    print(f"\nSampling batches...")
    batch_size = 32
    
    for i in range(3):
        batch = buffer.sample(batch_size, device='cpu')
        
        print(f"\nBatch {i+1}:")
        print(f"  manager_id shape: {batch['manager_id'].shape}")
        print(f"  g shape: {batch['g'].shape}")
        print(f"  X shape: {batch['X'].shape}")
        print(f"  mask shape: {batch['mask'].shape}")
        print(f"  A shape: {batch['A'].shape}")
        print(f"  r shape: {batch['r'].shape}")
        print(f"  g_next shape: {batch['g_next'].shape}")
        print(f"  X_next shape: {batch['X_next'].shape}")
        print(f"  mask_next shape: {batch['mask_next'].shape}")
        print(f"  done shape: {batch['done'].shape}")
        
        # Check churn in this batch
        mask_sum = batch['mask'].sum(dim=1)
        mask_next_sum = batch['mask_next'].sum(dim=1)
        churn_in_batch = (mask_sum != mask_next_sum).sum().item()
        print(f"  Churn events in batch: {churn_in_batch}/{batch_size}")
    
    # Test buffer overflow
    print(f"\nTesting buffer overflow...")
    print(f"Adding 600 more transitions (total 1100, capacity {capacity})...")
    for i in range(600):
        manager_id = i % 4
        g = np.random.randn(g_dim)
        X = np.random.randn(N_max, x_dim)
        mask = np.ones(N_max) * (np.random.rand(N_max) > 0.5)
        A = np.random.randn(N_max, p) * mask[:, None]
        r = np.random.randn()
        g_next = np.random.randn(g_dim)
        X_next = np.random.randn(N_max, x_dim)
        mask_next = np.ones(N_max) * (np.random.rand(N_max) > 0.5)
        done = False
        
        buffer.add(manager_id, g, X, mask, A, r, g_next, X_next, mask_next, done)
    
    print(f"Buffer size after overflow: {len(buffer)} (should be {capacity})")
    assert len(buffer) == capacity, "Buffer overflow handling failed"
    print(f"✓ Buffer correctly maintains capacity limit")
    
    # Test clear
    print(f"\nTesting clear...")
    buffer.clear()
    print(f"Buffer size after clear: {len(buffer)}")
    assert len(buffer) == 0, "Buffer clear failed"
    print(f"✓ Buffer clear successful")
    
    print("\n=== Replay buffer tests passed ===")


if __name__ == "__main__":
    test_replay_buffer()
