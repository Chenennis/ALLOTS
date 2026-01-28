"""
FOgym Adapter for EA Algorithm

This module provides conversion functions between FOgym's observation/action formats
and EA's set-based representations:
- Extract per-device states from Manager (bypassing aggregated features)
- Extract global features from environment
- Convert EA actions to FOgym format

Author: FOenv Team
Date: 2026-01-12
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def pad_device_state(raw_state: np.ndarray, device_type: str, x_dim: int = 6) -> np.ndarray:
    """
    Pad device state to unified x_dim
    
    Device dimensions:
    - Battery: 4D [soc, max_charge_power, max_discharge_power, health]
    - Dishwasher: 6D [deployed, running, completed, progress, urgency, energy]
    - EV: 3D [soc, is_connected, urgency]
    - HeatPump: 3D [current_temp, target_temp, comfort]
    - PV: 2D [current_power, forecast_power]
    
    Padding strategy: original features first, then pad with zeros
    
    Args:
        raw_state: Original device state from mdp.get_state_features()
        device_type: Device type string
        x_dim: Target dimension (default 6)
    
    Returns:
        padded_state: [x_dim] numpy array
    """
    raw_dim = len(raw_state)
    
    if raw_dim > x_dim:
        logger.warning(f"Device state dim {raw_dim} exceeds x_dim {x_dim}, truncating")
        return raw_state[:x_dim]
    
    # Create padded state
    padded_state = np.zeros(x_dim, dtype=np.float32)
    padded_state[:raw_dim] = raw_state
    
    return padded_state


def extract_device_states_from_manager(
    manager,
    N_max: int = 60,
    x_dim: int = 6
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract per-device states directly from Manager
    
    This bypasses FOgym's aggregated features and gets raw device states
    from manager.device_mdps, which is essential for EA's set-based architecture.
    
    Args:
        manager: ManagerAgent instance
        N_max: Maximum number of devices (for padding)
        x_dim: Device state dimension
    
    Returns:
        X: [N_max, x_dim] - Padded device states
        mask: [N_max] - Active device mask (1 for active, 0 for padding)
        device_ids: List of device IDs (for reference)
    """
    device_states = []
    device_ids = []
    
    # Iterate through all controllable devices
    for device_id in manager.controllable_devices:
        mdp = manager.device_mdps[device_id]
        device_type = manager.device_types[device_id]
        
        # Get raw state from MDP
        raw_state = mdp.get_state_features()
        
        # Pad to x_dim
        padded_state = pad_device_state(raw_state, device_type, x_dim)
        
        device_states.append(padded_state)
        device_ids.append(device_id)
    
    n_devices = len(device_states)
    
    # Initialize padded arrays
    X = np.zeros((N_max, x_dim), dtype=np.float32)
    mask = np.zeros(N_max, dtype=np.float32)
    
    # Fill active slots
    if n_devices > 0:
        if n_devices > N_max:
            logger.warning(f"Manager has {n_devices} devices, exceeds N_max={N_max}, truncating")
            n_devices = N_max
        
        X[:n_devices] = np.array(device_states[:n_devices])
        mask[:n_devices] = 1.0
    
    return X, mask, device_ids


def extract_time_features(current_time: datetime) -> np.ndarray:
    """
    Extract time-based features
    
    Features:
    - hour_sin, hour_cos (2D)
    - day_of_week_sin, day_of_week_cos (2D)
    - day_of_month (1D, normalized)
    - month_sin, month_cos (2D)
    - is_weekend (1D)
    - is_peak_hour (1D, morning/evening peaks)
    
    Total: 9D
    """
    hour = current_time.hour
    day_of_week = current_time.weekday()  # 0=Monday, 6=Sunday
    day_of_month = current_time.day
    month = current_time.month
    
    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Normalized day of month
    day_norm = day_of_month / 31.0
    
    # Binary features
    is_weekend = 1.0 if day_of_week >= 5 else 0.0
    is_peak_hour = 1.0 if (7 <= hour <= 9) or (17 <= hour <= 20) else 0.0
    
    return np.array([
        hour_sin, hour_cos,
        dow_sin, dow_cos,
        day_norm,
        month_sin, month_cos,
        is_weekend,
        is_peak_hour
    ], dtype=np.float32)


def extract_price_features(env_state: Dict) -> np.ndarray:
    """
    Extract price-related features
    
    Features:
    - current_price (1D, normalized)
    - price_trend (1D, -1 to 1)
    - price_volatility (1D)
    - price_percentile (1D, 0 to 1, position in recent history)
    
    Total: 4D
    """
    current_price = env_state.get('price', 0.5)
    
    # Normalize price (assume range 0-1 EUR/kWh)
    price_norm = np.clip(current_price, 0.0, 1.0)
    
    # Price trend (from history if available)
    price_history = env_state.get('price_history', [current_price])
    if len(price_history) >= 2:
        price_trend = (price_history[-1] - price_history[-2]) / max(price_history[-2], 0.01)
        price_trend = np.clip(price_trend, -1.0, 1.0)
    else:
        price_trend = 0.0
    
    # Price volatility
    if len(price_history) >= 3:
        price_volatility = np.std(price_history[-3:]) / max(np.mean(price_history[-3:]), 0.01)
        price_volatility = np.clip(price_volatility, 0.0, 1.0)
    else:
        price_volatility = 0.0
    
    # Price percentile (position in recent history)
    if len(price_history) >= 10:
        price_percentile = np.mean(np.array(price_history[-10:]) <= current_price)
    else:
        price_percentile = 0.5
    
    return np.array([
        price_norm,
        price_trend,
        price_volatility,
        price_percentile
    ], dtype=np.float32)


def extract_weather_features(env_state: Dict) -> np.ndarray:
    """
    Extract weather-related features
    
    Features:
    - temperature (1D, normalized to 0-1, assume -10 to 40 C)
    - solar_irradiance (1D, normalized to 0-1, 0-1000 W/m^2)
    - wind_speed (1D, normalized to 0-1, 0-20 m/s)
    
    Total: 3D
    """
    temperature = env_state.get('temperature', 20.0)
    solar_irradiance = env_state.get('solar_irradiance', 500.0)
    wind_speed = env_state.get('wind_speed', 5.0)
    
    # Normalize
    temp_norm = np.clip((temperature + 10) / 50.0, 0.0, 1.0)
    solar_norm = np.clip(solar_irradiance / 1000.0, 0.0, 1.0)
    wind_norm = np.clip(wind_speed / 20.0, 0.0, 1.0)
    
    return np.array([
        temp_norm,
        solar_norm,
        wind_norm
    ], dtype=np.float32)


def extract_market_features(env_state: Dict) -> np.ndarray:
    """
    Extract market-related features
    
    Features:
    - total_demand (1D, normalized)
    - total_supply (1D, normalized)
    - market_clearing_price (1D, normalized)
    - imbalance (1D, demand - supply, normalized)
    
    Total: 4D
    """
    demand = env_state.get('total_demand', 1000.0)
    supply = env_state.get('total_supply', 1000.0)
    clearing_price = env_state.get('clearing_price', 0.5)
    
    # Normalize (assume max 10000 kW)
    demand_norm = np.clip(demand / 10000.0, 0.0, 1.0)
    supply_norm = np.clip(supply / 10000.0, 0.0, 1.0)
    clearing_price_norm = np.clip(clearing_price, 0.0, 1.0)
    
    # Market imbalance
    imbalance = demand - supply
    imbalance_norm = np.clip(imbalance / 5000.0, -1.0, 1.0)
    
    return np.array([
        demand_norm,
        supply_norm,
        clearing_price_norm,
        imbalance_norm
    ], dtype=np.float32)


def extract_manager_features(manager) -> np.ndarray:
    """
    Extract manager-specific features
    
    Features:
    - total_devices (1D, normalized by N_max)
    - active_ratio (1D, controllable / total)
    - avg_battery_soc (1D)
    - avg_ev_soc (1D)
    - avg_heatpump_comfort (1D)
    - total_pv_power (1D, normalized)
    
    Total: 6D
    """
    total_devices = len(manager.device_mdps)
    controllable_devices = len(manager.controllable_devices)
    
    # Device counts
    total_norm = total_devices / 60.0  # N_max
    active_ratio = controllable_devices / max(total_devices, 1)
    
    # Aggregate device states by type
    battery_socs = []
    ev_socs = []
    heatpump_comforts = []
    pv_powers = []
    
    for device_id, mdp in manager.device_mdps.items():
        device_type = manager.device_types[device_id]
        state = mdp.get_state_features()
        
        if device_type == 'battery' and len(state) > 0:
            battery_socs.append(state[0])  # SOC is first element
        elif device_type == 'ev' and len(state) > 0:
            ev_socs.append(state[0])
        elif device_type == 'heat_pump' and len(state) > 2:
            heatpump_comforts.append(state[2])  # Comfort is third element
        elif device_type == 'pv' and len(state) > 0:
            pv_powers.append(state[0])
    
    avg_battery_soc = np.mean(battery_socs) if battery_socs else 0.5
    avg_ev_soc = np.mean(ev_socs) if ev_socs else 0.5
    avg_heatpump_comfort = np.mean(heatpump_comforts) if heatpump_comforts else 0.5
    total_pv_power = np.sum(pv_powers) if pv_powers else 0.0
    total_pv_power_norm = np.clip(total_pv_power / 100.0, 0.0, 1.0)  # Assume max 100 kW
    
    return np.array([
        total_norm,
        active_ratio,
        avg_battery_soc,
        avg_ev_soc,
        avg_heatpump_comfort,
        total_pv_power_norm
    ], dtype=np.float32)


def extract_global_features_from_env(
    env,
    manager_id: str,
    current_time: datetime
) -> np.ndarray:
    """
    Extract global features from environment
    
    Combines:
    - Time features (9D)
    - Price features (4D)
    - Weather features (3D)
    - Market features (4D)
    - Manager features (6D)
    
    Total: 26D (not 50D as initially estimated, more compact)
    
    Note: Can be extended to reach ~50D by adding:
    - Forecast features (next 24 hours)
    - Historical aggregates
    - Network topology features
    
    Args:
        env: MultiAgentFlexOfferEnv instance
        manager_id: Manager ID
        current_time: Current simulation time
    
    Returns:
        g: [g_dim] Global features array
    """
    # Get environment state
    env_state = env.env_dynamics.get_current_state(current_time)
    
    # Get manager
    manager = env.manager_agents.get(manager_id)
    if manager is None:
        logger.error(f"Manager {manager_id} not found")
        # Return zero features
        return np.zeros(26, dtype=np.float32)
    
    # Extract feature components
    time_feats = extract_time_features(current_time)  # 9D
    price_feats = extract_price_features(env_state)  # 4D
    weather_feats = extract_weather_features(env_state)  # 3D
    market_feats = extract_market_features(env_state)  # 4D
    manager_feats = extract_manager_features(manager)  # 6D
    
    # Concatenate all features
    g = np.concatenate([
        time_feats,
        price_feats,
        weather_feats,
        market_feats,
        manager_feats
    ])
    
    return g


def convert_ea_action_to_fogym(
    A_padded: np.ndarray,
    mask: np.ndarray,
    device_ids: List[str]
) -> np.ndarray:
    """
    Convert EA's padded action to FOgym's flattened format
    
    EA output: [N_max, p] where inactive slots are masked
    FOgym expects: [n_active_devices * p] flattened array
    
    Args:
        A_padded: [N_max, p] - EA output actions (padded)
        mask: [N_max] - Active device mask
        device_ids: List of device IDs (for logging)
    
    Returns:
        action: [n_active * p] - Flattened action for FOgym
    """
    # Get number of active devices
    n_active = int(mask.sum())
    
    if n_active == 0:
        logger.warning("No active devices, returning empty action")
        return np.array([], dtype=np.float32)
    
    # Extract active actions
    A_active = A_padded[:n_active]  # [n_active, p]
    
    # Flatten
    action = A_active.flatten()  # [n_active * p]
    
    return action


def convert_fogym_action_to_dict(
    action_flat: np.ndarray,
    device_ids: List[str],
    p: int = 5
) -> Dict[str, np.ndarray]:
    """
    Convert flattened action to device-keyed dictionary
    
    This is an alternative format that some FOgym interfaces might prefer.
    
    Args:
        action_flat: [n_devices * p] - Flattened action
        device_ids: List of device IDs
        p: Action dimension per device
    
    Returns:
        action_dict: {device_id: action_vector}
    """
    n_devices = len(device_ids)
    
    if len(action_flat) != n_devices * p:
        logger.error(f"Action dimension mismatch: expected {n_devices * p}, got {len(action_flat)}")
        # Return zero actions as fallback
        return {device_id: np.zeros(p, dtype=np.float32) for device_id in device_ids}
    
    action_dict = {}
    for i, device_id in enumerate(device_ids):
        action_dict[device_id] = action_flat[i * p:(i + 1) * p]
    
    return action_dict


# ====================
# Batch Processing Utils
# ====================

def prepare_batch_observations(
    managers: Dict,
    env,
    current_time: datetime,
    N_max: int = 60,
    x_dim: int = 6,
    g_dim: int = 26
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]]:
    """
    Prepare observations for all managers in batch
    
    Args:
        managers: Dict of ManagerAgent instances
        env: Environment instance
        current_time: Current simulation time
        N_max: Maximum devices
        x_dim: Device state dimension
        g_dim: Global feature dimension
    
    Returns:
        obs_dict: {manager_id: (g, X, mask, device_ids)}
    """
    obs_dict = {}
    
    for manager_id, manager in managers.items():
        # Extract device states
        X, mask, device_ids = extract_device_states_from_manager(manager, N_max, x_dim)
        
        # Extract global features
        g = extract_global_features_from_env(env, manager_id, current_time)
        
        obs_dict[manager_id] = (g, X, mask, device_ids)
    
    return obs_dict


def convert_batch_actions(
    actions_dict: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]],
    p: int = 5
) -> Dict[str, np.ndarray]:
    """
    Convert batch of EA actions to FOgym format
    
    Args:
        actions_dict: {manager_id: (A_padded, mask, device_ids)}
        p: Action dimension per device
    
    Returns:
        fogym_actions: {manager_id: flattened_action}
    """
    fogym_actions = {}
    
    for manager_id, (A_padded, mask, device_ids) in actions_dict.items():
        action = convert_ea_action_to_fogym(A_padded, mask, device_ids)
        fogym_actions[manager_id] = action
    
    return fogym_actions
