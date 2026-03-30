"""
Modified MPE: Multi-Controller Cooperative Coverage with Entity Churn

Uses MPE physics engine (World, Agent, Landmark from core.py) directly,
manages active/inactive entities via masking (same paradigm as FOgym).

4 controllers, each managing N_max particle slots with mid-episode churn.
"""

import sys
import os
import numpy as np
from typing import Dict, Tuple, Optional, List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Direct import of MPE core physics - avoid pettingzoo __init__ which requires pip install
import importlib.util
_core_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pzoo", "pettingzoo", "mpe", "_mpe_utils", "core.py"
)
_spec = importlib.util.spec_from_file_location("mpe_core", _core_path)
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)
World = _core.World
Agent = _core.Agent
Landmark = _core.Landmark
Entity = _core.Entity
from Modified_MPE.mpe_churn.churn_config import (
    N_CONTROLLERS, N_INIT_PER, N_MAX_PER, N_LANDMARKS, T,
    CHURN_STEPS, MIN_ACTIVE_PER, WORLD_SIZE, AGENT_SIZE, LANDMARK_SIZE,
    X_DIM, G_DIM, P_DIM, REWARD_WEIGHTS, SENSITIVITY,
)


class MPEChurnEnv:
    """
    Multi-Controller Cooperative Coverage with Entity Churn.

    Structure mirrors FOgym:
    - 4 controllers (= 4 managers in FOgym)
    - Each controller manages N_max particle slots
    - Entities can join/leave mid-episode via churn
    - Observations: per-entity (X) + global context (g) + active mask

    Interface:
    - reset() -> obs_dict: Dict[ctrl_id -> (X, g, mask)]
    - step(actions_dict) -> (obs_dict, rewards_dict, done, info)
    """

    def __init__(
        self,
        n_controllers: int = N_CONTROLLERS,
        n_init_per: int = N_INIT_PER,
        n_max_per: int = N_MAX_PER,
        n_landmarks: int = N_LANDMARKS,
        episode_length: int = T,
        churn_steps: List[int] = None,
        churn_rho_range: Tuple[float, float] = (0.10, 0.15),
        collision_penalty: float = 1.0,
        reward_weights: Dict[str, float] = None,
        seed: Optional[int] = None,
    ):
        self.n_controllers = n_controllers
        self.n_init_per = n_init_per
        self.n_max_per = n_max_per
        self.n_landmarks = n_landmarks
        self.episode_length = episode_length
        self.churn_steps = churn_steps if churn_steps is not None else CHURN_STEPS
        self.rho_min, self.rho_max = churn_rho_range
        self.collision_penalty = collision_penalty
        self.reward_weights = reward_weights or REWARD_WEIGHTS

        self.total_agents = n_controllers * n_max_per
        self.total_init = n_controllers * n_init_per

        self.ctrl_ids = [f"ctrl_{i}" for i in range(n_controllers)]

        # Random state
        self.rng = np.random.RandomState(seed)

        # Will be initialized in reset()
        self.world = None
        self.active_masks = None  # {ctrl_id: np.ndarray [n_max_per]}
        self.current_step = 0
        self.landmark_positions = None  # [n_landmarks, 2]

        # Controller → agent index mapping
        # ctrl_i owns agents [i*n_max_per : (i+1)*n_max_per]
        self.ctrl_agent_ranges = {}
        for i in range(n_controllers):
            start = i * n_max_per
            end = (i + 1) * n_max_per
            self.ctrl_agent_ranges[self.ctrl_ids[i]] = (start, end)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Reset environment.

        Returns:
            obs_dict: {ctrl_id: (X[N_max, 6], g[14], mask[N_max])}
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.current_step = 0

        # Create world
        self.world = World()
        self.world.dim_c = 0

        # Create all agents (pre-allocated)
        for i in range(self.total_agents):
            agent = Agent()
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = AGENT_SIZE
            agent.movable = True
            agent.state.p_pos = self.rng.uniform(-WORLD_SIZE, WORLD_SIZE, 2)
            agent.state.p_vel = np.zeros(2)
            agent.state.c = np.zeros(self.world.dim_c) if self.world.dim_c > 0 else np.zeros(0)
            agent.action.u = np.zeros(2)
            self.world.agents.append(agent)

        # Create all landmarks
        for i in range(self.n_landmarks):
            landmark = Landmark()
            landmark.name = f"landmark_{i}"
            landmark.collide = False
            landmark.movable = False
            landmark.size = LANDMARK_SIZE
            landmark.state.p_pos = self.rng.uniform(-WORLD_SIZE, WORLD_SIZE, 2)
            landmark.state.p_vel = np.zeros(2)
            self.world.landmarks.append(landmark)

        self.landmark_positions = np.array([lm.state.p_pos for lm in self.world.landmarks])

        # Initialize active masks: first n_init_per active per controller
        self.active_masks = {}
        for ctrl_id in self.ctrl_ids:
            mask = np.zeros(self.n_max_per)
            mask[:self.n_init_per] = 1.0
            self.active_masks[ctrl_id] = mask

        # Place inactive agents far away
        self._place_inactive_agents()

        return self._get_obs()

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[Dict, Dict[str, float], bool, dict]:
        """
        Step environment.

        Args:
            actions: {ctrl_id: action[N_max, P_DIM]}

        Returns:
            obs_dict, rewards_dict, done, info
        """
        # Apply actions to active agents
        self._apply_actions(actions)

        # Physics step
        self.world.step()

        # Sanitize: fix NaN positions/velocities from collision edge cases
        for agent in self.world.agents:
            if np.any(np.isnan(agent.state.p_pos)):
                agent.state.p_pos = np.zeros(2)
            if np.any(np.isnan(agent.state.p_vel)):
                agent.state.p_vel = np.zeros(2)

        # Clip positions to world bounds
        for agent in self.world.agents:
            agent.state.p_pos = np.clip(agent.state.p_pos, -WORLD_SIZE, WORLD_SIZE)

        self.current_step += 1

        # Check for churn
        churn_triggered = False
        churn_details = {}
        if self.current_step in self.churn_steps:
            churn_triggered = True
            churn_details = self._execute_churn()

        # Compute rewards
        rewards = self._compute_rewards()

        # Check done
        done = (self.current_step >= self.episode_length)

        # Get observations
        obs = self._get_obs()

        info = {
            'churn_triggered': churn_triggered,
            'churn_details': churn_details,
            'step': self.current_step,
            'active_counts': {k: int(v.sum()) for k, v in self.active_masks.items()},
        }

        return obs, rewards, done, info

    def _apply_actions(self, actions: Dict[str, np.ndarray]):
        """Apply actions to active agents only."""
        for ctrl_id, action in actions.items():
            start, end = self.ctrl_agent_ranges[ctrl_id]
            mask = self.active_masks[ctrl_id]

            for j in range(self.n_max_per):
                agent = self.world.agents[start + j]
                if mask[j] == 1.0:
                    # action[j] is [P_DIM] = [no_action, left, right, down, up]
                    # Convert to 2D force (same as simple_spread)
                    a = action[j]
                    force = np.zeros(2)
                    force[0] += a[2] - a[1]  # right - left
                    force[1] += a[4] - a[3]  # up - down
                    agent.action.u = force * SENSITIVITY
                else:
                    agent.action.u = np.zeros(2)

    def _get_obs(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get observations for all controllers.

        Returns:
            {ctrl_id: (X[N_max, X_DIM], g[G_DIM], mask[N_max])}
        """
        obs = {}
        for i, ctrl_id in enumerate(self.ctrl_ids):
            X = self._get_entity_obs(ctrl_id)
            g = self._get_global_ctx(ctrl_id)
            mask = self.active_masks[ctrl_id].copy()
            obs[ctrl_id] = (X, g, mask)
        return obs

    def _get_entity_obs(self, ctrl_id: str) -> np.ndarray:
        """
        Per-entity observation for a controller's fleet.

        Returns:
            X: [N_max, X_DIM=6]
            [pos_x, pos_y, vel_x, vel_y, nearest_lm_rel_x, nearest_lm_rel_y]
        """
        start, end = self.ctrl_agent_ranges[ctrl_id]
        mask = self.active_masks[ctrl_id]
        X = np.zeros((self.n_max_per, X_DIM))

        for j in range(self.n_max_per):
            if mask[j] == 0.0:
                continue
            agent = self.world.agents[start + j]
            pos = agent.state.p_pos
            vel = agent.state.p_vel

            # Find nearest landmark
            dists = np.linalg.norm(self.landmark_positions - pos, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_rel = self.landmark_positions[nearest_idx] - pos

            X[j] = [pos[0], pos[1], vel[0], vel[1], nearest_rel[0], nearest_rel[1]]

        return X

    def _get_global_ctx(self, ctrl_id: str) -> np.ndarray:
        """
        Global context for a controller.

        Returns:
            g: [G_DIM=14]
            [step/T, total_active/total_max,
             own_active/N_max, centroid_x, centroid_y,
             × (M-1): other_centroid_x, other_centroid_y, other_active_ratio]
        """
        g = np.zeros(G_DIM)
        ctrl_idx = self.ctrl_ids.index(ctrl_id)

        # Public info
        total_active = sum(m.sum() for m in self.active_masks.values())
        g[0] = self.current_step / self.episode_length  # time progress
        g[1] = total_active / self.total_agents          # total active ratio

        # Private info
        mask = self.active_masks[ctrl_id]
        start, end = self.ctrl_agent_ranges[ctrl_id]
        n_active = mask.sum()
        g[2] = n_active / self.n_max_per  # own active ratio

        # Own fleet centroid
        if n_active > 0:
            positions = []
            for j in range(self.n_max_per):
                if mask[j] == 1.0:
                    positions.append(self.world.agents[start + j].state.p_pos)
            centroid = np.mean(positions, axis=0)
            g[3] = centroid[0]
            g[4] = centroid[1]

        # Cross-controller info: (M-1) × [centroid_x, centroid_y, active_ratio]
        offset = 5
        for other_idx, other_id in enumerate(self.ctrl_ids):
            if other_id == ctrl_id:
                continue
            other_mask = self.active_masks[other_id]
            other_start, other_end = self.ctrl_agent_ranges[other_id]
            other_n_active = other_mask.sum()

            if other_n_active > 0:
                other_positions = []
                for j in range(self.n_max_per):
                    if other_mask[j] == 1.0:
                        other_positions.append(self.world.agents[other_start + j].state.p_pos)
                other_centroid = np.mean(other_positions, axis=0)
                g[offset] = other_centroid[0]
                g[offset + 1] = other_centroid[1]
            g[offset + 2] = other_n_active / self.n_max_per
            offset += 3

        return g

    def _compute_rewards(self) -> Dict[str, float]:
        """
        Compute per-controller rewards.

        reward_i = α * global_coverage + β * local_efficiency_i - γ * collision_penalty_i
        """
        alpha = self.reward_weights['global_coverage']
        beta = self.reward_weights['local_efficiency']
        gamma = self.reward_weights['collision_penalty']

        # Collect all active agent positions
        all_active_positions = []
        ctrl_active_positions = {}
        for ctrl_id in self.ctrl_ids:
            start, end = self.ctrl_agent_ranges[ctrl_id]
            mask = self.active_masks[ctrl_id]
            positions = []
            for j in range(self.n_max_per):
                if mask[j] == 1.0:
                    positions.append(self.world.agents[start + j].state.p_pos.copy())
            ctrl_active_positions[ctrl_id] = positions
            all_active_positions.extend(positions)

        # Global coverage: avg min distance from each landmark to nearest active agent
        global_coverage = 0.0
        if len(all_active_positions) > 0:
            all_pos = np.array(all_active_positions)
            for lm_pos in self.landmark_positions:
                dists = np.linalg.norm(all_pos - lm_pos, axis=1)
                global_coverage -= np.min(dists)
            global_coverage /= self.n_landmarks

        rewards = {}
        for ctrl_id in self.ctrl_ids:
            # Local efficiency: avg distance of fleet's agents to nearest landmark
            local_eff = 0.0
            positions = ctrl_active_positions[ctrl_id]
            if len(positions) > 0:
                for pos in positions:
                    dists = np.linalg.norm(self.landmark_positions - pos, axis=1)
                    local_eff -= np.min(dists)
                local_eff /= len(positions)

            # Collision penalty: count collisions involving this controller's agents
            n_collisions = 0
            start, end = self.ctrl_agent_ranges[ctrl_id]
            mask = self.active_masks[ctrl_id]
            for j in range(self.n_max_per):
                if mask[j] == 0.0:
                    continue
                agent = self.world.agents[start + j]
                # Check against ALL other active agents
                for other_ctrl_id in self.ctrl_ids:
                    other_start, other_end = self.ctrl_agent_ranges[other_ctrl_id]
                    other_mask = self.active_masks[other_ctrl_id]
                    for k in range(self.n_max_per):
                        if other_mask[k] == 0.0:
                            continue
                        other_agent = self.world.agents[other_start + k]
                        if agent is other_agent:
                            continue
                        dist = np.linalg.norm(agent.state.p_pos - other_agent.state.p_pos)
                        if dist < agent.size + other_agent.size:
                            n_collisions += 1

            n_collisions //= 2  # Each collision counted twice

            rewards[ctrl_id] = (
                alpha * global_coverage
                + beta * local_eff
                - gamma * self.collision_penalty * n_collisions
            )

        return rewards

    def _execute_churn(self) -> dict:
        """
        Execute entity churn for all controllers independently.

        Returns:
            churn_details: {ctrl_id: {'left': int, 'joined': int, 'active_before': int, 'active_after': int}}
        """
        details = {}
        for ctrl_id in self.ctrl_ids:
            mask = self.active_masks[ctrl_id]
            start, end = self.ctrl_agent_ranges[ctrl_id]
            n_active = int(mask.sum())

            # Sample churn ratio
            rho = self.rng.uniform(self.rho_min, self.rho_max)
            k = max(1, round(rho * n_active))
            k_leave = k // 2
            k_join = k - k_leave

            # Enforce minimum active
            k_leave = min(k_leave, max(0, n_active - MIN_ACTIVE_PER))

            # Leave: deactivate random active agents
            active_indices = np.where(mask == 1.0)[0]
            if k_leave > 0 and len(active_indices) > 0:
                leave_indices = self.rng.choice(active_indices, min(k_leave, len(active_indices)), replace=False)
                for idx in leave_indices:
                    mask[idx] = 0.0
                    agent = self.world.agents[start + idx]
                    agent.state.p_pos = np.array([100.0, 100.0])
                    agent.state.p_vel = np.zeros(2)
                    agent.action.u = np.zeros(2)
            else:
                leave_indices = []

            # Join: activate random inactive agents
            inactive_indices = np.where(mask == 0.0)[0]
            if k_join > 0 and len(inactive_indices) > 0:
                join_indices = self.rng.choice(inactive_indices, min(k_join, len(inactive_indices)), replace=False)
                for idx in join_indices:
                    mask[idx] = 1.0
                    agent = self.world.agents[start + idx]
                    agent.state.p_pos = self.rng.uniform(-WORLD_SIZE, WORLD_SIZE, 2)
                    agent.state.p_vel = np.zeros(2)
                    agent.action.u = np.zeros(2)
            else:
                join_indices = []

            self.active_masks[ctrl_id] = mask
            details[ctrl_id] = {
                'left': len(leave_indices),
                'joined': len(join_indices),
                'active_before': n_active,
                'active_after': int(mask.sum()),
            }

        return details

    def _place_inactive_agents(self):
        """Move all inactive agents to far-away parking position."""
        for ctrl_id in self.ctrl_ids:
            start, end = self.ctrl_agent_ranges[ctrl_id]
            mask = self.active_masks[ctrl_id]
            for j in range(self.n_max_per):
                if mask[j] == 0.0:
                    agent = self.world.agents[start + j]
                    agent.state.p_pos = np.array([100.0, 100.0])
                    agent.state.p_vel = np.zeros(2)
                    agent.action.u = np.zeros(2)
                    agent.collide = False  # Inactive agents don't collide
                else:
                    self.world.agents[start + j].collide = True
