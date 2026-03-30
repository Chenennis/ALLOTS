"""
Multidrone: Multi-Controller Cooperative Coverage with Fleet Dynamics.

Uses PyFlyt Aviary (PyBullet physics) for realistic 3D quadrotor simulation.
4 controllers, each managing N_max drone slots with mid-episode churn.

Structure mirrors MPEChurnEnv but in 3D with real drone physics.
"""

import sys
import os
import numpy as np
from typing import Dict, Tuple, Optional, List

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "Multidrone"))

from Multidrone.drone_churn.churn_config import (
    N_CONTROLLERS, N_INIT_PER, N_MAX_PER, N_WAYPOINTS,
    MAX_DURATION_SECONDS, AGENT_HZ, MIN_ACTIVE_PER,
    CHURN_TIMES_SEC, COVERAGE_X, COVERAGE_Y, CRUISE_ALTITUDE,
    FLIGHT_DOME_SIZE, X_DIM, G_DIM, P_DIM, REWARD_WEIGHTS,
)


class DroneChurnEnv:
    """
    Multi-Controller Cooperative Coverage with Fleet Dynamics.

    Uses PyFlyt Aviary for 3D quadrotor physics.
    Interface mirrors MPEChurnEnv:
    - reset() -> obs_dict: Dict[ctrl_id -> (X, g, mask)]
    - step(actions_dict) -> (obs_dict, rewards_dict, done, info)
    """

    def __init__(
        self,
        n_controllers: int = N_CONTROLLERS,
        n_init_per: int = N_INIT_PER,
        n_max_per: int = N_MAX_PER,
        n_waypoints: int = N_WAYPOINTS,
        max_duration: float = MAX_DURATION_SECONDS,
        agent_hz: int = AGENT_HZ,
        churn_times_sec: List[float] = None,
        churn_rho_range: Tuple[float, float] = (0.10, 0.15),
        reward_weights: Dict[str, float] = None,
        seed: Optional[int] = None,
    ):
        self.n_controllers = n_controllers
        self.n_init_per = n_init_per
        self.n_max_per = n_max_per
        self.n_waypoints = n_waypoints
        self.max_duration = max_duration
        self.agent_hz = agent_hz
        self.churn_times = churn_times_sec if churn_times_sec is not None else CHURN_TIMES_SEC
        self.rho_min, self.rho_max = churn_rho_range
        self.reward_weights = reward_weights or REWARD_WEIGHTS

        self.total_drones = n_controllers * n_max_per
        self.max_steps = int(max_duration * agent_hz)
        self.physics_steps_per_agent_step = 240 // agent_hz  # 240Hz / 40Hz = 6

        # Convert churn times to agent steps
        self.churn_steps = [int(t * agent_hz) for t in self.churn_times]

        self.ctrl_ids = [f"ctrl_{i}" for i in range(n_controllers)]
        self.rng = np.random.RandomState(seed)

        # State
        self.aviary = None
        self.active_masks = None
        self.waypoint_positions = None
        self.current_step = 0

        # Controller → drone index mapping
        self.ctrl_drone_ranges = {}
        for i in range(n_controllers):
            start = i * n_max_per
            end = (i + 1) * n_max_per
            self.ctrl_drone_ranges[self.ctrl_ids[i]] = (start, end)

        # Parking position for inactive drones (high altitude, out of coverage area)
        self.parking_pos = np.array([COVERAGE_X[1] + 25.0, COVERAGE_Y[1] + 25.0, 10.0])

    def reset(self, seed: Optional[int] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Reset environment. Creates new Aviary with pre-allocated drones."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.current_step = 0

        # Generate start positions for all drones
        # Each controller's drones start in a corner of the coverage area,
        # so agents must actively fly to cover waypoints spread across the map.
        corners = [
            (COVERAGE_X[0], COVERAGE_Y[0]),  # bottom-left
            (COVERAGE_X[1], COVERAGE_Y[0]),  # bottom-right
            (COVERAGE_X[0], COVERAGE_Y[1]),  # top-left
            (COVERAGE_X[1], COVERAGE_Y[1]),  # top-right
        ]
        spawn_radius = 3.0  # small jitter around corner
        start_pos = np.zeros((self.total_drones, 3))
        for i in range(self.total_drones):
            ctrl_idx = i // self.n_max_per
            slot_idx = i % self.n_max_per
            if slot_idx < self.n_init_per:
                cx, cy = corners[ctrl_idx % len(corners)]
                x = cx + self.rng.uniform(-spawn_radius, spawn_radius)
                y = cy + self.rng.uniform(-spawn_radius, spawn_radius)
                x = np.clip(x, COVERAGE_X[0], COVERAGE_X[1])
                y = np.clip(y, COVERAGE_Y[0], COVERAGE_Y[1])
                start_pos[i] = [x, y, CRUISE_ALTITUDE]
            else:
                start_pos[i] = self.parking_pos

        start_orn = np.zeros((self.total_drones, 3))

        # Destroy old aviary if exists
        if self.aviary is not None:
            self.aviary.disconnect()

        from PyFlyt.core import Aviary
        self.aviary = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            drone_type="quadx",
            render=False,
            seed=int(self.rng.randint(0, 2**31)),
        )
        self.aviary.set_mode(6)  # World velocity control [vx, vy, vr, vz]

        # Initialize active masks
        self.active_masks = {}
        for ctrl_id in self.ctrl_ids:
            mask = np.zeros(self.n_max_per)
            mask[:self.n_init_per] = 1.0
            self.active_masks[ctrl_id] = mask

        # Set armed status (only active drones are armed)
        armed = []
        for ctrl_id in self.ctrl_ids:
            mask = self.active_masks[ctrl_id]
            for j in range(self.n_max_per):
                armed.append(bool(mask[j]))
        self.aviary.set_armed(armed)

        # Generate waypoints
        self.waypoint_positions = np.zeros((self.n_waypoints, 3))
        for i in range(self.n_waypoints):
            self.waypoint_positions[i] = [
                self.rng.uniform(*COVERAGE_X),
                self.rng.uniform(*COVERAGE_Y),
                CRUISE_ALTITUDE,
            ]

        # Let physics settle for a moment
        for _ in range(10):
            self.aviary.set_all_setpoints(np.zeros((self.total_drones, 4)))
            self.aviary.step()

        return self._get_obs()

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict[str, float], bool, dict]:
        """Step environment.

        Args:
            actions: {ctrl_id: action[N_max, P_DIM=4]}  where P_DIM=4 is [vx, vy, vr, vz]
        """
        # Build setpoints for all drones
        setpoints = np.zeros((self.total_drones, 4))
        for ctrl_id, action in actions.items():
            start, end = self.ctrl_drone_ranges[ctrl_id]
            mask = self.active_masks[ctrl_id]
            for j in range(self.n_max_per):
                if mask[j] == 1.0:
                    # Scale action from [-1,1] to reasonable velocity range
                    a = action[j]
                    setpoints[start + j] = [
                        a[0] * 2.0,   # vx: [-2, 2] m/s
                        a[1] * 2.0,   # vy: [-2, 2] m/s
                        a[2] * 1.0,   # vr: [-1, 1] rad/s
                        a[3] * 1.0,   # vz: [-1, 1] m/s
                    ]

        # Physics steps
        self.aviary.set_all_setpoints(setpoints)
        for _ in range(self.physics_steps_per_agent_step):
            self.aviary.step()

        self.current_step += 1

        # Check churn
        churn_triggered = False
        churn_details = {}
        if self.current_step in self.churn_steps:
            churn_triggered = True
            churn_details = self._execute_churn()

        # Compute rewards
        rewards = self._compute_rewards(actions)

        done = (self.current_step >= self.max_steps)
        obs = self._get_obs()

        info = {
            'churn_triggered': churn_triggered,
            'churn_details': churn_details,
            'step': self.current_step,
            'time_sec': self.current_step / self.agent_hz,
            'active_counts': {k: int(v.sum()) for k, v in self.active_masks.items()},
        }

        return obs, rewards, done, info

    def _get_obs(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get observations for all controllers."""
        obs = {}
        for ctrl_id in self.ctrl_ids:
            X = self._get_entity_obs(ctrl_id)
            g = self._get_global_ctx(ctrl_id)
            mask = self.active_masks[ctrl_id].copy()
            obs[ctrl_id] = (X, g, mask)
        return obs

    def _get_entity_obs(self, ctrl_id: str) -> np.ndarray:
        """Per-entity obs: [pos3D, vel3D, target_rel3D] = 9D."""
        start, end = self.ctrl_drone_ranges[ctrl_id]
        mask = self.active_masks[ctrl_id]
        X = np.zeros((self.n_max_per, X_DIM))

        for j in range(self.n_max_per):
            if mask[j] == 0.0:
                continue
            drone_idx = start + j
            state = self.aviary.state(drone_idx)  # (4, 3)
            pos = state[3]  # world position [x, y, z]
            vel = state[2]  # body velocity [u, v, w]

            # Find nearest waypoint
            dists = np.linalg.norm(self.waypoint_positions - pos, axis=1)
            nearest_idx = np.argmin(dists)
            target_rel = self.waypoint_positions[nearest_idx] - pos
            # Normalize positions and relative target by coverage scale
            cov_scale = max(COVERAGE_X[1] - COVERAGE_X[0], COVERAGE_Y[1] - COVERAGE_Y[0])

            X[j] = [pos[0] / cov_scale, pos[1] / cov_scale, pos[2] / cov_scale,
                     vel[0], vel[1], vel[2],
                     target_rel[0] / cov_scale, target_rel[1] / cov_scale, target_rel[2] / cov_scale]

        return X

    def _get_global_ctx(self, ctrl_id: str) -> np.ndarray:
        """Global context: 17D."""
        g = np.zeros(G_DIM)

        # Public
        total_active = sum(m.sum() for m in self.active_masks.values())
        total_max = self.total_drones
        covered = self._count_covered_waypoints()

        g[0] = self.current_step / self.max_steps  # time progress
        g[1] = total_active / total_max             # total active ratio
        g[2] = covered / self.n_waypoints           # coverage ratio

        # Private
        mask = self.active_masks[ctrl_id]
        start, end = self.ctrl_drone_ranges[ctrl_id]
        n_active = mask.sum()
        g[3] = n_active / self.n_max_per

        # Fleet centroid (3D, but use x,y only for compactness → g_dim stays 17)
        cov_scale = max(COVERAGE_X[1] - COVERAGE_X[0], COVERAGE_Y[1] - COVERAGE_Y[0])
        if n_active > 0:
            positions = []
            for j in range(self.n_max_per):
                if mask[j] == 1.0:
                    positions.append(self.aviary.state(start + j)[3])
            centroid = np.mean(positions, axis=0)
            g[4] = centroid[0] / cov_scale
            g[5] = centroid[1] / cov_scale
            g[6] = centroid[2] / cov_scale

        # Cross-controller: (M-1) × [cx, cy, active_ratio] = 3*3=9 → non-z centroid
        offset = 7
        for other_id in self.ctrl_ids:
            if other_id == ctrl_id:
                continue
            other_mask = self.active_masks[other_id]
            other_start, _ = self.ctrl_drone_ranges[other_id]
            other_n = other_mask.sum()

            if other_n > 0:
                other_pos = []
                for j in range(self.n_max_per):
                    if other_mask[j] == 1.0:
                        other_pos.append(self.aviary.state(other_start + j)[3])
                oc = np.mean(other_pos, axis=0)
                g[offset] = oc[0] / cov_scale
                g[offset + 1] = oc[1] / cov_scale
            # +2 offset here to keep z out, saving 1 dim per controller
            g[offset + 2] = other_n / self.n_max_per
            offset += 3

        # Remaining: offset should be 7 + 3*3 = 16. One more slot:
        # g[16] = average distance to nearest uncovered waypoint (extra info)
        if n_active > 0 and covered < self.n_waypoints:
            all_pos = []
            for j in range(self.n_max_per):
                if mask[j] == 1.0:
                    all_pos.append(self.aviary.state(start + j)[3])
            avg_d = 0
            for pos in all_pos:
                dists = np.linalg.norm(self.waypoint_positions - pos, axis=1)
                avg_d += np.min(dists)
            g[16] = avg_d / n_active / 10.0  # normalize by ~max distance

        return g

    def _count_covered_waypoints(self, threshold: float = 1.0) -> int:
        """Count waypoints with at least one active drone within threshold distance."""
        all_positions = []
        for ctrl_id in self.ctrl_ids:
            start, end = self.ctrl_drone_ranges[ctrl_id]
            mask = self.active_masks[ctrl_id]
            for j in range(self.n_max_per):
                if mask[j] == 1.0:
                    all_positions.append(self.aviary.state(start + j)[3])

        if not all_positions:
            return 0

        all_pos = np.array(all_positions)
        covered = 0
        for wp in self.waypoint_positions:
            dists = np.linalg.norm(all_pos - wp, axis=1)
            if np.min(dists) < threshold:
                covered += 1
        return covered

    def _compute_rewards(self, actions: Dict) -> Dict[str, float]:
        """Compute per-controller rewards."""
        alpha = self.reward_weights['global_coverage']
        beta = self.reward_weights['local_efficiency']
        gamma = self.reward_weights['collision_penalty']
        delta = self.reward_weights['energy_penalty']

        # All active positions
        all_active = []
        ctrl_positions = {}
        for ctrl_id in self.ctrl_ids:
            start, end = self.ctrl_drone_ranges[ctrl_id]
            mask = self.active_masks[ctrl_id]
            pos_list = []
            for j in range(self.n_max_per):
                if mask[j] == 1.0:
                    pos_list.append(self.aviary.state(start + j)[3].copy())
            ctrl_positions[ctrl_id] = pos_list
            all_active.extend(pos_list)

        # Global coverage
        global_cov = 0.0
        if all_active:
            all_pos = np.array(all_active)
            for wp in self.waypoint_positions:
                dists = np.linalg.norm(all_pos - wp, axis=1)
                global_cov -= np.min(dists)
            global_cov /= self.n_waypoints

        rewards = {}
        for ctrl_id in self.ctrl_ids:
            pos_list = ctrl_positions[ctrl_id]

            # Local efficiency
            local_eff = 0.0
            if pos_list:
                for pos in pos_list:
                    dists = np.linalg.norm(self.waypoint_positions - pos, axis=1)
                    local_eff -= np.min(dists)
                local_eff /= len(pos_list)

            # Collision penalty (from aviary contact array if available)
            n_collisions = 0
            if hasattr(self.aviary, 'contact_array'):
                start, end = self.ctrl_drone_ranges[ctrl_id]
                mask = self.active_masks[ctrl_id]
                for j in range(self.n_max_per):
                    if mask[j] == 1.0:
                        drone = self.aviary.drones[start + j]
                        if hasattr(drone, 'body_ids'):
                            for bid in drone.body_ids:
                                if np.any(self.aviary.contact_array[bid]):
                                    n_collisions += 1
                                    break

            # Energy penalty
            energy = 0.0
            if ctrl_id in actions:
                mask = self.active_masks[ctrl_id]
                act = actions[ctrl_id]
                for j in range(self.n_max_per):
                    if mask[j] == 1.0:
                        energy += np.linalg.norm(act[j])
                if sum(mask) > 0:
                    energy /= sum(mask)

            rewards[ctrl_id] = (
                alpha * global_cov
                + beta * local_eff
                - gamma * n_collisions
                - delta * energy * 0.1
            )

        return rewards

    def _execute_churn(self) -> dict:
        """Execute entity churn for all controllers."""
        details = {}
        armed_list = []

        for ctrl_id in self.ctrl_ids:
            mask = self.active_masks[ctrl_id]
            start, end = self.ctrl_drone_ranges[ctrl_id]
            n_active = int(mask.sum())

            rho = self.rng.uniform(self.rho_min, self.rho_max)
            k = max(1, round(rho * n_active))
            k_leave = k // 2
            k_join = k - k_leave
            k_leave = min(k_leave, max(0, n_active - MIN_ACTIVE_PER))

            # Leave
            active_indices = np.where(mask == 1.0)[0]
            leave_indices = []
            if k_leave > 0 and len(active_indices) > 0:
                leave_indices = self.rng.choice(active_indices, min(k_leave, len(active_indices)), replace=False).tolist()
                for idx in leave_indices:
                    mask[idx] = 0.0

            # Join
            inactive_indices = np.where(mask == 0.0)[0]
            join_indices = []
            if k_join > 0 and len(inactive_indices) > 0:
                join_indices = self.rng.choice(inactive_indices, min(k_join, len(inactive_indices)), replace=False).tolist()
                for idx in join_indices:
                    mask[idx] = 1.0
                    # Reset drone position
                    drone_idx = start + idx
                    new_pos = [
                        self.rng.uniform(*COVERAGE_X),
                        self.rng.uniform(*COVERAGE_Y),
                        CRUISE_ALTITUDE,
                    ]
                    self.aviary.resetBasePositionAndOrientation(
                        self.aviary.drones[drone_idx].Id,
                        new_pos, [0, 0, 0, 1],
                    )
                    self.aviary.resetBaseVelocity(
                        self.aviary.drones[drone_idx].Id,
                        [0, 0, 0], [0, 0, 0],
                    )

            self.active_masks[ctrl_id] = mask
            details[ctrl_id] = {
                'left': len(leave_indices), 'joined': len(join_indices),
                'active_before': n_active, 'active_after': int(mask.sum()),
            }

        # Update armed status
        for ctrl_id in self.ctrl_ids:
            mask = self.active_masks[ctrl_id]
            for j in range(self.n_max_per):
                armed_list.append(bool(mask[j]))
        self.aviary.set_armed(armed_list)

        return details

    def close(self):
        if self.aviary is not None:
            self.aviary.disconnect()
            self.aviary = None
