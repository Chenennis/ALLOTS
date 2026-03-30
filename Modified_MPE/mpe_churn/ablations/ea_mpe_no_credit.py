"""
Ablation: EA without per-device Entity Credit on MPE.
Uses uniform weights instead of softmax-weighted advantages.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Modified_MPE.mpe_churn.mpe_algo_wrapper import MPE_EA
from Modified_MPE.mpe_churn.mpe_configs import EA_CONFIG


class MPE_EA_NoCredit(MPE_EA):
    """EA without Entity Credit. Reuses MPE_EA but with EAAgentNoCredit."""

    def __init__(self, device: str = "cpu"):
        super(MPE_EA, self).__init__(device)
        from Test.Ablation.agents.ea_no_credit import EAAgentNoCredit

        self.agents = {}
        for i, ctrl_id in enumerate(self.ctrl_ids):
            self.agents[ctrl_id] = EAAgentNoCredit(
                x_dim=EA_CONFIG['x_dim'],
                g_dim=EA_CONFIG['g_dim'],
                p=EA_CONFIG['p'],
                N_max=EA_CONFIG['N_max'],
                num_managers=EA_CONFIG['num_managers'],
                emb_dim=EA_CONFIG['emb_dim'],
                token_dim=EA_CONFIG['token_dim'],
                hidden_dim=EA_CONFIG['hidden_dim'],
                gamma=EA_CONFIG['gamma'],
                tau=EA_CONFIG['tau'],
                lr_actor=EA_CONFIG['lr_actor'],
                lr_critic=EA_CONFIG['lr_critic'],
                policy_delay=EA_CONFIG['policy_delay'],
                noise_scale=EA_CONFIG['noise_scale'],
                noise_clip=EA_CONFIG['noise_clip'],
                advantage_tau=EA_CONFIG['advantage_tau'],
                buffer_capacity=EA_CONFIG['buffer_capacity'],
                device=device,
            )
