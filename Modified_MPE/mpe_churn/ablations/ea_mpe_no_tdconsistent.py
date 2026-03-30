"""
Ablation: EA without TD-Consistent bootstrapping on MPE.
Uses current mask for TD target instead of next mask.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Modified_MPE.mpe_churn.mpe_algo_wrapper import MPE_EA
from Modified_MPE.mpe_churn.mpe_configs import EA_CONFIG


class MPE_EA_NoTDConsistent(MPE_EA):
    """EA without TD-Consistent. Reuses MPE_EA but with EAAgentNoTDConsistent."""

    def __init__(self, device: str = "cpu"):
        super(MPE_EA, self).__init__(device)
        from Test.Ablation.agents.ea_no_tdconsistent import EAAgentNoTDConsistent

        self.agents = {}
        for i, ctrl_id in enumerate(self.ctrl_ids):
            self.agents[ctrl_id] = EAAgentNoTDConsistent(
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
                credit_warmup_start=EA_CONFIG.get('credit_warmup_start', 100),
                credit_warmup_end=EA_CONFIG.get('credit_warmup_end', 300),
                credit_max_weight=EA_CONFIG.get('credit_max_weight', 0.05),
                device=device,
            )
