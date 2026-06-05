from dataclasses import dataclass

@dataclass
class Config:
    alpha_init: float = 0.0001
    beta_init: float = 0.0001
    p0: float = 0.001
    pi0: float = 0.001
    batch_size: int = 1024
    nni_max_iters: int = 50
    em_max_iter: int = 30
    # Soft branch-proportion EM (sntree_extended)
    run_soft_em: bool = True
    soft_em_max_iter: int = 50
    soft_em_joint: bool = False  # jointly update alpha/beta in soft pass
    alpha_dir: float = 1.0       # Dirichlet concentration on pi_b
