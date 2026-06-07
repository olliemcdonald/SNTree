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
    # Soft branch-proportion EM
    soft_em_max_iter: int = 50
    alpha_dir: float = 1.0       # Dirichlet concentration on pi_b
    # Pipeline control
    run_hard_em: bool = False    # opt-in: run hard EM before soft EM pass 1
    soft_em_pass2: bool = True   # run a second soft EM pass after refinement
