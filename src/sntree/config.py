from dataclasses import dataclass

@dataclass
class Config:
    alpha_init: float = 0.0001
    beta_init: float = 0.0001
    p0: float = 0.001
    batch_size: int = 1024
    max_iters: int = 50
    em_max_iter: int = 30