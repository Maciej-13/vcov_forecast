from dataclasses import dataclass, field
from typing import Optional, Dict, Union, List

# Source: https://pyportfolioopt.readthedocs.io/en/latest/RiskModels.html
RISK_MODELS = [
    "sample_cov",
    "semicovariance",
    "exp_cov",
    "ledoit_wolf",
    "ledoit_wolf_constant_variance",
    "ledoit_wolf_single_factor",
    "ledoit_wolf_constant_correlation",
    "oracle_approximating",
]

# Self-defined covariances
ESTIMATORS = [
    "lstm",
]

# Combination of covariance matrices
VALID_COV = RISK_MODELS + ESTIMATORS


@dataclass
class StrategyParameters:
    window: int
    covariance: str
    rebalancing: Optional[int] = None
    fees: Optional[float] = None
    additional: Dict[str, Union[int, float, str]] = field(default_factory=dict)

    def __post_init__(self):
        self._validate_covariance(self.covariance)

    @staticmethod
    def _validate_covariance(covariance: str, valid_methods: Optional[List[str]] = None):
        valid_methods = VALID_COV if valid_methods is None else valid_methods
        if covariance not in valid_methods:
            raise ValueError(f"Unknown covariance estimation method: {covariance}! Use one of: {valid_methods}")
