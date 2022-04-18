from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict

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

# Available methods to estimate returns
RETURNS = [
    'mean_historical_return',
    'ema_historical_return',
    'capm_return'
]

# Available optimization rules
OPTIMIZE = [
    'min_volatility',
    'max_sharpe',
    'max_quadratic_utility',
    'efficient_risk',
    'efficient_return'
]


@dataclass
class StrategyParameters:
    window: int
    covariance_model: str
    portfolio_value: Union[int, float]
    returns: str = "mean_historical_return"
    optimize: str = 'min_volatility'
    cov_params: Dict[str, Union[int, str, float]] = field(default_factory=dict)
    warmup_period: int = 0
    rebalancing: Optional[int] = None
    fee_multiplier: Optional[float] = None

    def __post_init__(self):
        self._validate_parameter(self.covariance_model, VALID_COV)
        self._validate_parameter(self.returns, RETURNS)
        self._validate_parameter(self.optimize, OPTIMIZE)
        self.classical: bool = True if self.covariance_model in RISK_MODELS else False

    @staticmethod
    def _validate_parameter(parameter: str, valid_methods: List[str]):
        if parameter not in valid_methods:
            raise ValueError(
                f"Unknown parameter: {parameter}! Use one of: {valid_methods}")
