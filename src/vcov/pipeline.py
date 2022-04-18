import pickle
import json
import datetime

from typing import Union, Optional
from dataclasses import asdict

from vcov.modules.data_handling.assets import Assets
from vcov.modules.portfolio.parameters import StrategyParameters
from vcov.modules.models.hyperparameters import Estimator, GluonHyperparameters, LstmHyperparameters
from vcov.modules.strategy.strategies import EquallyWeighted, RiskModels, LstmModels, GluonModels
from vcov.modules.strategy.performance import PerformanceStatistics

DATA_PATH = "..."


def pipeline(
        strategy: str,
        strategy_parameters: StrategyParameters,
        path: Optional[str],
        hyperparameters: Optional[Union[LstmHyperparameters, GluonHyperparameters]] = None,
):
    log_file = f"{path}/log.txt"
    with open(log_file, 'a') as f:
        f.write(f'Started Execution: {datetime.datetime.now()}\n')

    with open(DATA_PATH + "clean/filtered_data_0.5_0.01.pickle", "rb") as f:
        data = pickle.load(f)

    with open(DATA_PATH + "top_20.json", "r") as f:
        selection = json.load(f)

    data = Assets(data[selection].loc[data.index > "2017-10-01"])
    try:
        if strategy == 'weighted':
            strategy = EquallyWeighted(
                data=data.prices,
                portfolio_value=strategy_parameters.portfolio_value,
                fee_multiplier=strategy_parameters.fee_multiplier,
                save_results=log_file,
            )

            equity_line = strategy.apply_strategy(
                returns_method=strategy_parameters.returns,
                optimize=strategy_parameters.optimize,
            )

        elif strategy == 'risk':
            strategy = RiskModels(
                data=data.prices,
                window=strategy_parameters.window,
                rebalancing=strategy_parameters.rebalancing,
                portfolio_value=strategy_parameters.portfolio_value,
                fee_multiplier=strategy_parameters.fee_multiplier,
                save_results=log_file,
            )

            equity_line = strategy.apply_strategy(
                covariance_method=strategy_parameters.covariance_model,
                returns_method=strategy_parameters.returns,
                optimize=strategy_parameters.optimize,
                **strategy_parameters.cov_params
            )

        elif strategy == 'lstm':
            strategy = LstmModels(
                data=data.prices,
                portfolio_value=strategy_parameters.portfolio_value,
                fee_multiplier=strategy_parameters.fee_multiplier,
                window=strategy_parameters.window,
                rebalancing=strategy_parameters.rebalancing,
                warmup_period=strategy_parameters.warmup_period,
                save_results=log_file,
            )

            equity_line = strategy.apply_strategy(
                returns_method=strategy_parameters.returns,
                optimize=strategy_parameters.optimize,
                epochs=hyperparameters.epochs,
                batch_size=hyperparameters.batch_size,
                length=hyperparameters.length,
                architecture=hyperparameters.architecture,
                dropout_rate=hyperparameters.dropout_rate,
                stopping_patience=hyperparameters.stopping_patience,
                bidirectional=hyperparameters.bidirectional,
                merge_mode=hyperparameters.merge_mode,
                convolutional_layer=hyperparameters.convolutional_layer,
                kernel_size=hyperparameters.kernel_size,
                padding=hyperparameters.padding,
                filters=hyperparameters.filters,
                pool_size=hyperparameters.pool_size,
            )

        elif strategy == 'gluon':
            strategy = GluonModels(
                data=data.prices,
                portfolio_value=strategy_parameters.portfolio_value,
                fee_multiplier=strategy_parameters.fee_multiplier,
                window=strategy_parameters.window,
                rebalancing=strategy_parameters.rebalancing,
                warmup_period=strategy_parameters.warmup_period,
                save_results=log_file,
            )

            equity_line = strategy.apply_strategy(
                returns_method=strategy_parameters.returns,
                optimize=strategy_parameters.optimize,
                parameters=hyperparameters,
            )

        else:
            raise ValueError(f"Unknown strategy {strategy}")

        # Assess the performance
        stats = PerformanceStatistics()
        with open(f'{path}/performance.json', 'w') as f:
            json.dump(stats.get_performance_statistics(prices=equity_line), f)

        # Save run info
        history = {str(k.date()): {i: asdict(t) for i, t in v.items()} for k, v in strategy.trading.history.items()}
        with open(f'{path}/trade_history.json', 'w') as f:
            json.dump(history, f)

        with open(f'{path}/log.txt', 'a') as f:
            f.write(f'Fees: {strategy.trading.accumulated_fees}\n')
            f.write(f'Cash: {strategy.cash}\n')

    except Exception as e:
        print(strategy, strategy_parameters, e, sep="\n")
        with open(f'{path}/log.txt', 'a') as f:
            f.write(f'{e}\n')
