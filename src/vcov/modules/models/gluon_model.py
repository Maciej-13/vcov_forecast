from mxnet.context import cpu
from gluonts.mx.trainer import Trainer
from gluonts.mx.distribution import LowrankMultivariateGaussianOutput, MultivariateGaussianOutput
from gluonts.mx.distribution.lowrank_gp import LowrankGPOutput
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.model.gpvar import GPVAREstimator

from vcov.modules.models.hyperparameters import GluonHyperparameters


def trainer_from_params(parameters: GluonHyperparameters) -> Trainer:
    return Trainer(
        ctx=cpu(0),
        epochs=parameters.epochs,
        batch_size=parameters.batch_size,
        learning_rate=parameters.learning_rate if parameters.low_rank else parameters.learning_rate_fullrank,
        minimum_learning_rate=parameters.minimum_learning_rate,
        patience=parameters.patience,
        num_batches_per_epoch=parameters.num_batches_per_epoch,
        hybridize=parameters.hybridize,
    )


def distribution_output_from_params(parameters: GluonHyperparameters, target_dim: int, low_rank: bool):
    if not low_rank:
        likelihood = MultivariateGaussianOutput(dim=target_dim)
    else:
        likelihood = LowrankMultivariateGaussianOutput(
            dim=target_dim,
            rank=min(parameters.rank, target_dim) if parameters.rank is not None else target_dim,
        )
    return likelihood


def get_multivariate_estimator(parameters: GluonHyperparameters, target_dim: int, prediction_length: int, freq: str,
                               copula: bool = True, scaling: bool = False):
    distribution_output = distribution_output_from_params(
        target_dim=target_dim,
        low_rank=parameters.low_rank,
        parameters=parameters
    )

    return DeepVAREstimator(
        target_dim=target_dim,
        num_cells=parameters.num_cells,
        num_layers=parameters.num_layers,
        batch_size=parameters.batch_size,
        dropout_rate=parameters.dropout_rate,
        prediction_length=prediction_length,
        context_length=parameters.context_length,
        cell_type=parameters.cell_type,
        freq=freq,
        pick_incomplete=False,
        distr_output=distribution_output,
        conditioning_length=parameters.conditioning_length,
        trainer=trainer_from_params(
            parameters=parameters
        ),
        scaling=scaling,
        use_marginal_transformation=copula,
        lags_seq=parameters.lags_seq,
    )


def get_gp_estimator(parameters: GluonHyperparameters, target_dim: int, prediction_length: int, freq: str,
                     cdf: bool = True, scaling: bool = False):
    distribution_output = LowrankGPOutput(
        dim=target_dim,
        rank=min(parameters.rank, target_dim) if parameters.rank is not None else target_dim,
        dropout_rate=parameters.dropout_rate
    )

    return GPVAREstimator(
        target_dim=target_dim,
        num_cells=parameters.num_cells,
        num_layers=parameters.num_layers,
        batch_size=parameters.batch_size,
        dropout_rate=parameters.dropout_rate,
        prediction_length=prediction_length,
        context_length=parameters.context_length,
        cell_type=parameters.cell_type,
        target_dim_sample=parameters.target_dim_sample,
        lags_seq=parameters.lags_seq,
        conditioning_length=parameters.conditioning_length,
        scaling=scaling,
        freq=freq,
        use_marginal_transformation=cdf,
        distr_output=distribution_output,
        trainer=trainer_from_params(
            parameters=parameters
        )
    )
