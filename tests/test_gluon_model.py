from vcov.modules.models.gluon_model import trainer_from_params, distribution_output_from_params, \
    get_multivariate_estimator, get_gp_estimator
from vcov.modules.models.hyperparameters import GluonHyperparameters

from gluonts.mx.trainer import Trainer
from gluonts.mx.distribution import LowrankMultivariateGaussianOutput, MultivariateGaussianOutput

from gluonts.model.deepvar import DeepVAREstimator
from gluonts.model.gpvar import GPVAREstimator


def test_trainer_from_params():
    params = GluonHyperparameters(
        cell_type="lstm",
        batch_size=24,
        num_cells=20,
        num_layers=2,
        epochs=20
    )
    trainer = trainer_from_params(params)
    assert isinstance(trainer, Trainer)
    assert trainer.epochs == 20
    assert trainer.num_batches_per_epoch == 5
    assert trainer.batch_size == 24
    assert trainer.learning_rate == 0.01
    assert trainer.patience == 5
    assert trainer.hybridize is False


def test_distribution_output_from_params():
    params = GluonHyperparameters(
        cell_type="lstm",
        batch_size=24,
        num_cells=20,
        num_layers=2,
        epochs=20,
        rank=4
    )
    dist = distribution_output_from_params(params, target_dim=2, low_rank=False)
    assert isinstance(dist, MultivariateGaussianOutput)
    dist = distribution_output_from_params(params, target_dim=2, low_rank=True)
    assert isinstance(dist, LowrankMultivariateGaussianOutput)
    assert dist.rank == 2


def test_get_multivariate_estimator():
    params = GluonHyperparameters(
        cell_type="lstm",
        batch_size=24,
        num_cells=20,
        num_layers=2,
        epochs=20,
        rank=4,
        low_rank=True,
    )
    model = get_multivariate_estimator(
        params,
        target_dim=4,
        prediction_length=1,
        freq='D',
    )
    assert isinstance(model, DeepVAREstimator)
    assert model.batch_size == 24
    assert model.cell_type == "lstm"
    assert model.num_cells == 20
    assert model.num_layers == 2
    assert model.dropout_rate == 0.1
    assert model.trainer.learning_rate == 0.1


def test_get_gp_estimator():
    params = GluonHyperparameters(
        cell_type="lstm",
        batch_size=24,
        num_cells=20,
        num_layers=2,
        epochs=20,
        rank=4
    )
    model = get_gp_estimator(
        params,
        target_dim=4,
        prediction_length=1,
        freq='D',
    )
    assert isinstance(model, GPVAREstimator)
    assert model.batch_size == 24
    assert model.cell_type == "lstm"
    assert model.num_cells == 20
    assert model.num_layers == 2
    assert model.dropout_rate == 0.1
    assert model.trainer.learning_rate == 0.01
