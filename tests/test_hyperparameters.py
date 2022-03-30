from vcov.modules.models.hyperparameters import (
    LstmHyperparameters,
    TCNHyperparameters,
    GluonHyperparameters,
    Estimator
)


def test_lstm_hyperparameters():
    params = LstmHyperparameters(
        epochs=50,
        batch_size=2,
        length=4,
        architecture=(20, 10),
        dropout_rate=0.1,
        stopping_patience=5,

        # Bidirectional LSTM params
        bidirectional=False,
        merge_mode="TEST",

        # Conv-Lstm Params
        convolutional_layer=False,
        kernel_size=5,
        padding=5,
        filters=2,
        pool_size=2,
    )

    assert hasattr(params, 'epochs')
    assert hasattr(params, 'batch_size')
    assert hasattr(params, 'length')
    assert hasattr(params, 'architecture')
    assert hasattr(params, 'dropout_rate')
    assert hasattr(params, 'stopping_patience')
    assert hasattr(params, 'bidirectional')
    assert hasattr(params, 'merge_mode')
    assert hasattr(params, 'convolutional_layer')
    assert hasattr(params, 'kernel_size')
    assert hasattr(params, 'padding')
    assert hasattr(params, 'filters')
    assert hasattr(params, 'pool_size')

    assert isinstance(params.epochs, int)
    assert isinstance(params.batch_size, int)
    assert isinstance(params.length, int)
    assert isinstance(params.architecture, tuple)
    assert isinstance(params.dropout_rate, float)
    assert isinstance(params.stopping_patience, int)
    assert isinstance(params.bidirectional, bool)
    assert isinstance(params.merge_mode, str)
    assert isinstance(params.convolutional_layer, bool)
    assert isinstance(params.kernel_size, int)
    assert isinstance(params.padding, int)
    assert isinstance(params.filters, int)
    assert isinstance(params.pool_size, int)

    assert params.epochs == 50
    assert params.batch_size == 2
    assert params.length == 4
    assert params.architecture == (20, 10)
    assert params.dropout_rate == 0.1
    assert params.stopping_patience == 5
    assert params.bidirectional is False
    assert params.merge_mode == 'TEST'
    assert params.convolutional_layer is False
    assert params.kernel_size == 5
    assert params.padding == 5
    assert params.filters == 2
    assert params.pool_size == 2


def test_lstm_hyperparameters_to_dict():
    params = LstmHyperparameters(
        epochs=50,
        batch_size=2,
        length=4,
        architecture=(20, 10),
        dropout_rate=0.1,
        stopping_patience=5,

        # Bidirectional LSTM params
        bidirectional=False,
        merge_mode="TEST",

        # Conv-Lstm Params
        convolutional_layer=False,
        kernel_size=5,
        padding=5,
        filters=2,
        pool_size=2,
    ).to_dict()

    assert isinstance(params, dict)
    assert len(params) == 13


def test_tcn_hyperparameters():
    params = TCNHyperparameters(
        nb_filters=2,
        kernel_size=3,
        nb_stacks=4,
        dilations=(8, 16),
        padding="TEST",
        dropout=0.1
    )

    assert hasattr(params, 'nb_filters')
    assert hasattr(params, 'kernel_size')
    assert hasattr(params, 'nb_stacks')
    assert hasattr(params, 'dilations')
    assert hasattr(params, 'padding')
    assert hasattr(params, 'dropout')

    assert params.nb_filters == 2
    assert params.kernel_size == 3
    assert params.nb_stacks == 4
    assert params.dilations == (8, 16)
    assert params.padding == "TEST"
    assert params.dropout == 0.1


def test_tcn_hyperparameters_to_dict():
    params = TCNHyperparameters(
        nb_filters=2,
        kernel_size=3,
        nb_stacks=4,
        dilations=(8, 16),
        padding="TEST",
        dropout=0.1
    ).to_dict()

    assert isinstance(params, dict)
    assert len(params) == 6


def test_gluon_hyperparameters():
    params = GluonHyperparameters(
        estimator=Estimator.VAR,
        cell_type="lstm",
        batch_size=16,
        epochs=100,
        num_cells=40,
        num_layers=2,
        dropout_rate=0.1,
        patience=5,
        learning_rate=1e-3,
        learning_rate_fullrank=1e-5,
        minimum_learning_rate=1e-5,
        rank=10,
        num_eval_samples=400,
        num_batches_per_epoch=100,
        conditioning_length=100,
        hybridize=False,
        target_dim_sample=2,
        lags_seq=[2, 4],
        scaling=False,
        context_length=12
    )

    assert params.estimator == Estimator.VAR
    assert params.cell_type == 'lstm'
    assert params.batch_size == 16
    assert params.epochs == 100
    assert params.num_cells == 40
    assert params.num_layers == 2
    assert params.dropout_rate == 0.1
    assert params.patience == 5
    assert params.learning_rate == 1e-3
    assert params.learning_rate_fullrank == 1e-5
    assert params.minimum_learning_rate == 1e-5
    assert params.rank == 10
    assert params.num_eval_samples == 400
    assert params.num_batches_per_epoch == 100
    assert params.conditioning_length == 100
    assert params.hybridize is False
    assert params.target_dim_sample == 2
    assert params.lags_seq == [2, 4]
    assert params.scaling is False
    assert params.context_length == 12


def test_gluon_hyperparameters_to_dict():
    params = GluonHyperparameters(
        estimator=Estimator.VAR,
        cell_type="lstm",
        batch_size=16,
        num_cells=40,
        num_layers=2,
        epochs=100,
        dropout_rate=0.1,
        patience=5,
        learning_rate=1e-3,
        learning_rate_fullrank=1e-5,
        minimum_learning_rate=1e-5,
        rank=10,
        num_eval_samples=400,
        num_batches_per_epoch=100,
        conditioning_length=100,
        hybridize=False,
        target_dim_sample=2,
        lags_seq=[2, 4],
        scaling=False,
        context_length=12
    ).to_dict()

    assert isinstance(params, dict)
    assert len(params) == 22
