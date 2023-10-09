import numpy as np
import pytest
from forecat import CNNArch

from alquitable.losses import loss_functions

X_timeseries = 168
Y_timeseries = 24
n_features_train = 18
n_features_predict = 1

input_args = {
    "X_timeseries": X_timeseries,
    "Y_timeseries": Y_timeseries,
    "n_features_train": n_features_train,
    "n_features_predict": n_features_predict,
}

foreCNN = CNNArch(**input_args)
foreCNN_model = foreCNN.architeture()

input_shape = (10, *foreCNN_model.input_shape[1:])
output_shape = (10, *foreCNN_model.output_shape[1:])
print(input_shape)
dummy_X = np.full(input_shape, 1)
dummy_Y = np.full(output_shape, 1)


@pytest.mark.parametrize("loss", loss_functions)
def test_model_making(loss):
    loss_name, loss_funtion = loss
    foreCNN_model.compile(loss=loss_funtion)

    foreCNN_model.fit(dummy_X, dummy_Y, epochs=1)


# TODO: add tests to test for results in losses