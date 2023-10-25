import inspect
import sys

import keras_core
from keras_core import ops

OFFSET = 30

@keras_core.saving.register_keras_serializable()
def weighted_loss(y_true, y_pred):
    # Calculate weights based on the absolute difference
    #   between true and predicted values
    weights = ops.abs(y_true - y_pred) 
    loss = ops.square(y_true - y_pred)  # Calculate the squared loss
    # Multiply the weights with the loss
    weighted_loss = ops.multiply(weights, loss)  
    return ops.mean(weighted_loss)  # Calculate the mean of the weighted loss

@keras_core.saving.register_keras_serializable()
def mean_squared_diff_error(y_true, y_pred):
    return ops.mean(ops.abs(ops.square(y_pred) - ops.square(y_true)))

@keras_core.saving.register_keras_serializable()
def mean_squared_error(y_true, y_pred):
    erro = y_pred - y_true
    se = ops.square(erro)
    mse = ops.mean(se)
    return mse

@keras_core.saving.register_keras_serializable()
def mean_squared_diff_log_error(y_true, y_pred):
    return ops.mean(
        ops.abs(
            ops.log(ops.square(y_true) + 1) - ops.log(ops.square(y_pred) + 1)
        )
    )


@keras_core.saving.register_keras_serializable()
def mean_absolute_percentage_error(y_true, y_pred):
    return (ops.abs(y_pred - y_true) + OFFSET) / (y_true + OFFSET)


module = inspect.currentframe().f_globals["__name__"]
# Get all the functions defined in the module
loss_functions = inspect.getmembers(sys.modules[module], inspect.isfunction)
