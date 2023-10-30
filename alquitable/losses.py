import inspect
import sys

from keras_core import Loss, ops


class MeanSquaredDiffError(Loss):
    def call(self, y_true, y_pred):
        return ops.mean(ops.abs(ops.square(y_pred) - ops.square(y_true)))


class MeanSquaredDiffLogError(Loss):
    def call(self, y_true, y_pred):
        return ops.mean(
            ops.abs(
                ops.log(ops.square(y_true) + 1)
                - ops.log(ops.square(y_pred) + 1)
            )
        )


class MeanCubicError(Loss):
    def call(self, y_true, y_pred):
        erro = y_pred - y_true
        ce = ops.multiply(ops.square(erro), ops.abs(erro))
        mce = ops.mean(ce)
        return mce


module = inspect.currentframe().f_globals["__name__"]


# Define a predicate function to filter out Loss subclasses
def is_loss_subclass(cls):
    return inspect.isclass(cls) and issubclass(cls, Loss)


# Get all the Loss subclasses defined in the module
loss_functions = inspect.getmembers(sys.modules[module], is_loss_subclass)
loss_functions = [(n, f) for n, f in loss_functions if n != "Loss"]
