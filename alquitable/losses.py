import inspect
import sys

from keras_core import Loss, ops


class MeanSquaredDiffError(Loss):
    def __init__(self, name="mean_squared_diff_error", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        return ops.mean(ops.abs(ops.square(y_pred) - ops.square(y_true)))


class MeanSquaredDiffLogError(Loss):
    def __init__(self, name="mean_squared_diff_log_error", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        return ops.mean(
            ops.abs(
                ops.log(ops.square(y_true) + 1)
                - ops.log(ops.square(y_pred) + 1)
            )
        )


class MeanCubicError(Loss):
    def __init__(self, name="mean_cubic_error", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        erro = y_pred - y_true
        ce = ops.multiply(ops.square(erro), ops.abs(erro))
        mce = ops.mean(ce)
        return mce

class MeanPercentualDiffError(Loss):
    def __init__(self, name="mean_percentual_diff_error", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        erro = y_pred - y_true
        m_mask = erro<=0
        s_mask = erro>=0

        erro_m = ops.abs(erro[m_mask])
        erro_s = ops.abs(erro[s_mask])
        true_m = y_true[m_mask]
        true_s = y_true[s_mask]

        # Check for zeros before division
        erro_perc_m = ops.where(ops.not_equal(true_m, 0), ops.divide(erro_m, true_m), 0)*100
        erro_perc_s = ops.where(ops.not_equal(true_s, 0), ops.divide(erro_s, true_s), 0)*100



        res = ops.mean([erro_perc_m, erro_perc_s])

        return res

class MeanPercentualDiffNoZeroError(Loss):
    def __init__(self, name="mean_percentual_diff_no_zero_error", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        erro = y_pred - y_true
        m_mask = erro<0
        s_mask = erro>0

        erro_m = ops.abs(erro[m_mask])
        erro_s = ops.abs(erro[s_mask])
        true_m = y_true[m_mask]
        true_s = y_true[s_mask]

        # Check for zeros before division
        erro_perc_m = ops.where(ops.not_equal(true_m, 0), ops.divide(erro_m, true_m), 0)*100
        erro_perc_s = ops.where(ops.not_equal(true_s, 0), ops.divide(erro_s, true_s), 0)*100



        res = ops.mean([erro_perc_m, erro_perc_s])

        return res
module = inspect.currentframe().f_globals["__name__"]


# Define a predicate function to filter out Loss subclasses
def is_loss_subclass(cls):
    return inspect.isclass(cls) and issubclass(cls, Loss)


# Get all the Loss subclasses defined in the module
loss_functions = inspect.getmembers(sys.modules[module], is_loss_subclass)
loss_functions = [(n, f) for n, f in loss_functions if n != "Loss"]
ALL_LOSSES_DICT = {n:f for n, f in loss_functions if n != "Loss"}