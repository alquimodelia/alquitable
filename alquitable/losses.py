import inspect
import sys

import keras
from keras import Loss, ops


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

class SumSquaredError(Loss):
    def __init__(self, name="sum_squared_error", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        erro = y_pred - y_true
        se = ops.square(erro)
        sse = ops.sum(se)
        return sse

class PercentageMeanError(Loss):
    def __init__(self, name="percentage_mean_error",absolute=True, epsilon=None,**kwargs):
        self.absolute=absolute
        self.epsilon= epsilon or ops.convert_to_tensor(keras.backend.epsilon())

        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        erro = y_pred - y_true
        if self.absolute:
            erro=ops.abs(erro)
        mean_error=ops.mean(erro)
        erro_per = mean_error/ops.mean(y_true)
        return erro_per*100


class MeanPercentageError(Loss):
    def __init__(self, name="mean_percentage_error",absolute=True, epsilon=None,**kwargs):
        self.absolute=absolute
        self.epsilon= epsilon or ops.convert_to_tensor(keras.backend.epsilon())

        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        erro = y_pred - y_true
        if self.absolute:
            erro=ops.abs(erro)

        erro_per = ops.where(ops.not_equal(y_true, 0), ops.divide(erro, y_true), ops.divide(erro, ops.mean(y_true)))*100
        # erro_per_m = ops.where(ops.not_equal(true_m, 0), ops.divide(erro_m, true_m), ops.minimum(ops.divide(erro_m, ops.maximum(ops.mean(true_m),epsilon)),100))*100 if ops.size(pred_m) else ops.convert_to_tensor(0)

        return ops.mean(erro_per)

class MeanPercentualDiffError(Loss):
    def __init__(self, name="mean_percentual_diff_error", epsilon=None,**kwargs):
        self.epsilon= epsilon or ops.convert_to_tensor(keras.backend.epsilon())

        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        erro = y_pred - y_true
        m_mask = erro<0
        s_mask = erro>=0

        erro_m = ops.abs(erro[m_mask])
        erro_s = ops.abs(erro[s_mask])
        true_m = y_true[m_mask]
        true_s = y_true[s_mask]
        pred_s = y_pred[s_mask]
        pred_m = y_pred[m_mask]


        m_not_zero = ops.not_equal(true_m, 0) #& ops.not_equal(pred_m, 0)
        s_not_zero = ops.not_equal(true_s, 0) #& ops.not_equal(pred_s, 0)

        erro_per_m = ops.where(m_not_zero, ops.divide(erro_m, true_m), 0)*100 if ops.size(pred_m) else ops.convert_to_tensor(0)
        erro_per_s = ops.where(s_not_zero, ops.divide(erro_s, true_s), 0)*100 if ops.size(pred_s) else ops.convert_to_tensor(0)

        # TODO: check if you can inerit from the othe class
        # erro_per_m = ops.where(ops.not_equal(true_m, 0), ops.divide(erro_m, true_m), ops.minimum(ops.divide(erro_m, ops.maximum(ops.mean(true_m),epsilon)),100))*100 if ops.size(pred_m) else ops.convert_to_tensor(0)
        # erro_per_s = ops.where(ops.not_equal(true_s, 0), ops.divide(erro_s, true_s), ops.minimum(ops.divide(erro_s, ops.maximum(ops.mean(true_s),epsilon)),100))*100 if ops.size(pred_s) else ops.convert_to_tensor(0)


        erro_per_m=ops.mean(erro_per_m)
        erro_per_s=ops.mean(erro_per_s)    




        return ops.mean([erro_per_m, erro_per_s])

class MeanPercentualDiffNoZeroError(Loss):
    def __init__(self, name="mean_percentual_diff_no_zero_error", epsilon=None,**kwargs):
        self.epsilon= epsilon or ops.convert_to_tensor(keras.backend.epsilon())

        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        erro = y_pred - y_true
        m_mask = erro<0
        s_mask = erro>0

        erro_m = ops.abs(erro[m_mask])
        erro_s = ops.abs(erro[s_mask])
        true_m = y_true[m_mask]
        true_s = y_true[s_mask]
        pred_s = y_pred[s_mask]
        pred_m = y_pred[m_mask]

        m_not_zero = ops.not_equal(true_m, 0) #& ops.not_equal(pred_m, 0)
        s_not_zero = ops.not_equal(true_s, 0) #& ops.not_equal(pred_s, 0)

        erro_per_m = ops.where(m_not_zero, ops.divide(erro_m, true_m), 0)*100 if ops.size(pred_m) else ops.convert_to_tensor(0)
        erro_per_s = ops.where(s_not_zero, ops.divide(erro_s, true_s), 0)*100 if ops.size(pred_s) else ops.convert_to_tensor(0)
        erro_per_m=ops.mean(erro_per_m)
        erro_per_s=ops.mean(erro_per_s)    



        return ops.mean([erro_per_m, erro_per_s])



class GENEO_Keras_Loss(Loss):
    def __init__(self, name="geneo_loss",loss_to_use=None, alpha=1, rho=3, epsilon=0.1, gamma=1,**kwargs):
        if loss_to_use is None:
            loss_to_use = keras.losses.MeanSquaredError()
        self.loss_to_use=loss_to_use
        self.alpha = alpha
        self.rho = rho
        self.epsilon = epsilon
        self.gamma = gamma

        super().__init__(name=name, **kwargs)


    def cvx_loss(self, cvx_coeffs):
        """
        Penalizes non-positive convex parameters;
        The last cvx coefficient is calculated in function of the previous ones: phi_n = 1 - sum_i^N-1(phi_i)

        This results from the the relaxation of the cvx restriction: sum(cvx_coeffs) == 1
        """
        if len(cvx_coeffs) == 0:
            return 0

        last_phi = [phi_name for phi_name in cvx_coeffs if not cvx_coeffs[phi_name].requires_grad][0]

        return self.rho * (ops.sum([ops.relu(-phi) for phi in cvx_coeffs])
                            + ops.relu(-(1 - ops.sum(cvx_coeffs) + cvx_coeffs[-1])))

    def positive_regularizer(self, params):
        """
        Penalizes non positive parameters
        """
        if len(params) == 0:
            return 0
        return self.rho * ops.sum([ops.relu(-g) for g in params])

    def call(self, y_true, y_pred, cvx_coeffs, geneo_params):
        dense_criterion = self.loss(y_true, y_pred)
        cvx_penalty = self.cvx_loss(cvx_coeffs)
        non_positive_penalty = self.positive_regularizer(geneo_params)
        return dense_criterion + cvx_penalty + non_positive_penalty



module = inspect.currentframe().f_globals["__name__"]


# Define a predicate function to filter out Loss subclasses
def is_loss_subclass(cls):
    return inspect.isclass(cls) and issubclass(cls, Loss)


# Get all the Loss subclasses defined in the module
loss_functions = inspect.getmembers(sys.modules[module], is_loss_subclass)
loss_functions = [(n, f) for n, f in loss_functions if n != "Loss"]
ALL_LOSSES_DICT = {n:f for n, f in loss_functions if n != "Loss"}