import importlib

import keras
from keras import Loss, ops


class AdvanceLoss(Loss):
    """
    Wrapper on Loss to be able to have a Loss wich can use different losses.
    """

    def __init__(self, loss_to_use=None, name=None, **kwargs):
        if loss_to_use is None:
            loss_to_use = keras.losses.MeanSquaredError()
        if isinstance(loss_to_use, dict):
            module = importlib.import_module(loss_to_use["module"])
            # Assuming the loss func is a Loss
            loss_cls = getattr(module, loss_to_use["class_name"])
            loss_to_use = loss_cls.from_config(config=loss_to_use["config"])
        self.loss_to_use = loss_to_use
        name = f"{name}_{loss_to_use.name}"

        super().__init__(name=name, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "loss_to_use": self.loss_to_use,
        }


class MirrorWeights(AdvanceLoss):
    """
    Create a custom loss function that assigns weights to predictions and true values.

    The loss function returned by this function calculates the difference between
    predictions and true values, and assigns weights based on whether the predictions
    are greater than or lower than the true values. If `weight_on_surplus` is True,
    the function assigns higher weights to predictions that are greater than the true
    values. Otherwise, it assigns higher weights to predictions that are lower than
    the true values.

    The weights are determined by the following rules:

    - If `ratio` is None and `weight_on_surplus` is True, the function assigns a weight
      of 1 to predictions that are greater than the true values, and a weight of 0 to
      predictions that are lower than or equal to the true values.
    - If `ratio` is None and `weight_on_surplus` is False, the function assigns a weight
      of 0 to predictions that are greater than the true values, and a weight of 1 to
      predictions that are lower than or equal to the true values.
    - If `ratio` is not None, the function assigns a weight equal to `ratio` to predictions
      that are greater than the true values or lower than the true values, depending on
      the value of `weight_on_surplus`.

    The loss function returned by this function uses `loss_to_use` to calculate the
    final loss. If `loss_to_use` is None, the function uses MeanSquaredError as the
    loss function.

    Parameters
    ----------
    ratio : float, optional
        The ratio to use when assigning weights. If None, the function uses a default
        ratio of 1.
    loss_to_use : callable, optional
        The loss function to use when calculating the final loss. If None, the function
        uses MeanSquaredError.
    weight_on_surplus : bool, optional
        Whether to assign weights based on whether the predictions are greater than
        the true values. If True, the function assigns higher weights to predictions
        that are greater than the true values. If False, the function assigns higher
        weights to predictions that are lower than the true values.

    Returns
    -------
    callable
        A loss function that assigns weights to predictions and true values based on
        the rules described above, and uses `loss_to_use` to calculate the final loss.
    """

    def __init__(
        self,
        ratio=None,
        ratio_m=None,
        ratio_s=None,
        loss_to_use=None,
        weight_on_surplus=True,
        name="mirror_weights",
        ratio_on_weigths=False,
        **kwargs,
    ):
        self.ratio_on_weigths = ratio_on_weigths
        self.ratio = ratio
        self.ratio_m = ratio_m
        self.ratio_s = ratio_s
        self.weight_on_surplus = weight_on_surplus
        super().__init__(name=name, loss_to_use=loss_to_use, **kwargs)

    def call(self, y_true, y_pred):
        diff = y_pred - y_true
        greater = ops.greater(diff, 0)
        greater = ops.cast(greater, keras.backend.floatx())
        greater = greater + 1

        surplus_values = 1
        missing_values = 1

        if self.ratio:
            if not self.ratio_on_weigths:
                if self.ratio > 0:
                    surplus_values = 1 if self.ratio is None else self.ratio
                else:
                    missing_values = 1 if self.ratio is None else self.ratio
            else:
                if self.ratio > 0:
                    surplus_values = ops.maximum(0.0, -diff) * abs(self.ratio)
                else:
                    missing_values = ops.maximum(0.0, diff) * abs(self.ratio)

        if self.ratio is None:
            if self.weight_on_surplus:
                surplus_values = ops.maximum(0.0, -diff)
            else:
                missing_values = ops.maximum(0.0, diff)

        if self.ratio_m:
            missing_values = missing_values * self.ratio_m
        if self.ratio_s:
            surplus_values = surplus_values * self.ratio_s

        weights = ops.where(greater == 1, surplus_values, missing_values)

        return self.loss_to_use(y_true, y_pred, sample_weight=weights)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "ratio": self.ratio,
            "ratio_m": self.ratio_m,
            "ratio_s": self.ratio_s,
            "weight_on_surplus": self.weight_on_surplus,
            "ratio_on_weigths": self.ratio_on_weigths,
        }


class MirrorLoss(AdvanceLoss):
    # Cuts of the diference in sampling number around the surplus and missing error
    def __init__(
        self,
        ratio=None,
        ratio_m=None,
        ratio_s=None,
        loss_to_use=None,
        weight_on_surplus=True,
        name="mirror_loss",
        **kwargs,
    ):
        self.loss_to_use = loss_to_use
        self.ratio = ratio
        self.ratio_m = ratio_m
        self.ratio_s = ratio_s
        self.weight_on_surplus = weight_on_surplus
        super().__init__(name=name, loss_to_use=loss_to_use, **kwargs)

    def _mean_losses(self, m_loss, s_loss):
        # ratio = self.ratio or 1
        # m_ratio = ratio
        # s_ratio = m_ratio - 1
        m_ratio = self.ratio_m or 1
        s_ratio = self.ratio_s or 1

        return ((m_loss * m_ratio) + (s_loss * s_ratio)) / (m_ratio + s_ratio)

    def call(self, y_true, y_pred):
        erro = y_pred - y_true
        m_mask = erro < 0
        s_mask = erro >= 0

        true_m = y_true[m_mask]
        true_s = y_true[s_mask]
        pred_m = y_pred[m_mask]
        pred_s = y_pred[s_mask]

        m_loss = self.loss_to_use(true_m, pred_m)
        s_loss = self.loss_to_use(true_s, pred_s)

        return self._mean_losses(m_loss, s_loss)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "ratio": self.ratio,
            "ratio_m": self.ratio_m,
            "ratio_s": self.ratio_s,
            "ratio_on_weigths": self.ratio_on_weigths,
        }


class MirrorLossNorm(AdvanceLoss):
    # Cuts of the diference in sampling number around the surplus and missing error
    def __init__(
        self,
        ratio=None,
        ratio_m=None,
        ratio_s=None,
        loss_to_use=None,
        weight_on_surplus=True,
        name="mirror_loss_norm",
        **kwargs,
    ):
        self.ratio = ratio
        self.ratio_m = ratio_m
        self.ratio_s = ratio_s
        self.weight_on_surplus = weight_on_surplus
        super().__init__(name=name, loss_to_use=loss_to_use, **kwargs)

    def _mean_losses(self, m_loss, s_loss):
        # ratio = self.ratio or 1
        # m_ratio = ratio
        # s_ratio = m_ratio - 1
        m_ratio = self.ratio_m or 1
        s_ratio = self.ratio_s or 1

        return ((m_loss * m_ratio) + (s_loss * s_ratio)) / (m_ratio + s_ratio)

    def call(self, y_true, y_pred):
        erro = y_pred - y_true
        m_mask = erro < 0
        s_mask = erro >= 0

        true_m = y_true[m_mask]
        true_s = y_true[s_mask]
        pred_m = y_pred[m_mask]
        pred_s = y_pred[s_mask]

        m_loss = self.loss_to_use(true_m, pred_m)
        m_loss = (m_loss / (4824.70**2)) * 100

        s_loss = self.loss_to_use(true_s, pred_s)

        s_loss = (s_loss / (18903.6**2)) * 100

        return self._mean_losses(m_loss, s_loss)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "ratio": self.ratio,
            "ratio_m": self.ratio_m,
            "ratio_s": self.ratio_s,
            "ratio_on_weigths": self.ratio_on_weigths,
        }


class MirrorLossNormMinMax(AdvanceLoss):
    # Cuts of the diference in sampling number around the surplus and missing error
    def __init__(
        self,
        ratio=None,
        ratio_m=None,
        ratio_s=None,
        loss_to_use=None,
        weight_on_surplus=True,
        name="mirror_loss_norm_min_max",
        **kwargs,
    ):
        self.ratio = ratio
        self.ratio_m = ratio_m
        self.ratio_s = ratio_s
        self.weight_on_surplus = weight_on_surplus
        super().__init__(name=name, loss_to_use=loss_to_use, **kwargs)

    def _mean_losses(self, m_loss, s_loss):
        # ratio = self.ratio or 1
        # m_ratio = ratio
        # s_ratio = m_ratio - 1
        m_ratio = self.ratio_m or 1
        s_ratio = self.ratio_s or 1

        return ((m_loss * m_ratio) + (s_loss * s_ratio)) / (m_ratio + s_ratio)

    def call(self, y_true, y_pred):
        erro = y_pred - y_true
        m_mask = erro < 0
        s_mask = erro >= 0

        true_m = y_true[m_mask]
        true_s = y_true[s_mask]
        pred_m = y_pred[m_mask]
        pred_s = y_pred[s_mask]

        m_feat_max = 4824.70**2
        m_feat_min = 0**2

        s_feat_max = 18903.6**2
        s_feat_min = 3118.29**2

        m_loss = self.loss_to_use(true_m, pred_m)
        m_loss = ((m_loss - m_feat_min) / (m_feat_max - m_feat_min)) * 100
        s_loss = self.loss_to_use(true_s, pred_s)
        s_loss = ((s_loss - s_feat_min) / (s_feat_max - s_feat_min)) * 100

        return self._mean_losses(m_loss, s_loss)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "ratio": self.ratio,
            "ratio_m": self.ratio_m,
            "ratio_s": self.ratio_s,
            "ratio_on_weigths": self.ratio_on_weigths,
        }


class MirrorPercentage(AdvanceLoss):
    # Only works properly for losses where the return is the same dimension as predict: rmse, mae
    def __init__(
        self,
        ratio=None,
        loss_to_use=None,
        weight_on_surplus=True,
        name="mirror_percentage",
        funtion_to_dimention=None,
        **kwargs,
    ):
        self.ratio = ratio
        self.weight_on_surplus = weight_on_surplus
        self.funtion_to_dimention = funtion_to_dimention

        super().__init__(name=name, loss_to_use=loss_to_use, **kwargs)
        if funtion_to_dimention is None:
            if isinstance(self.loss_to_use, keras.losses.MeanSquaredError):
                self.funtion_to_dimention = ops.sqrt

    def _calculate_loss(
        self, true_values, pred_values, prefix, overal_avg_true=None
    ):
        loss = self.loss_to_use(true_values, pred_values)

        # if overal_avg_pred:
        #     # Compute the averages of the predicted values
        #     avg_pred_values=overal_avg_pred
        # else:

        if self.funtion_to_dimention:
            loss = self.funtion_to_dimention(loss)

        epsilon = 1e-8

        # Normalize the losses
        norm_loss = (
            ops.abs((loss + epsilon) / (overal_avg_true + epsilon)) * 100
        )

        return norm_loss

    def call(self, y_true, y_pred):
        erro = y_pred - y_true
        # overal_avg_pred = ops.mean(y_pred)
        overal_avg_true = ops.mean(y_true)

        m_mask = erro < 0
        s_mask = erro >= 0
        # z_mask = erro == 0

        true_m = y_true[m_mask]
        true_s = y_true[s_mask]
        pred_m = y_pred[m_mask]
        pred_s = y_pred[s_mask]

        # true_z = y_true[z_mask] # True values for erro == 0
        # pred_z = y_pred[z_mask] # Predicted values for erro == 0

        # Calculate the loss for m_mask, s_mask, and z_mask
        norm_m_loss = (
            self._calculate_loss(
                true_m, pred_m, "m", overal_avg_true=overal_avg_true
            )
            if ops.size(pred_m)
            else ops.convert_to_tensor(0)
        )
        norm_s_loss = (
            self._calculate_loss(
                true_s, pred_s, "s", overal_avg_true=overal_avg_true
            )
            if ops.size(pred_s)
            else ops.convert_to_tensor(0)
        )
        # norm_z_loss = self._calculate_loss(true_z, pred_z, 'z', overal_avg_pred=overal_avg_pred) if ops.size(pred_z) else ops.convert_to_tensor(0) # New loss calculation for erro == 0

        return ops.mean([norm_m_loss, norm_s_loss])

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "ratio": self.ratio,
            "ratio_m": self.ratio_m,
            "ratio_s": self.ratio_s,
            "ratio_on_weigths": self.ratio_on_weigths,
        }


class MirrorNormalized(AdvanceLoss):
    # Only works properly for losses where the return is the same dimension as predict: rmse, mae
    def __init__(
        self,
        ratio=None,
        loss_to_use=None,
        weight_on_surplus=True,
        name="mirror_normalized",
        funtion_to_dimention=None,
        **kwargs,
    ):
        self.ratio = ratio
        self.weight_on_surplus = weight_on_surplus
        self.funtion_to_dimention = funtion_to_dimention

        super().__init__(name=name, loss_to_use=loss_to_use, **kwargs)
        if funtion_to_dimention is None:
            if isinstance(self.loss_to_use, keras.losses.MeanSquaredError):
                self.funtion_to_dimention = ops.sqrt

    def call(self, y_true, y_pred):
        erro = y_pred - y_true
        # overal_avg_pred = ops.mean(y_pred)
        ops.mean(y_true)

        m_mask = erro < 0
        s_mask = erro >= 0
        # z_mask = erro == 0

        true_m = y_true[m_mask]
        true_s = y_true[s_mask]
        pred_m = y_pred[m_mask]
        pred_s = y_pred[s_mask]

        # Compute the means and variances of true and predicted values for both positive and negative errors
        m_mean, m_var = ops.mean(ops.concatenate([true_m, pred_m])), ops.var(
            ops.concatenate([true_m, pred_m])
        )
        s_mean, s_var = ops.mean(ops.concatenate([true_s, pred_s])), ops.var(
            ops.concatenate([true_s, pred_s])
        )
        # Add a small constant to avoid division by zero

        m_var += 1e-10
        s_var += 1e-10

        # Normalize the true and predicted values
        true_m_normalized = (((true_m - m_mean) / ops.sqrt(m_var)) + 1) * 100
        true_s_normalized = (((true_s - s_mean) / ops.sqrt(s_var)) + 1) * 100
        pred_m_normalized = (((pred_m - m_mean) / ops.sqrt(m_var)) + 1) * 100
        pred_s_normalized = (((pred_s - s_mean) / ops.sqrt(s_var)) + 1) * 100

        # Compute the loss on the normalized values
        m_loss = self.loss_to_use(true_m_normalized, pred_m_normalized)
        s_loss = self.loss_to_use(true_s_normalized, pred_s_normalized)

        return ops.mean([m_loss, s_loss])

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "ratio": self.ratio,
            "ratio_m": self.ratio_m,
            "ratio_s": self.ratio_s,
            "ratio_on_weigths": self.ratio_on_weigths,
        }


class DerivativeLoss(AdvanceLoss):
    def __init__(
        self, name="derivative_loss", loss_to_use=None, axis=-1, **kwargs
    ):
        self.axis = axis
        self.loss_to_use = loss_to_use
        super().__init__(name=name, loss_to_use=loss_to_use, **kwargs)

    def call(self, y_true, y_pred):
        derivative_truth = ops.diff(y_true, axis=self.axis)
        derivative_pred = ops.diff(y_pred, axis=self.axis)
        return self.loss_to_use(derivative_truth, derivative_pred)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "axis": self.axis,
        }


# class AdvanceSlopeLoss(Loss):
#     def __init__(self,name="advance_slope_loss",loss_to_use=None,**kwargs):
#         if loss_to_use is None:
#             loss_to_use = keras.losses.MeanSquaredError()
#         self.loss_to_use=loss_to_use
#         name = f"{name}_{loss_to_use.name}"
#         super().__init__(name=name, **kwargs)

#     def call(self, y_true, y_pred):
#         diff = y_pred - y_true
#         slope_loss = SlopeLoss()(y_pred, y_true)
#         derivative_truth = ops.diff(y_true.ravel())
#         derivative_pred = ops.diff(y_pred.ravel())

#         return self.loss_to_use(derivative_truth, derivative_pred)

#     def get_config(self):
#         base_config = super().get_config()
#         return {
#             **base_config,
#             "loss_to_use":self.loss_to_use,
#         }


class JointLoss(Loss):
    def __init__(
        self,
        name="joint_loss",
        loss_to_use=None,
        operation=None,
        operation_kwargs=None,
        **kwargs,
    ):
        if loss_to_use is None:
            loss_to_use = keras.losses.MeanSquaredError()

        if not isinstance(loss_to_use, list):
            loss_to_use = [loss_to_use]
        self.loss_to_use = loss_to_use
        self.operation = operation or ops.multiply

        self.operation_kwargs = operation_kwargs or {}

        name = f"{name}"
        for loss in loss_to_use:
            name += f"_{loss.name}"
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        result = None
        for loss in self.loss_to_use:
            if result is None:
                result = loss(y_true, y_pred)
            else:
                result = self.operation(
                    result, loss(y_true, y_pred), **self.operation_kwargs
                )
        return result

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "loss_to_use": self.loss_to_use,
            "operation": self.operation,
            "operation_kwargs": self.operation_kwargs,
        }


class NanValueLoss(AdvanceLoss):
    def __init__(
        self,
        loss_to_use=None,
        nan_value=None,
        name="nan_value_loss",
        nan_value_loss=-10,
        **kwargs,
    ):
        self.nan_value = nan_value
        self.nan_value_loss = nan_value_loss
        super().__init__(name=name, loss_to_use=loss_to_use, **kwargs)

    def call(self, y_true, y_pred):
        flatten_y_true = ops.ravel(y_true)
        flatten_y_pred = ops.ravel(y_pred)
        valid_mask = ops.not_equal(flatten_y_true, self.nan_value)
        # REPLACE zero with a negative number in true, then make the nan zero, then take indices of zero

        # ops.take(flatten_y_pred, [2, 3, 4, 5])
        # flatten_y_true_neg = ops.where(
        #     flatten_y_true == 0, flatten_y_true, self.nan_value_loss
        # )

        # flatten_y_true_neg = ops.where(valid_mask, flatten_y_true_neg, 0)
        # ops.nonzero(flatten_y_true_neg)

        # #s = ops.outer(flatten_y_true, valid_mask)
        masked_y_pred = ops.take(flatten_y_pred, ops.where(valid_mask))
        masked_y_true = flatten_y_true[valid_mask]

        # Apply the mask element-wise
        masked_y_true = ops.take(y_true, ops.where(valid_mask))
        masked_y_pred = ops.take(y_pred, ops.where(valid_mask))
        return self.loss_to_use(masked_y_true, masked_y_pred)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "nan_value": self.nan_value,
        }


class NanValueLossA(Loss):
    def __init__(
        self,
        loss_to_use=None,
        nan_value=None,
        name="nan_value_loss",
        **kwargs,
    ):
        self.nan_value = nan_value
        self.loss_to_use = loss_to_use or keras.losses.MeanSquaredError()
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):

        valid_mask = ops.not_equal(y_true, self.nan_value)
        return self.loss_to_use(y_true[valid_mask], y_pred[valid_mask])


class NanValueLossB(Loss):
    def __init__(
        self,
        loss_to_use=None,
        nan_value=None,
        name="nan_value_loss",
        nan_value_loss=-10,
        **kwargs,
    ):
        self.nan_value = nan_value
        self.nan_value_loss = nan_value_loss
        self.loss_to_use = loss_to_use
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):

        valid_mask = ops.not_equal(y_true, self.nan_value)
        valid_indices = ops.where(valid_mask)
        masked_y_pred = ops.take(y_pred, valid_indices)
        masked_y_true = ops.take(y_true, valid_indices)

        return self.loss_to_use(masked_y_true, masked_y_pred)


class NanValueLossZerod(AdvanceLoss):
    def __init__(
        self,
        loss_to_use=None,
        nan_value=None,
        name="nan_value_loss",
        mask_with_weights=False,
        **kwargs,
    ):
        self.nan_value = nan_value
        self.mask_with_weights = mask_with_weights
        super().__init__(name=name, loss_to_use=loss_to_use, **kwargs)

    def call(self, y_true, y_pred):
        # REPLACE zero with a negative number in true, then make the nan zero, then take indices of zero
        extra_args = {}
        if self.nan_value == self.nan_value:
            y_true_zerod = ops.where(y_true == self.nan_value, 0, y_true)
            y_pred_zerod = ops.where(y_true == self.nan_value, 0, y_pred)
            masked_weights = ops.where(y_true == self.nan_value, 0, 1)
        else:
            y_true_zerod = ops.where(y_true != y_true, 0, y_true)
            y_pred_zerod = ops.where(y_true != y_true, 0, y_pred)
            masked_weights = ops.where(y_true != y_true, 0, 1)

        if self.mask_with_weights:
            extra_args["sample_weight"] = masked_weights

        return self.loss_to_use(y_true_zerod, y_pred_zerod, **extra_args)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "nan_value": self.nan_value,
        }
