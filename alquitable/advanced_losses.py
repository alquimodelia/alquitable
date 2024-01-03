import keras_core
from keras_core import Loss, ops


def mirror_weights(ratio=None, loss_to_use=None, weight_on_surplus=True):
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
    if loss_to_use is None:
        loss_to_use = keras_core.losses.MeanSquaredError()

    # bigger than 1 ration will give more weight pred values lower than true
    @keras_core.saving.register_keras_serializable()
    def loss(y_true, y_pred):
        diff = y_pred - y_true

        greater = ops.greater(diff, 0)
        # 0 for lower, 1 for greater
        greater = ops.cast(greater, keras_core.backend.floatx())
        # 1 for lower, 2 for greater
        greater = greater + 1

        # Now is at 1:2 ratio
        surplus_values = 1 if ratio is None else ratio
        missing_values = 1 if ratio is None else ratio

        if ratio is None:
            if weight_on_surplus:
                surplus_values = ops.maximum(0.0, -diff)
            else:
                missing_values = ops.maximum(0.0, diff)

        weights = ops.where(greater == 1, surplus_values, missing_values)

        return loss_to_use(y_true, y_pred, sample_weight=weights)

    return loss



class MirrorWeights(Loss):
    def __init__(self, ratio=None, loss_to_use=None, weight_on_surplus=True, name="mirror_weights",**kwargs):
        if loss_to_use is None:
            loss_to_use = keras_core.losses.MeanSquaredError()
        self.loss_to_use=loss_to_use
        self.ratio = ratio
        self.weight_on_surplus = weight_on_surplus
        name = f"{name}_{loss_to_use.name}"
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        diff = y_pred - y_true
        greater = ops.greater(diff, 0)
        greater = ops.cast(greater, keras_core.backend.floatx())
        greater = greater + 1

        surplus_values = 1 if self.ratio is None else self.ratio
        missing_values = 1 if self.ratio is None else self.ratio

        if self.ratio is None:
            if self.weight_on_surplus:
                surplus_values = ops.maximum(0.0, -diff)
            else:
                missing_values = ops.maximum(0.0, diff)

        weights = ops.where(greater == 1, surplus_values, missing_values)

        return self.loss_to_use(y_true, y_pred, sample_weight=weights)


class MirrorLoss(Loss):
    # Cuts of the diference in sampling number around the surplus and missing error
    def __init__(self, ratio=None, loss_to_use=None, weight_on_surplus=True, name="mirror_loss",**kwargs):
        if loss_to_use is None:
            loss_to_use = keras_core.losses.MeanSquaredError()
        self.loss_to_use=loss_to_use
        self.ratio = ratio
        self.weight_on_surplus = weight_on_surplus
        name = f"{name}_{loss_to_use.name}"
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        erro = y_pred - y_true
        m_mask = erro<0
        s_mask = erro>=0

        true_m = y_true[m_mask]
        true_s = y_true[s_mask]
        pred_m = y_pred[m_mask]
        pred_s = y_pred[s_mask]


        m_loss = self.loss_to_use(true_m, pred_m)
        s_loss = self.loss_to_use(true_s, pred_s)


        return ops.mean([m_loss, s_loss])



class MirrorPercentage(Loss):
    # Only works properly for losses where the return is the same dimension as predict: rmse, mae
    def __init__(self, ratio=None, loss_to_use=None, weight_on_surplus=True, name="mirror_percentage",
    funtion_to_dimention=None,
    
    **kwargs):
        if loss_to_use is None:
            loss_to_use = keras_core.losses.MeanSquaredError()
        self.loss_to_use=loss_to_use
        self.ratio = ratio
        self.weight_on_surplus = weight_on_surplus
        self.funtion_to_dimention = funtion_to_dimention

        name = f"{name}_{loss_to_use.name}"
        if funtion_to_dimention is None:
            if isinstance(self.loss_to_use, keras_core.losses.MeanSquaredError):
                self.funtion_to_dimention = ops.sqrt
        super().__init__(name=name, **kwargs)

    def _calculate_loss(self, true_values, pred_values, prefix, overal_avg_true=None):
        loss = self.loss_to_use(true_values, pred_values)

        # if overal_avg_pred:
        #     # Compute the averages of the predicted values
        #     avg_pred_values=overal_avg_pred
        # else:

        if self.funtion_to_dimention:
            loss = self.funtion_to_dimention(loss)

        epsilon = 1e-8

        # Normalize the losses
        norm_loss = ops.abs((loss + epsilon) / (overal_avg_true + epsilon))*100

        return norm_loss

    def call(self, y_true, y_pred):
        erro = y_pred - y_true
        # overal_avg_pred = ops.mean(y_pred)
        overal_avg_true = ops.mean(y_true)

        m_mask = erro<0
        s_mask = erro>=0
        # z_mask = erro == 0


        true_m = y_true[m_mask]
        true_s = y_true[s_mask]
        pred_m = y_pred[m_mask]
        pred_s = y_pred[s_mask]
        
        # true_z = y_true[z_mask] # True values for erro == 0
        # pred_z = y_pred[z_mask] # Predicted values for erro == 0
        
        # Calculate the loss for m_mask, s_mask, and z_mask
        norm_m_loss = self._calculate_loss(true_m, pred_m, "m",overal_avg_true=overal_avg_true) if ops.size(pred_m) else ops.convert_to_tensor(0)
        norm_s_loss = self._calculate_loss(true_s, pred_s, "s",overal_avg_true=overal_avg_true) if ops.size(pred_s) else ops.convert_to_tensor(0)
        # norm_z_loss = self._calculate_loss(true_z, pred_z, 'z', overal_avg_pred=overal_avg_pred) if ops.size(pred_z) else ops.convert_to_tensor(0) # New loss calculation for erro == 0

        return ops.mean([norm_m_loss, norm_s_loss])




class MirrorNormalized(Loss):
    # Only works properly for losses where the return is the same dimension as predict: rmse, mae
    def __init__(self, ratio=None, loss_to_use=None, weight_on_surplus=True, name="mirror_normalized",
    funtion_to_dimention=None,
    
    **kwargs):
        if loss_to_use is None:
            loss_to_use = keras_core.losses.MeanSquaredError()
        self.loss_to_use=loss_to_use
        self.ratio = ratio
        self.weight_on_surplus = weight_on_surplus
        self.funtion_to_dimention = funtion_to_dimention

        name = f"{name}_{loss_to_use.name}"
        if funtion_to_dimention is None:
            if isinstance(self.loss_to_use, keras_core.losses.MeanSquaredError):
                self.funtion_to_dimention = ops.sqrt
        super().__init__(name=name, **kwargs)

 

    def call(self, y_true, y_pred):
        erro = y_pred - y_true
        # overal_avg_pred = ops.mean(y_pred)
        ops.mean(y_true)

        m_mask = erro<0
        s_mask = erro>=0
        # z_mask = erro == 0


        true_m = y_true[m_mask]
        true_s = y_true[s_mask]
        pred_m = y_pred[m_mask]
        pred_s = y_pred[s_mask]
        
        # Compute the means and variances of true and predicted values for both positive and negative errors
        m_mean, m_var = ops.mean(ops.concatenate([true_m,pred_m])), ops.var(ops.concatenate([true_m,pred_m]))
        s_mean, s_var = ops.mean(ops.concatenate([true_s,pred_s])), ops.var(ops.concatenate([true_s,pred_s]))
        # Add a small constant to avoid division by zero

        m_var += 1e-10
        s_var += 1e-10

        # Normalize the true and predicted values
        true_m_normalized = (((true_m - m_mean) / ops.sqrt(m_var))+1)*100
        true_s_normalized = (((true_s - s_mean) / ops.sqrt(s_var))+1)*100
        pred_m_normalized = (((pred_m - m_mean) / ops.sqrt(m_var))+1)*100
        pred_s_normalized = (((pred_s - s_mean) / ops.sqrt(s_var))+1)*100

        # Compute the loss on the normalized values
        m_loss = self.loss_to_use(true_m_normalized, pred_m_normalized)
        s_loss = self.loss_to_use(true_s_normalized, pred_s_normalized)



        return ops.mean([m_loss, s_loss])
