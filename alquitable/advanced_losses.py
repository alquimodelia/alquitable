import keras_core
from keras_core import ops


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
