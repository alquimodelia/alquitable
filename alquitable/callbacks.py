import json
import math
import os

from keras_core.callbacks import Callback


def update_history_dict(new_log, old_log):
    """
    Update the history dictionary with new log entries.

    This function merges the old and new log dictionaries, ensuring that
    all entries are combined correctly.

    Parameters
    ----------
    new_log : dict
        The new log dictionary to be added.
    old_log : dict, optional
        The old log dictionary to be updated. If not provided, an empty
        dictionary will be used.

    Returns
    -------
    dict
        The updated log dictionary.
    """
    history_to_save = {}
    if old_log:
        if isinstance(old_log, dict):
            for key in old_log:
                old = old_log[key]
                new = new_log.get(key, [])
                if not isinstance(new, list):
                    new = [new]
                if not isinstance(old, list):
                    old = [old]
                old = [f for f in old if f]
                old = [f for f in old if not math.isnan(f)]
                history_to_save[key] = old + new
    for key in new_log:
        if key not in history_to_save:
            history_to_save[key] = new_log[key]
    return history_to_save


class SaveModelCallback(Callback):
    """
    A Keras callback that saves the model at specified frequency.

    This callback saves the model at the end of every specified epoch.
    It also saves the training logs to a JSON file.
    """

    def __init__(
        self,
        save_frequency,
        model_keras_filename,
        model_log_filename,
        logs=None,
        start_epoch=0,
    ):
        """
        Initialize the SaveModelCallback.

        Parameters
        ----------
        save_frequency : int
            The frequency of saving the model, in terms of epochs.
        model_keras_filename : str
            The filename of the saved model.
        model_log_filename : str
            The filename of the saved log.
        logs : dict, optional
            The training logs. If not provided, an empty dictionary will be used.
        start_epoch : int, optional
            The start epoch. Default is 0.
        """
        super(SaveModelCallback, self).__init__()
        self.save_frequency = save_frequency
        self.model_keras_filename = model_keras_filename
        self.model_log_filename = model_log_filename
        self.start_epoch = start_epoch
        self.logs = logs or {}

    def on_epoch_end(self, epoch, logs=None):
        """
        Save the model and log at the end of an epoch.

        Parameters
        ----------
        epoch : int
            The current epoch.
        logs : dict, optional
            The training logs. If not provided, an empty dictionary will be used.
        """
        epoc_save = epoch + 1 + self.start_epoch
        self.logs = update_history_dict(logs, self.logs)
        if (epoc_save) % self.save_frequency == 0:
            model_save_name = self.model_keras_filename.format(epoch=epoc_save)
            # Save the model
            self.model.save(model_save_name)

            # Save the logs to a JSON file
            with open(self.model_log_filename, "w") as f:
                json.dump(self.logs, f)


class StopOnNanLoss(Callback):
    """
    A Keras callback that stops training when loss is NaN or Infinity.

    This callback checks the loss value at the end of each epoch and the end
    of each batch. If the loss is NaN or Infinity, it saves the model weights
    from the last epoch where the loss was not NaN or Infinity, stops the
    training, and saves the training logs to a JSON file.
    """

    def __init__(
        self,
        filepath,
        model_log_filename,
        logs=None,
        save_frequency=1,
        model_keras_filename=None,
        start_epoch=0,
    ):
        """
        Initialize the StopOnNanLoss callback.

        Parameters
        ----------
        filepath : str
            The filepath where the model will be saved.
        model_log_filename : str
            The filename of the saved log.
        logs : dict, optional
            The training logs. If not provided, an empty dictionary will be used.
        save_frequency : int, optional
            The frequency of saving the model, in terms of epochs. Default is 1.
        model_keras_filename : str, optional
            The filename of the saved model. If not provided, the filepath will be used.
        start_epoch : int, optional
            The start epoch. Default is 0.
        """
        super(StopOnNanLoss, self).__init__()
        self.filepath = filepath
        self.last_good_model = None
        self.last_good_epoch = None
        self.logs = logs or {}
        self.model_log_filename = model_log_filename
        self.model_keras_filename = model_keras_filename or filepath
        self.start_epoch = start_epoch
        self.save_frequency = save_frequency

    def on_epoch_end(self, epoch, logs=None):
        """
        Check the loss at the end of an epoch and save the model if necessary.

        Parameters
        ----------
        epoch : int
            The current epoch.
        logs : dict, optional
            The training logs. If not provided, an empty dictionary will be used.
        """
        loss = logs.get("loss")
        nan_value = math.isnan(loss) or math.isinf(loss)
        if loss is not None and isinstance(loss, float) and not nan_value:
            self.last_good_model = self.model.get_weights()
            self.last_good_epoch = epoch
            self.logs = update_history_dict(logs, self.logs)
            epoc_save = epoch + 1 + self.start_epoch
            if (epoc_save) % self.save_frequency == 0:
                model_save_name = self.model_keras_filename.format(
                    epoch=epoc_save
                )
                # Save the model
                self.model.save(model_save_name)

                # Save the logs to a JSON file
                with open(self.model_log_filename, "w") as f:
                    json.dump(self.logs, f)

    def on_train_batch_end(self, batch, logs=None):
        """
        Check the loss at the end of a batch and save the model if necessary.

        Parameters
        ----------
        batch : int
            The current batch.
        logs : dict, optional
            The training logs. If not provided, an empty dictionary will be used.
        """
        loss = logs.get("loss")
        nan_value = math.isnan(loss) or math.isinf(loss)
        if loss is not None and isinstance(loss, float) and nan_value:
            print(f"Stopping training due to NaN loss at batch {batch}.")
            if self.last_good_model is not None:
                self.model.set_weights(self.last_good_model)
            if self.last_good_epoch is not None:
                frq_model_filename = self.filepath.replace(
                    ".keras", f"freq_saves/{self.last_good_epoch}.keras"
                )
                os.makedirs(os.path.dirname(frq_model_filename), exist_ok=True)
                self.model.save(frq_model_filename)
                with open(self.model_log_filename, "w") as f:
                    json.dump(self.logs, f)
            self.model.save(
                self.filepath.replace(".keras", "freq_saves/unfinished.keras")
            )
            self.model.stop_training = True
        else:
            self.last_good_model = self.model.get_weights()
            unfinished = self.filepath.replace(
                ".keras", "freq_saves/unfinished.keras"
            )
            if os.path.exists(unfinished):
                os.remove(unfinished)
