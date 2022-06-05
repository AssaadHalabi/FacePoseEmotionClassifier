import numpy as np
import telegram_send
from random import randint

from tensorflow.keras.callbacks import Callback

class TelegramNotifier(Callback):
    def __init__(self, time_stamp, run_id):
        self.unique_run_id = randint(1000000, 10000000)
        self.time_stamp = time_stamp
        self.run_id = run_id

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        message = f"({self.unique_run_id}) - {self.time_stamp} - RUN ID: {self.run_id} - EPOCH {epoch} COMPLETE.\n"
        metrics_str = "\n".join([f"{key}: {value:.3f}" for key, value in logs.items()])
        message += metrics_str
        try:
            telegram_send.send(messages=[message])
        except:
            print("Could not send telegram message.")

class F1History(Callback):
    def __init__(self, train, validation=None):
        super(F1History, self).__init__()
        self.validation = validation
        self.train = train

    def on_epoch_end(self, epoch, logs={}):

        logs['F1_score_train'] = float('-inf')
        X_train, y_train = self.train[0], self.train[1]
        y_pred = (self.model.predict(X_train).ravel()>0.5)+0
        score = f1_score(y_train, y_pred)       

        if (self.validation):
            logs['F1_score_val'] = float('-inf')
            X_valid, y_valid = self.validation[0], self.validation[1]
            y_val_pred = (self.model.predict(X_valid).ravel()>0.5)+0
            val_score = f1_score(y_valid, y_val_pred)
            logs['F1_score_train'] = np.round(score, 5)
            logs['F1_score_val'] = np.round(val_score, 5)
        else:
            logs['F1_score_train'] = np.round(score, 5)