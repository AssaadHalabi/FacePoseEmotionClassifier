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
