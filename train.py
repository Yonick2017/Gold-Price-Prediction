import time
import datetime
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from model import *

def build_model():
    inp = Input(shape = (SEQ_LEN, 1))

    # LSTM before attention layers
    x = Bidirectional(LSTM(128, return_sequences=True))(inp)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    x, slf_attn = MultiHeadAttention(n_head=3, d_model=300, d_k=64, d_v=64, dropout=0.1)(x, x, x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    x = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs = inp, outputs = x)
    model.compile(
        loss = "mean_squared_error",
        #optimizer = Adam(lr = config["lr"], decay = config["lr_d"]),
        optimizer = "adam")

    # Save entire model to a HDF5 file
    model.save('GE_stock_price_model.h5')

    return model

multi_head = build_model()

start = time.time()

multi_head.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_valid, y_valid),
                    #callbacks = [checkpoint , lr_reduce]
             )

end = time.time()

elapsed = end - start
print(datetime.timedelta(seconds=elapsed))
