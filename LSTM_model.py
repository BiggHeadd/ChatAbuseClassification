# -*- coding: utf-8 -*-
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model
from keras.optimizers import rmsprop
def lstm_model(): 
    max_length = 20
    len_vec = 32
    cell_num = 10
    hidden_num = 10
    dropout = 0.3

    inputs = Input(shape=(max_length, len_vec), name="MyLstmInput")

    lstm_out = LSTM(cell_num, return_sequences=False)(inputs)
    tanh_out = Dense(hidden_num, activation='tanh')(lstm_out)
    dropout = Dropout(dropout)(tanh_out)
    output = Dense(2, activation='softmax', name="MyLstmOutput")(dropout)

    model = Model(inputs=inputs, outputs=output)
    _rmsprop = rmsprop(lr=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=_rmsprop, metrics=['accuracy'])
    print(model.summary())

    del model

if __name__ == "__main__":
    lstm_model()
