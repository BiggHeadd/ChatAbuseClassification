# -*- coding: utf-8 -*-
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model
from keras.optimizers import rmsprop
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

def lstm_model(index, x_train, y_train, x_test, y_test, layer_num, cell_num, hidden_num, nb_epoch, dropout, lr, batch_size, decay=1e-6): 
    max_length = 25
    len_vec = 32
    max_len = 25

    save_index_path = "save_model/"+str(int(index)) + "/"
    print(save_index_path)
    if not os.path.exists(save_index_path):
        os.mkdir(save_index_path)

    weights_filepath = save_index_path + "/" + str(index) + 'epoch_{epoch:02d}-val_acc_{val_acc:.4f}.hdf5'

#________________________________________________________________________
#load_data_start
    x_train = sequence.pad_sequences(x_train, maxlen=max_len, dtype='float64')
    x_test = sequence.pad_sequences(x_test, maxlen=max_len, dtype='float64')
    x_train.reshape((len(x_train), max_len, len_vec))
    x_test.reshape((len(x_test), max_len, len_vec))

    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)

#load_data_end
#_____________________________________________________________________


    print("build..........")

    inputs = Input(shape=(max_length, len_vec), name="MyLstmInput")

    lstm_out = LSTM(cell_num, return_sequences=False)(inputs)
    tanh_out = Dense(hidden_num, activation='tanh')(lstm_out)
    dropout = Dropout(dropout)(tanh_out)
    output = Dense(2, activation='softmax', name="MyLstmOutput")(dropout)

    model = Model(inputs=inputs, outputs=output)
    _rmsprop = rmsprop(lr=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=_rmsprop, metrics=['accuracy'])

    print("<<<<<<<<<<training>>>>>>>>>")
    model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test), nb_epoch=nb_epoch, shuffle=True,
            callbacks=[ModelCheckpoint(weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
                EarlyStopping(monitor='val_acc', verbose=1, patience=30, mode='max')])
    del model

if __name__ == "__main__":
    lstm_model()
