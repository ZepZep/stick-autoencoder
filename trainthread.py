import multiprocessing
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt

from stickgenerator import get_stick

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

class TrainThread(multiprocessing.Process):
    def __init__(self, in_queue, out_queue):
        super(TrainThread, self).__init__()
        self.in_queue: multiprocessing.Queue = in_queue
        self.out_queue: multiprocessing.Queue = out_queue
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        
        self.session_set = False
        
        self.N_STICKS = 10000

    def run(self):
        self._init_model()
        print("Creating data ... ", end="", flush=True)
        self._init_data()
        print("DONE")
        
        while True:
            cmd, data = self.in_queue.get()
            if cmd == "train":
                rep, ep = (int(x) for x in data)
                self._train(rep, ep)
            elif cmd == "quit":
                break


    def _init_model(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
        set_session(tf.Session(config=config))
        input_img = Input(shape=(128, 128, 1))

        x = Conv2D(32, (5, 5), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((4, 4), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((4, 4), padding='same')(x)
        x = Conv2D(2, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Flatten()(x)
        
        encoded = Dense(16, activation='softmax')(x)
        print("shape of encoded", K.int_shape(encoded))
        
        x = Reshape((4, 4, 1))(encoded)
        x = Conv2D(2, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((4, 4))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x) 
        x = UpSampling2D((4, 4))(x)
        decoded = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)

        print("shape of decoded", K.int_shape(decoded))

        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        
        self.encoder = Model(input_img, encoded)
        self.encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        
        #self.decoder = Model(encoded, decoded)
        #self.decoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    def _init_data(self):
        stick_list = np.zeros((self.N_STICKS, 128, 128, 1))
        corp_pos_list = 2*np.random.random((self.N_STICKS, 5)) - 1

        for i, corp_pos in enumerate(corp_pos_list):
            stick_list[i] = np.array(get_stick(corp_pos), dtype=np.float32).reshape((128, 128, 1))

        self.x_train = stick_list

    def _train(self, rep, ep):
        if not self.session_set:
            self.session_set = True

        
        for i in range(rep):
            self.autoencoder.fit(self.x_train, self.x_train, epochs=ep, batch_size=128,
                                 validation_split=0.1, verbose=1)
            
            self.autoencoder.save("nns/ac")
            self.encoder.save("nns/enc")
            #self.decoder.save("nns/dec")
            
            self.out_queue.put("newnn")


