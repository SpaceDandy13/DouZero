"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM


class LandlordLstmModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm = LSTM(128,return_sequences=True, return_state=True) # 162, 128
        self.dense1 = Dense(512)    #373 + 128,
        self.dense2 = Dense(512)
        self.dense3 = Dense(512)
        self.dense4 = Dense(512)
        self.dense5 = Dense(512)
        self.dense6 = Dense(1)

    def call(self, z, x, return_value=False, flags=None):
        lstm_out, h_n, _ = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = tf.concat([lstm_out,x], axis=-1)
        x = self.dense1(x)
        x = tf.nn.relu(x)
        x = self.dense2(x)
        x = tf.nn.relu(x)
        x = self.dense3(x)
        x = tf.nn.relu(x)
        x = self.dense4(x)
        x = tf.nn.relu(x)
        x = self.dense5(x)
        x = tf.nn.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = tf.random.uniform(shape = [1,], minval=0, maxval=x.shape[0], dtype=tf.int32)
            else:
                action = tf.math.argmax(x,axis=0)[0]
            return dict(action=action)

class FarmerLstmModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm = LSTM(128,return_sequences=True, return_state=True) # 162, 128
        self.dense1 = Dense(512)    #484 + 128, 
        self.dense2 = Dense(512)
        self.dense3 = Dense(512)
        self.dense4 = Dense(512)
        self.dense5 = Dense(512)
        self.dense6 = Dense(1)

    def call(self, z, x, return_value=False, flags=None):
        lstm_out, h_n, _ = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = tf.concat([lstm_out,x], axis=-1)
        x = self.dense1(x)
        x = tf.nn.relu(x)
        x = self.dense2(x)
        x = tf.nn.relu(x)
        x = self.dense3(x)
        x = tf.nn.relu(x)
        x = self.dense4(x)
        x = tf.nn.relu(x)
        x = self.dense5(x)
        x = tf.nn.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = tf.random.uniform(shape = [1,], minval=0, maxval=x.shape[0], dtype=tf.int32)
            else:
                action = tf.math.argmax(x,axis=0)[0]
            return dict(action=action)

# Model dict is only used in evaluation but not training
model_dict = {}
model_dict['landlord'] = LandlordLstmModel
model_dict['landlord_up'] = FarmerLstmModel
model_dict['landlord_down'] = FarmerLstmModel

class Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.models['landlord'] = LandlordLstmModel()
        self.models['landlord_up'] = FarmerLstmModel()
        self.models['landlord_down'] = FarmerLstmModel()

    def call(self, position, z, x, training=False, flags=None):
        model = self.models[position]
        return model.call(z, x, training, flags)

    def share_memory(self):
        self.models['landlord']#.share_memory()
        self.models['landlord_up']#.share_memory()
        self.models['landlord_down']#.share_memory()

    def eval(self):
        self.models['landlord']#.eval()
        self.models['landlord_up']#.eval()
        self.models['landlord_down']#.eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models
