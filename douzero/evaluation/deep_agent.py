import tensorflow as tf
import numpy as np

from douzero.env.env import get_obs

def _load_model(position, model_path):
    from douzero.dmc.models import Model
    model = Model(device='cpu')
    model_pos = model.get_model(position)
    
    model_pos.call(np.zeros([1,5,162]),np.zeros([1,373]))
    model_pos.load_weights(model_path)

    return model_pos

class DeepAgent:

    def __init__(self, position, model_path):
        self.model = _load_model(position, model_path)

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        obs = get_obs(infoset) 

        z_batch = tf.cast(obs['z_batch'], dtype=tf.float32)
        x_batch = tf.cast(obs['x_batch'], dtype=tf.float32)
        y_pred = self.model.call(z_batch, x_batch, return_value=True)['values']
        y_pred = y_pred.numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]

        return best_action
