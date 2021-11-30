"""
Here, we wrap the original environment to make it easier
to use. When a game is finished, instead of mannualy reseting
the environment, we do it automatically.
"""
import numpy as np
import tensorflow as tf

def _format_observation(obs, device):
    """
    A utility function to process observations and
    move them to CUDA.
    """
    position = obs['position']
    if not device == "cpu":
        device = 'cuda:' + str(device)
    x_batch = tf.convert_to_tensor(obs['x_batch'])
    z_batch = tf.convert_to_tensor(obs['z_batch'])
    x_no_action = tf.convert_to_tensor(obs['x_no_action'])
    z = tf.convert_to_tensor(obs['z'])
    obs = {'x_batch': x_batch,
           'z_batch': z_batch,
           'legal_actions': obs['legal_actions'],
           }
    return position, obs, x_no_action, z

class Environment:
    def __init__(self, env, device):
        """ Initialzie this environment wrapper
        """
        self.env = env
        self.device = device
        self.episode_return = None

    def initial(self):
        initial_position, initial_obs, x_no_action, z = _format_observation(self.env.reset(), self.device)
        initial_reward = tf.zeros(1)
        self.episode_return = tf.zeros(1)
        initial_done = tf.ones(1, dtype=tf.bool)

        return initial_position, initial_obs, dict(
            done=initial_done,
            episode_return=self.episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
            )
        
    def step(self, action):
        obs, reward, done, _ = self.env.step(action)

        self.episode_return += reward
        episode_return = self.episode_return 

        if done:
            obs = self.env.reset()
            self.episode_return = tf.zeros(1)

        position, obs, x_no_action, z = _format_observation(obs, self.device)
        reward = tf.reshape(tf.convert_to_tensor(reward), 1)
        done = tf.reshape(tf.convert_to_tensor(done), 1)
        
        return position, obs, dict(
            done=done,
            episode_return=episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
            )

    def close(self):
        self.env.close()
