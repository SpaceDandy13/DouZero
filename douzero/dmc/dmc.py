import os
import threading
import time
import timeit
import pprint
from collections import deque
import numpy as np

import queue as mp

import tensorflow as tf

from .file_writer import FileWriter
from .models import Model
from .utils import get_batch, log, create_env, create_buffers, create_optimizers, act, act1

mean_episode_return_buf = {p:deque(maxlen=100) for p in ['landlord', 'landlord_up', 'landlord_down']}

def compute_loss(logits, targets):
    loss = tf.math.reduce_mean((tf.experimental.numpy.squeeze(logits, axis=-1) - targets)**2)
    return loss

def learn(position,
          model,
          batch,
          optimizer,
          flags):
    """Performs a learning (optimization) step."""
    
    obs_x_no_action = batch['obs_x_no_action']
    obs_action = batch['obs_action']
    obs_x = tf.cast(tf.concat((obs_x_no_action, obs_action), axis=2),dtype=tf.float32)
    shape_obs_x = obs_x.get_shape().as_list()
    shape_obs_x[1] = shape_obs_x[0]*shape_obs_x[1]
    shape_obs_x.pop(0)
    obs_x = tf.reshape(obs_x, shape_obs_x)

    obs_z = tf.cast(batch['obs_z'],dtype=tf.float32)
    shape_obs_z = obs_z.get_shape().as_list()
    shape_obs_z[1] = shape_obs_z[0]*shape_obs_z[1]
    shape_obs_z.pop(0)
    obs_z = tf.reshape(obs_z, shape_obs_z)

    shape_target = batch['target'].get_shape().as_list()
    shape_target[1] = shape_target[0]*shape_target[1]
    shape_target.pop(0)
    target = tf.reshape(batch['target'], shape_target)
    target = tf.cast(target,dtype=tf.float32)

    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf[position].append(tf.reduce_mean(episode_returns))
        

    with tf.GradientTape() as tape:
        learner_outputs = model(obs_z, obs_x, True)
        loss = compute_loss(learner_outputs['values'], target)
    stats = {
        'mean_episode_return_'+position: tf.reduce_mean(tf.stack([_r for _r in mean_episode_return_buf[position]])),
        'loss_'+position: loss,
    }
        # todo optimizer.zero_grad()
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients = [tf.clip_by_norm(g, flags.max_grad_norm) for g in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


    # for actor_model in actor_models.values():
    #     actor_model.get_model(position).load_state_dict(model.state_dict())
    return stats

def train(flags):  
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with  multiple threads.
    """

    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    
    # Initialize actor models
    # models = {}
    device = 'cpu'
    # for device in device_iterator:
    #     model = Model(device=device)
    #     model.share_memory()
    #     model.eval()
    #     models[device] = model
    model = Model(device='cpu')

    # Initialize buffers
    buffers = create_buffers(flags, [device])
   
    # Initialize queues
    actor_processes = []
    # ctx = mp.get_context('spawn')
    free_queue = {}
    full_queue = {}
        
    
    _free_queue = {'landlord': mp.SimpleQueue(), 'landlord_up': mp.SimpleQueue(), 'landlord_down': mp.SimpleQueue()}
    _full_queue = {'landlord': mp.SimpleQueue(), 'landlord_up': mp.SimpleQueue(), 'landlord_down': mp.SimpleQueue()}
    free_queue[device] = _free_queue
    full_queue[device] = _full_queue

    # Learner model for training
    # learner_model = Model(device=flags.training_device)

    # Create optimizers
    optimizers = create_optimizers(flags, model)

    # Stat Keys
    stat_keys = [
        'mean_episode_return_landlord',
        'loss_landlord',
        'mean_episode_return_landlord_up',
        'loss_landlord_up',
        'mean_episode_return_landlord_down',
        'loss_landlord_down',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'landlord':0, 'landlord_up':0, 'landlord_down':0}

    # Load models if any ===========
    



    for m in range(flags.num_buffers):
        free_queue[device]['landlord'].put(m)
        free_queue[device]['landlord_up'].put(m)
        free_queue[device]['landlord_down'].put(m)

    
    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        _models = model.get_models()

        # Save the weights for evaluation purpose
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.savedir, flags.xpid, position+'_weights_'+str(frames)+'.ckpt')))
            model.get_model(position).save_weights(model_weights_dir)




    fps_log = []
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames < flags.total_frames:
            free_queue[device], full_queue[device],buffers[device] = act1(0, device, free_queue[device], full_queue[device], model, buffers[device], flags)

            train_num = 0
            for position in ['landlord', 'landlord_up', 'landlord_down']:
                train_num += 1
                indices = [full_queue[device][position].get() for _ in range(flags.batch_size)]
                batch = {
                    key: tf.stack([buffers[device][position][key][m] for m in indices], axis=1)
                    for key in buffers[device][position]
                }
                for m in indices:
                    free_queue[device][position].put(m)
                _stats = learn(position, model.get_model(position), batch, 
                    optimizers[position], flags)

                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
                position_frames[position] += T * B

            if train_num == 0:
                continue
            start_frames = frames
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            # time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:  
                checkpoint(frames)
                last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            position_fps = {k:(position_frames[k]-position_start_frames[k])/(end_time-start_time) for k in position_frames}
            log.info('After %i (L:%i U:%i D:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f U:%.1f D:%.1f) Stats:\n%s',
                     frames,
                     position_frames['landlord'],
                     position_frames['landlord_up'],
                     position_frames['landlord_down'],
                     fps,
                     fps_avg,
                     position_fps['landlord'],
                     position_fps['landlord_up'],
                     position_fps['landlord_down'],
                     pprint.pformat(stats))

    except KeyboardInterrupt:
        return 
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    checkpoint(frames)
    plogger.close()
