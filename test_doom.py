import tensorflow as tf
import sys
import gym
import vizdoomgym
from doom_final import downsample_image, update_state
from parameters import IMG_SIZE
import numpy as np

NUM_EPISODES = 10
RENDER = True

def load(model_meta, model_path, sess):
    saver = tf.train.import_meta_graph(model_meta)
    saver.restore(sess, tf.train.latest_checkpoint(model_path))

    print("DQN Model restored.")

    predict_op = tf.get_default_graph().get_tensor_by_name("model/predict_layer/fully_connected/Relu:0")
    x = tf.get_default_graph().get_tensor_by_name("model/X:0")

    return x, predict_op

def main():
    assert len(sys.argv) == 5, "Did not provide 4 arguments!"
    model_meta = sys.argv[1]
    model_path = sys.argv[2]
    game_name = sys.argv[3]
    NUM_EPISODES = int(sys.argv[4])
    global RENDER

    sess = tf.Session()
    ip, predict_op = load(model_meta, model_path, sess)

    # Setting up gym env
    env = gym.make(game_name)
    try:
        if RENDER:
            env.render()
    except:
        RENDER = False
    episode_num = 1
    episode_rews = []
    while episode_num <= NUM_EPISODES:
        obs = env.reset()
        obs_small = downsample_image(obs, IMG_SIZE, down_only = True)
        state = np.stack([obs_small]*4, axis=2)
        episode_reward = 0
        while True:
            # Get action:
            predictions = sess.run(predict_op, feed_dict={ip: [state]})
            action = np.argmax(predictions[0])
            
            # Play!
            obs, reward, done, info = env.step(action)
            if RENDER:
                env.render()
            episode_reward += reward
            
            obs_small = downsample_image(obs, IMG_SIZE, down_only= True)
            update_state(state, obs_small)
            if done:
                episode_rews.append(episode_reward)
                print("Episode {} finished.".format(episode_num))
                print("Episode reward: {}".format(episode_reward))
                episode_num += 1
                break
    avg = sum(episode_rews)*1.0/len(episode_rews)
    print("Average Reward: {}".format(avg))

if __name__ == "__main__":
    main()
