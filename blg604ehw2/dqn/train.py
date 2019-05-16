""" Training and testion functions """

import gym
import numpy as np
from collections import namedtuple

from blg604ehw2.dqn.replaybuffer import Transition


ArgsDQN = namedtuple("ArgsDQN", """
                                env_name
                                nact
                                buffersize
                                max_epsilon
                                min_epsilon
                                target_update_period
                                gamma
                                lr
                                device
                                batch_size
                                episode
                                max_eps_len
                                """
                     )

ArgsDDPQN = namedtuple("ArgsDDPQN", """
                                    env_name
                                    nact
                                    buffersize
                                    max_epsilon
                                    min_epsilon
                                    target_replace_period
                                    gamma
                                    lr
                                    device
                                    batch_size
                                    episode
                                    max_eps_len
                                    """
                       )


def episodic_train(env, agent, args, epsilon):
    """ Train the agent in the env one episode.
        Return time steps passed and mean td_error
    """
    ### YOUR CODE HERE ###

    agent.train()
    state = env.reset()
    running_td_error = []
    time_step = 0

    while True:
        action = agent.e_greedy_policy(state, epsilon(time_step))
        next_state, reward, done, _ = env.step(action)

        agent.push_transition(Transition(state, action, reward, next_state, done), args.gamma)
        state = next_state
        time_step += 1

        if agent.buffer_size > args.batch_size:
            running_td_error.append(agent.update(args.batch_size, args.gamma))

        if done or time_step % args.max_eps_len == 0:
            break

    td_error = np.mean(running_td_error)

    ###       END      ###
    return time_step, td_error


def episodic_test(env, agent, args, render=False, monitor_path=None):
    """ Evaluate the agent and return episodic reward.

        Parameters:
            - env: Environment to evaluate
            - agent: Agent model
            - args: Hyperparamters of the model
            - render: Render the environment if True
            (default=False)
            - monitor_path: Render and save the mp4 file
            to the give path if any given (default=None)
    """

    agent.eval()
    if monitor_path:
        path = "monitor/" + monitor_path
        env = gym.wrappers.Monitor(
            env, path, video_callable=lambda eps_id: True, force=True)
        render = False

    eps_reward = 0
    state = env.reset()
    for time_step in range(args.max_eps_len):
        action = agent.greedy_policy(state)
        if render:
            env.render()
        state, reward, done, info = env.step(action)
        eps_reward += reward
        if done:
            break
    if monitor_path:
        env.close()

    return eps_reward
    