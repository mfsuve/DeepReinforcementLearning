""" Worker functions for training and testing """

import torch
import gym
import numpy as np
from collections import namedtuple
import torch.multiprocessing as mp

from blg604ehw2.utils import LoadingBar, normalize
from blg604ehw2.atari_wrapper import LazyFrames

# Hyperparamteres of A3C
A3C_args = namedtuple("A3C_args",
                      """
                        maxtimestep
                        maxlen
                        nstep
                        gamma 
                        lr 
                        beta 
                        device
                      """)


def train_worker(args, globalmodel, optim, envfunc, agentfunc, lock, logger):
    """ Training worker function.
        Train until the maximum time step is reached.
        Arguments:
            - args: Hyperparameters
            - globalmodel: Global(shared) agent for
            synchronization.
            - optim: Shared optimizer
            - envfunc: Environment generating function
            - agentfunc: Agent generating function
            - lock: Lock for shared memory
            - logger: Namedtuple of shared objects for
            logging purposes
    """
    env = envfunc()
    agent = agentfunc()
    agent.train()
    ### YOUR CODE HERE ###

    # Remember Logger has the shared time step value

    # Worker should be in a loop that terminates when
    # the shared time step value is higher than the
    # maximum time step.

    eps_time = 0
    done = True
    hx = None
    cx = None

    state = env.reset()
    if isinstance(state, LazyFrames):
        state = normalize(np.array(state, dtype='float32').transpose(2, 0, 1))
    state = torch.from_numpy(state).unsqueeze(0).to(args.device).float()

    while logger.time.value < args.maxtimestep:

        agent.synchronize(globalmodel.state_dict())

        if done:
            hx = torch.zeros(1, agent.hidden_size).to(args.device)
            cx = torch.zeros(1, agent.hidden_size).to(args.device)
        else:
            hx = hx.detach()
            cx = cx.detach()

        values, rewards, log_probs, entropies = [], [], [], []

        for step in range(args.nstep):

            action, value, log_prob, entropy, (hx, cx) = \
                agent.soft_policy((state, (hx, cx)))

            state, reward, done, _ = env.step(action.cpu().numpy()[0])
            if isinstance(state, LazyFrames):
                state = normalize(np.array(state, dtype='float32').transpose(2, 0, 1))
            state = torch.from_numpy(state).unsqueeze(0).to(args.device).float()
            reward = max(min(float(reward), 1.0), -1.0)

            eps_time += 1
            done = done or eps_time >= args.maxlen
            with lock:
                logger.time.value += 1

            values.append(value.squeeze())
            rewards.append(reward)
            log_probs.append(log_prob)
            entropies.append(entropy)

            if done:
                break

        if done:
            eps_time = 0
            state = env.reset()
            if isinstance(state, LazyFrames):
                state = normalize(np.array(state, dtype='float32').transpose(2, 0, 1))
            state = torch.from_numpy(state).unsqueeze(0).to(args.device).float()

        transitions = (values, rewards, log_probs, entropies)
        actor_loss, critic_loss = agent.loss(
            transitions, (state, (hx, cx)), done, args.gamma, args.beta)

        agent.zero_grad()
        (actor_loss + 0.5 * critic_loss).backward()
        with lock:
            agent.global_update(optim, globalmodel)


######################################################################################################
############################ ilk hali ##########################################
    # Transition = namedtuple('Transition', 'reward value entropy log_prob')
    #
    # eps_time = 0
    # state = env.reset()
    # ha = torch.zeros(1, 128).to(args.device)
    # hc = torch.zeros(1, 128).to(args.device)
    #
    # while logger.time.value < args.maxtimestep:
    #     agent.zero_grad()
    #     agent.synchronize(globalmodel.state_dict())
    #
    #     transitions = []
    #     for _ in range(args.nstep):
    #         dist, value, (ha, hc) = agent.soft_policy((state, ha, hc))
    #
    #         action = dist.sample().squeeze()
    #         action = torch.clamp(action, -.5, .5)
    #
    #         log_prob = dist.log_prob(action)
    #         entropy = dist.entropy()
    #
    #         state, reward, done, _ = env.step(action)
    #
    #         reward = torch.clamp(torch.FloatTensor([reward]), -1, 1)
    #         transitions.append(Transition(reward, value, entropy, log_prob))
    #
    #         eps_time += 1
    #         with lock:
    #             logger.time.value += 1
    #             if logger.time.value >= args.maxtimestep:
    #                 done = True
    #
    #         done = done or eps_time >= args.maxlen
    #
    #         if done:
    #             break
    #
    #     actor_loss, critic_loss = agent.loss(transitions, (state, ha, hc), done, args.gamma, args.beta)
    #
    #     # agent.zero_grad()
    #     (actor_loss + 0.5 * critic_loss).mean().backward()
    #     with lock:
    #         # globalmodel.zero_grad()
    #         agent.global_update(optim, globalmodel)
    #
    #     # Resetting for the next episode
    #     if done:
    #         state = env.reset()
    #         eps_time = 0
    #         ha = torch.zeros(1, 128).to(args.device)
    #         hc = torch.zeros(1, 128).to(args.device)
    #     else:
    #         ha = ha.detach()
    #         hc = hc.detach()

    ###       END      ###


def test_worker(args, globalmodel, envfunc, agentfunc, lock, logger,
                monitor_path=None, render=False):
    """ Evaluation worker function.
        Test the greedy agent until max time step is
        reached. After every episode, synchronize the
        agent. Loading bar is used to track learning
        process in the notebook.
        
        Arguments:
            - args: Hyperparameters
            - globalmodel: Global(shared) agent for
            synchronization.
            - envfunc: Environment generating function
            - agentfunc: Agent generating function
            - lock: Lock for shared memory
            - logger: Namedtuple of shared objects for
            logging purposes
            - monitor_path: Path for monitoring. If not
            given environment will not be monitored
            (default=None)
            - render: If true render the environment
            (default=False)
    """
    env = envfunc()
    agent = agentfunc()
    agent.eval()
    bar = LoadingBar(args.maxtimestep, "Time step")
    ### YOUR CODE HERE ###

    # Remember to call bar.process with time step and
    # best reward achived after each episode.
    # You may not use LoadingBar (optional).

    # You can include additional logging
    # Remember to change Logger namedtuple and
    # logger in order to do so.

    import time

    if monitor_path:
        path = "monitor/" + monitor_path
        env = gym.wrappers.Monitor(
            env, path, video_callable=lambda eps_id: True, force=True)
        render = False

    eps_reward = 0
    eps_time = 0
    best_reward = -np.inf
    done = True
    hx, cx = None, None
    start = time.time()

    state = env.reset()
    if isinstance(state, LazyFrames):
        state = normalize(np.array(state, dtype='float32').transpose(2, 0, 1))
    state = torch.from_numpy(state).unsqueeze(0).to(args.device).float()

    while logger.time.value < args.maxtimestep: # Terminate after max_time
        if done:
            with lock:
                agent.synchronize(globalmodel.state_dict())
            time_step = logger.time.value

        with torch.no_grad():
            if done:
                hx = torch.zeros(1, agent.hidden_size)
                cx = torch.zeros(1, agent.hidden_size)
            else:
                hx = hx.detach()
                cx = cx.detach()
            action, (hx, cx) = agent.greedy_policy((state, (hx, cx)))

        if render:
            env.render()

        state, reward, done, _ = env.step(action)
        if isinstance(state, LazyFrames):
            state = normalize(np.array(state, dtype='float32').transpose(2, 0, 1))
        state = torch.from_numpy(state).unsqueeze(0).to(args.device).float()

        eps_time += 1
        done = done or eps_time >= args.maxlen

        eps_reward += reward

        if done:
            print(f'Time {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start))}, episode reward {eps_reward}, episode length {eps_time}')
            if eps_reward > best_reward:
                best_reward = eps_reward
                logger.best_model.load_state_dict(agent.state_dict())

            logger.eps_reward.append(eps_reward)
            logger.best_reward.append(best_reward)
            logger.time_steps.append(time_step)

            eps_reward = 0
            eps_time = 0

            state = env.reset()
            if isinstance(state, LazyFrames):
                state = normalize(np.array(state, dtype='float32').transpose(2, 0, 1))
            state = torch.from_numpy(state).unsqueeze(0).to(args.device).float()

            bar.progress(time_step, best_reward)
            time.sleep(60)

    if monitor_path:
        env.close()

    bar.success(best_reward)

########################### ilk hali ###########################################

    # import time
    #
    # eps_reward = 0
    # best_reward = -np.inf
    # best_model = None
    # eps_time = 0
    # state = env.reset()
    # agent.synchronize(globalmodel.state_dict())
    # time_step = logger.time.value
    # start_time = time.time()
    # ha = torch.zeros(1, 128).to(args.device)
    # hc = torch.zeros(1, 128).to(args.device)
    # while True:
    #     with torch.no_grad():
    #         action, (ha, hc) = agent.greedy_policy((state, ha, hc))
    #
    #     if render:
    #         env.render()
    #
    #     state, reward, done, info = env.step(action)
    #
    #     reward = min(max(reward, -1), 1)
    #     eps_reward += reward
    #
    #     eps_time += 1
    #     done = done or eps_time >= args.maxlen
    #
    #     if done:
    #         print(f'Time {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))}, episode reward {eps_reward}, episode length {eps_time}')
    #         if eps_reward > best_reward:
    #             best_reward = eps_reward
    #             best_model = agent.state_dict()
    #             logger.best_model.load_state_dict(best_model)
    #
    #         logger.eps_reward.append(eps_reward)
    #         logger.best_reward.append(best_reward)
    #         logger.time_steps.append(time_step)
    #
    #         eps_time = 0
    #         eps_reward = 0
    #         state = env.reset()
    #
    #         ha = torch.zeros(1, 128).to(args.device)
    #         hc = torch.zeros(1, 128).to(args.device)
    #
    #         bar.progress(time_step, best_reward)
    #
    #         agent.synchronize(globalmodel.state_dict())
    #         time_step = logger.time.value
    #
    #         if time_step >= args.maxtimestep:
    #             break
    #
    # if monitor_path:
    #     env.close()
    #
    # bar.success(best_reward)

    ###       END      ###

























