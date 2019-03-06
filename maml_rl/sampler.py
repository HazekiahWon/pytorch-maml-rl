import gym
import torch
import multiprocessing as mp

from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import BatchEpisodes

def make_env(env_name):
    def _make_env():
        return gym.make(env_name)
    return _make_env

class BatchSampler(object):
    # TODO: checkout gym.env
    def __init__(self, env_name, batch_size, num_workers=mp.cpu_count() - 1):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.queue = mp.Queue()
        # sub proc for envs, reset, step,
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)],
            queue=self.queue)
        self._env = gym.make(env_name) # main env ?
# ==> env.step, reset
    def sample(self, policy, params=None, gamma=0.95, device='cpu'):
        """
        for the same mdp, sample batch_size trajectories and return
        :param policy:
        :param params:
        :param gamma:
        :param device:
        :return:
        """
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)

        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):# different instances of the same environments
            self.queue.put(None)
        # o, task_ids
        observations, batch_ids = self.envs.reset() # the i-th observation of different worker (running the same mdp), id of the worker
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad(): # temporarily set all parameters requires_grad=False
                observations_tensor = torch.from_numpy(observations).to(device=device) # turn numpy image to torch tensor
                # policy.forward() returns the distribution
                actions_tensor = policy(observations_tensor, params=params).sample() # sample action from policy given s
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
        return episodes #

    def reset_task(self, task):
        """
        assign the same task(mdp) to num_workers sub-processes to run in parallel
        :param task:
        :return:
        """
        tasks = [task for _ in range(self.num_workers)] # 1 dynamics - n_worker
        reset = self.envs.reset_task(tasks) # n worker set their tasks to the specified
        return all(reset)

    def sample_tasks(self, num_tasks):
        """
        sample num_tasks mdp of the same env
        :param num_tasks:
        :return:
        """
        tasks = self._env.unwrapped.sample_tasks(num_tasks) # sample num_tasks mdp
        return tasks
