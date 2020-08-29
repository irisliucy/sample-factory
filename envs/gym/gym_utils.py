import gym


class GymSpec:
    def __init__(self, name, env_id):
        self.name = name
        self.env_id = env_id


GYM_ENVS = [
    GymSpec('gym_lunarlander_discrete', 'LunarLander-v2'),
    GymSpec('gym_lunarlander_continuous', 'LunarLanderContinuous-v2'),
]


def gym_env_by_name(name):
    for cfg in GYM_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception('Unknown Gym env')


def make_gym_env(env_name, cfg, **kwargs):
    gym_spec = gym_env_by_name(env_name)
    env = gym.make(gym_spec.env_id)
    return env


class SingleGymMulti(gym.Env):
    def __init__(self, env_name, num_agents):
        self.num_agents = num_agents
        self.envs = []
        for i in range(num_agents):
            gym_spec = gym_env_by_name(env_name)
            e = gym.make(gym_spec.env_id)
            self.envs.append(e)

        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

    def reset(self):
        obs = []
        for env in self.envs:
            obs.append(env.reset())

        return obs

    def step(self, actions):
        obs, rews, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, d, info = env.step(actions[i])
            if d:
                o = env.reset()
            obs.append(o)
            rews.append(r)
            dones.append(d)
            infos.append(info)
        return obs, rews, dones, infos


def make_gym_env_single_multi(env_name):
    num_agents = 4
    return SingleGymMulti(env_name, num_agents)


def make_gym_env(env_name, cfg=None, **kwargs):
    if env_name == 'gym_lunarlander_continuous':
        gym_spec = gym_env_by_name(env_name)
        env = gym.make(gym_spec.env_id)
        return env
    elif env_name == 'gym_lunarlander_continuous_multi':
        env_name = 'gym_lunarlander_continuous'
        return make_gym_env_single_multi(env_name)
    else:
        raise NotImplementedError()
