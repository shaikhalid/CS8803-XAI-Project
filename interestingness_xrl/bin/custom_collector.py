from tianshou.data import Collector

class CustomCollector(Collector):
    def __init__(self, helper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Init")
        self.observations = []  # Store observations here
        self.helper = helper

    def collect(self, n_step=0, random=False, render=None):
        result = super().collect(n_step)
        old_obs = None
        old_s = None
        t = 0
        for i in range(n_step):
            obs = self.buffer.obs[i]
            r = self.buffer.rew[i]
            done = self.buffer.done[i]
            a = self.buffer.act[i]
            s = self.helper.get_state_from_observation(obs, r, done)
            if(old_obs is not None and old_s is not None):
                #self.helper.update_stats(e, t, old_obs, obs, old_s, a, r, s)
                print(i, t, old_obs, obs, old_s, a, r, s)
            old_obs = obs
            old_s = s
            t += 1
        return result

# Then, when you create your collector, use CustomCollector
# collector = CustomCollector(...)
