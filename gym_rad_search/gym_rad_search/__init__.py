from gym.envs.registration import register

register(
    id="RadSearchMulti-v1", entry_point="gym_rad_search.envs:RadSearch",
)
