from gymnasium.envs.registration import register
    

register(
    id="NetworkGrid-v0",
    entry_point="network_scenario_env:NetworkGrid",
)