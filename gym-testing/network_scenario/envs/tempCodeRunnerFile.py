    truncated = False
                reward = self._get_reward()
                observation = self._get_obs()
                info = self._get_info()  
                return observation, reward, terminated, truncated, info