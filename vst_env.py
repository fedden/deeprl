import gym
from gym import error, spaces, utils
from gym.utils import seeding
from plugin_feature_extractor import PluginFeatureExtractor
import numpy as np
import matplotlib.pyplot as plt

class DexedVstEnv(gym.Env):

    metadata = { 'render.modes': ['features_array', 'spectrograms'] }

    def __init__(self):
        algorithm_number = 18
        alg = (1.0 / 32.0) * float(algorithm_number - 1) + 0.001
        a = 4
        overriden_parameters = [(0, 1.0), (1, 0.0), (2, 1.0), (3, 0.0), (a, alg)]
        other_params = [((i + 5), 0.5) for i in range(17)]
        operator_one = [((i + 23), 0.0) for i in range(22)]
        operator_two = [((i + 45), 0.0) for i in range(22)]
        operator_thr = [((i + 67), 0.0) for i in range(22)]
        operator_fou = [((i + 89), 0.0) for i in range(22)]
        operator_fiv = [((i + 111), 0.0) for i in range(22)]
        operator_six = [((i + 133), 0.0) for i in range(22)]
        # overriden_parameters.extend(operator_one)
        overriden_parameters.extend(operator_two)
        overriden_parameters.extend(operator_thr)
        overriden_parameters.extend(operator_fou)
        overriden_parameters.extend(operator_fiv)
        overriden_parameters.extend(operator_six)
        overriden_parameters.extend(other_params)
        self.extractor = PluginFeatureExtractor(midi_note=24, note_length_secs=0.4,
                                                render_length_secs=0.7,
                                                overriden_parameters=overriden_parameters,
                                                pickle_path="/home/tollie/Development/TensorFlowSynthProgrammers/utils/normalisers/",
                                                warning_mode="ignore",
                                                normalise_audio=False)
        path = "/home/tollie/Development/vsts/dexed/Builds/Linux/build/Dexed.so"
        self.extractor.load_plugin(path)
        (features, parameters) = self.extractor.get_random_normalised_example()
        self.action_space = spaces.Discrete(len(parameters) * 2 + 1)
        self.inaction = len(parameters) * 2
        shape = (features.flatten().shape[0],)
        self.observation_space = spaces.Box(np.zeros(shape), np.ones(shape))
        self.reward = 0.0
        self.reward_range = (0, 1)
        self.reward_threshold = 0.90
        self.parameter_alpha = 0.05
        self.features = None
        self.parameters = None
        self.done = False
        self.step_number = 0
        self.step_threshold = 450
        self.patch_average_out_amount = 4

    def _reset(self):
        (features, parameters) = self.extractor.get_random_normalised_example()
        self.target_parameters = parameters
        self.target_features = features
        self.target_audio = self.extractor.get_audio_frames()
        self.done = False
        self.reward = 0.0
        self.step_number = 0
        self.parameters = np.zeros_like(parameters)
        self.get_average_features()
        return self.get_observation()

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.increment_step_count()
        if action == self.inaction or self.done:
            return self.get_observation(), self.reward, self.done, self.get_info()
        else:
            increment = self.parameter_alpha if action % 2 == 0 else -self.parameter_alpha
            action = int(np.floor(float(action) / 2.0))
            self.parameters[action] += increment
            self.parameters[action] = max(0.0, min(self.parameters[action], 1.0))
            self.get_average_features()
            self.get_reward()
            if self.reward >= self.reward_threshold:
                self.done = True
            return self.get_observation(), self.reward, self.done, self.get_info()

    def get_stats(self):
        return (self.parameters, self.features, self.target_parameters, self.target_features)

    def get_reward(self):
        feature_difference = np.subtract(self.target_features, self.features)
        abs_difference = np.absolute(feature_difference)
        summed_difference = np.sum(abs_difference)
        normalised_difference = summed_difference / self.features.size
        reversed_normalised_distance = 1.0 - normalised_difference
        self.reward = reversed_normalised_distance * 2.0 - 1.0

    def get_observation(self):
        return self.target_features - self.features

    def get_info(self):
        info = {
            'parameters': self.parameters,
            'features': self.features,
            'target_parameters': self.target_parameters,
            'target_features': self.target_features,
            'reward': self.reward
        }
        return info

    def get_average_features(self):
        patch = self.extractor.partial_patch_to_patch(self.parameters)
        patch = self.extractor.add_patch_indices(patch)
        self.features = self.extractor.get_features_from_patch(patch)
        for i in range(self.patch_average_out_amount):
            self.features += self.extractor.get_features_from_patch(patch)
        self.features /= (self.patch_average_out_amount + 1)

    def increment_step_count(self):
        if self.step_number > self.step_threshold:
            self.done = True
            self.step_number = 0
        self.step_number += 1

    def _render(self, mode='features_array', close=False):
        if mode == 'features_array':
            return self.features
        elif mode == 'spectrograms':
            fig = plt.figure(0)
            fig.suptitle('Actual Patch', fontsize=14, fontweight='bold')
            plt.xlabel('Time')

            plt.ylabel('Frequency')
            plt.specgram(self.extractor.get_audio_frames(), NFFT=256, Fs=256)
            fig = plt.figure(1)
            fig.suptitle('Target Patch', fontsize=14, fontweight='bold')
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.specgram(self.target_audio, NFFT=256, Fs=256)
            return fig
        else:
            super(DexedVstEnv, self).render(mode=mode)

    def _seed(self, seed=8):
        np.random.seed(seed)
        return [seed]
