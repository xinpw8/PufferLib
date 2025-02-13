import numpy as np
import random
import math

from copy import deepcopy

import pufferlib
import scipy.stats

import torch
import pyro
from pyro.contrib import gp as gp
from pyro.contrib.gp.kernels import Kernel
from pyro.contrib.gp.models import GPRegression


class Space:
    def __init__(self, min, max, scale, mean, is_integer=False):
        self.min = min
        self.max = max
        self.scale = scale
        self.mean = mean # TODO: awkward to have just this normalized
        self.norm_min = self.normalize(min)
        self.norm_max = self.normalize(max)
        self.norm_mean = self.normalize(mean)
        self.is_integer = is_integer

class Linear(Space):
    def normalize(self, value):
        #assert isinstance(value, (int, float))
        zero_one = (value - self.min)/(self.max - self.min)
        return 2*zero_one - 1

    def unnormalize(self, value):
        zero_one = (value + 1)/2
        value = zero_one * (self.max - self.min) + self.min
        if self.is_integer:
            value = round(value)
        return value

class Pow2(Space):
    def normalize(self, value):
        #assert isinstance(value, (int, float))
        #assert value != 0.0
        zero_one = (math.log(value, 2) - math.log(self.min, 2))/(math.log(self.max, 2) - math.log(self.min, 2))
        return 2*zero_one - 1

    def unnormalize(self, value):
        zero_one = (value + 1)/2
        log_spaced = zero_one*(math.log(self.max, 2) - math.log(self.min, 2)) + math.log(self.min, 2)
        rounded = round(log_spaced)
        return 2 ** rounded

class Log(Space):
    base: int = 10

    def normalize(self, value):
        #assert isinstance(value, (int, float))
        #assert value != 0.0
        zero_one = (math.log(value, self.base) - math.log(self.min, self.base))/(math.log(self.max, self.base) - math.log(self.min, self.base))
        return 2*zero_one - 1

    def unnormalize(self, value):
        zero_one = (value + 1)/2
        log_spaced = zero_one*(math.log(self.max, self.base) - math.log(self.min, self.base)) + math.log(self.min, self.base)
        value = self.base ** log_spaced
        if self.is_integer:
            value = round(value)
        return value

class Logit(Space):
    base: int = 10

    def normalize(self, value):
        #assert isinstance(value, (int, float))
        #assert value != 0.0
        #assert value != 1.0
        zero_one = (math.log(1-value, self.base) - math.log(1-self.min, self.base))/(math.log(1-self.max, self.base) - math.log(1-self.min, self.base))
        return 2*zero_one - 1

    def unnormalize(self, value):
        zero_one = (value + 1)/2
        log_spaced = zero_one*(math.log(1-self.max, self.base) - math.log(1-self.min, self.base)) + math.log(1-self.min, self.base)
        return 1 - self.base**log_spaced


def _carbs_params_from_puffer_sweep(sweep_config):
    param_spaces = {}
    for name, param in sweep_config.items():
        if name in ('method', 'name', 'metric', 'max_score'):
            continue

        assert isinstance(param, dict)
        if any(isinstance(param[k], dict) for k in param):
            param_spaces[name] = _carbs_params_from_puffer_sweep(param)
            continue
 
        assert 'distribution' in param
        distribution = param['distribution']
        search_center = param['mean']
        kwargs = dict(
            min=param['min'],
            max=param['max'],
            scale=param['scale'],
            mean=search_center,
        )
        if distribution == 'uniform':
            space = Linear(**kwargs)
        elif distribution == 'int_uniform':
            space = Linear(**kwargs, is_integer=True)
        elif distribution == 'uniform_pow2':
            space = Pow2(**kwargs, is_integer=True)
        elif distribution == 'log_normal':
            space = Log(**kwargs)
        elif distribution == 'logit_normal':
            space = Logit(**kwargs)
        else:
            raise ValueError(f'Invalid distribution: {distribution}')

        param_spaces[name] = space

    return param_spaces

def sample_normal(mu, sigma, num_samples):
    n_input, n_dim = mu.shape
    mu_idxs = np.random.randint(0, n_input, num_samples)
    return sigma*np.random.randn(num_samples, n_dim) + mu[mu_idxs]

def sample_uniform(mu, scale, num_samples):
    n_input, n_dim = mu.shape
    mu_idxs = np.random.randint(0, n_input, num_samples)
    return 2*scale*np.random.rand(num_samples, n_dim) - scale + mu[mu_idxs]

def fill(spaces, flat_sample, idx=0):
    for name, space in spaces.items():
        if isinstance(space, dict):
            idx = fill(spaces[name], flat_sample, idx=idx)
        else:
            spaces[name] = spaces[name].unnormalize(flat_sample[idx])
            idx += 1

    return idx

def pareto_points(observations):
    scores = np.array([e['output'] for e in observations])
    costs = np.array([e['cost'] for e in observations])
    pareto = []
    idxs = []
    for idx, obs in enumerate(observations):
        # TODO: Ties and groups
        higher_score = scores > scores[idx]
        lower_cost = costs < costs[idx]
        better = higher_score & lower_cost
        if not better.any():
            pareto.append(obs)
            idxs.append(idx)

    return pareto, idxs

def create_gp(x_dim, scale_length=1.0):
    # Dummy data
    X = scale_length * torch.ones((1, x_dim))
    y = torch.zeros((1,))

    matern_kernel = gp.kernels.Matern32(input_dim=x_dim, lengthscale=X)
    #linear_kernel = gp.kernels.Linear(x_dim)
    linear_kernel = gp.kernels.Polynomial(x_dim, degree=1)
    kernel = gp.kernels.Sum(linear_kernel, matern_kernel)
    #kernel = matern_kernel

    # Params taken from HEBO: https://arxiv.org/abs/2012.03826
    model = gp.models.GPRegression(X, y, kernel=kernel, jitter=1.0e-4)
    model.noise = pyro.nn.PyroSample(pyro.distributions.LogNormal(math.log(1e-2), 0.5))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    return model, optimizer

def quantile_transform(raw_scores):
    n_quantiles = int(np.sqrt(len(raw_scores)))
    q = np.linspace(0, 1, n_quantiles, endpoint=True)
    quantiles = np.quantile(raw_scores, q)
    p = np.interp(raw_scores, quantiles, q)
    p = np.clip(p, 0.01, 0.99)
    normalized_scores = np.sqrt(2) * scipy.special.erfinv(2*p - 1)
    return normalized_scores

class PufferCarbs:
    def __init__(self,
            sweep_config,
            max_suggestion_cost = None,
            resample_frequency = 5,
            num_random_samples = 10,
            global_search_scale = 1,
            random_suggestions = 1024,
            suggestions_per_pareto = 128,
        ):
        self.spaces = _carbs_params_from_puffer_sweep(sweep_config)
        self.flat_spaces = dict(pufferlib.utils.unroll_nested_dict(self.spaces))
        self.num_params = len(self.flat_spaces)

        self.metric = sweep_config['metric']
        self.max_score = sweep_config['max_score']

        assert self.metric['goal'] in ['maximize', 'minimize']
        self.optimize_direction = 1 if self.metric['goal'] == 'maximize' else -1

        self.num_random_samples = num_random_samples
        self.global_search_scale = global_search_scale
        self.random_suggestions = random_suggestions
        self.suggestions_per_pareto = suggestions_per_pareto
        self.resample_frequency = resample_frequency
        self.max_suggestion_cost = max_suggestion_cost

        self.success_observations = []
        self.failure_observations = []
        self.suggestion_idx = 0

        self.search_centers = np.array([
            e.norm_mean for e in self.flat_spaces.values()])
        self.min_bounds = np.array([
            e.norm_min for e in self.flat_spaces.values()])
        self.max_bounds = np.array([
            e.norm_max for e in self.flat_spaces.values()])
        self.search_scales = global_search_scale * np.array([
            e.scale for e in self.flat_spaces.values()])

        print('Min random sample:')
        for name, space in self.flat_spaces.items():
            print(f'\t{name}: {space.unnormalize(max(space.norm_mean - space.scale, space.norm_min))}')

        print('Max random sample:')
        for name, space in self.flat_spaces.items():
            print(f'\t{name}: {space.unnormalize(min(space.norm_mean + space.scale, space.norm_max))}')

        self.gp_score, self.score_opt = create_gp(self.num_params)
        self.gp_cost, self.cost_opt = create_gp(self.num_params)

    def suggest(self):
        self.suggestion_idx += 1
        # TODO: Clip random samples to bounds so we don't get bad high cost samples
        info = {}
        if self.suggestion_idx <= self.num_random_samples:
            suggestions = sample_uniform(
                self.search_centers[None, :], self.search_scales, self.random_suggestions)
            suggestions = np.clip(suggestions, self.min_bounds, self.max_bounds)
            best_idx = np.random.randint(0, self.random_suggestions)
            best = suggestions[best_idx]
        elif self.resample_frequency and self.suggestion_idx % self.resample_frequency == 0:
            candidates, _ = pareto_points(self.success_observations)
            suggestions = np.stack([e['input'] for e in candidates])
            best_idx = np.random.randint(0, len(candidates))
            best = suggestions[best_idx]
        else:
            params = np.array([e['input'] for e in self.success_observations])
            params = torch.from_numpy(params)
            eps = 1e-5

            # Scores variable y
            y = self.optimize_direction*np.array([e['output'] for e in self.success_observations])
            y_max = np.max(y)
            y_mean = np.mean(y)
            y_std = np.std(y)

            # Transformed scores
            max_score = self.max_score
            if max_score is None:
                max_score = np.max(y) + abs(np.max(y))

            yt = -np.log(1 - y/max_score + eps)
            yt_mean = np.mean(yt)
            yt_std = np.std(yt)
            yt_norm = (yt - yt_mean) / yt_std
            yt_norm_min = np.min(yt_norm)
            yt_norm = torch.from_numpy(yt_norm)
            self.gp_score.mean_function = lambda x: yt_norm_min
            self.gp_score.set_data(params, yt_norm)
            self.gp_score.train()
            gp.util.train(self.gp_score, self.score_opt)
            self.gp_score.eval()

            # Log costs
            c = np.array([e['cost'] for e in self.success_observations])
            log_c = np.log(c)
            log_c_max = np.max(log_c)
            log_c = torch.from_numpy(log_c)
            self.gp_cost.mean_function = lambda x: log_c_max
            self.gp_cost.set_data(params, log_c)
            self.gp_cost.train()
            gp.util.train(self.gp_cost, self.cost_opt)
            self.gp_cost.eval()

            ### Sample suggestions
            candidates, pareto_idxs = pareto_points(self.success_observations)
            search_centers = np.stack([e['input'] for e in candidates])
            suggestions = sample_normal(search_centers,
                self.search_scales, len(candidates)*self.suggestions_per_pareto)
            suggestions = np.clip(suggestions, self.min_bounds, self.max_bounds)

            ### Predict scores and costs
            suggestions = torch.from_numpy(suggestions)
            with torch.no_grad():
                gp_yt_norm, gp_yt_norm_var = self.gp_score(suggestions)
                gp_log_c, gp_log_c_var = self.gp_cost(suggestions)

            gp_c = np.exp(gp_log_c.numpy())
            gp_yt = (gp_yt_norm.numpy() * yt_std) + yt_mean
            gp_y = -max_score*(np.exp(-gp_yt) - 1 - eps)

            pareto_y = y[pareto_idxs]
            pareto_yt = yt[pareto_idxs]
            pareto_c = c[pareto_idxs]

            c_diff = gp_c[:, None] - pareto_c[None, :]
            pareto_dist = np.abs(c_diff)
            nearest_pareto_dist = np.min(pareto_dist, axis=1)

            c_diff[c_diff < 0] = np.inf
            nearest_idx = np.argmin(c_diff, axis=1)
            nearest_pareto_yt = pareto_yt[nearest_idx]
            nearest_pareto_y = pareto_y[nearest_idx]

            max_c_mask = gp_c < self.max_suggestion_cost
            suggestion_scores = max_c_mask * (gp_yt - nearest_pareto_yt) * nearest_pareto_dist

            # This works and uncovers approximate binary search when the GP is perfect
            # Can't include cost in denom because it biases this case
            # Instead, use conservative score and/or cost estimates
            # Just need to figure out why the GP is overconfident

            best_idx = np.argmax(suggestion_scores)
            info = dict(
                cost = gp_c[best_idx].item(),
                score = gp_y[best_idx].item(),
                nearby = nearest_pareto_y[best_idx].item(),
                dist = nearest_pareto_dist[best_idx].item(),
                rating = suggestion_scores[best_idx].item(),
            )
            print('Predicted -- ',
                f'Score: {info["score"]:.3f}',
                f'Nearby: {info["nearby"]:.3f}',
                f'Dist: {info["dist"]:.3f}',
                f'Cost: {info["cost"]:.3f}',
                f'Rating: {info["rating"]:.3f}',
            )

            best = suggestions[best_idx].numpy()

            '''
            from bokeh.models import ColumnDataSource, LinearColorMapper
            from bokeh.plotting import figure, show
            from bokeh.palettes import Turbo256


            # Create a ColumnDataSource that includes the 'order' for each point
            source = ColumnDataSource(data=dict(
                x=costs,
                y=raw_scores,
                order=list(range(len(costs)))  # index/order for each point
            ))

            test_source = ColumnDataSource(data=dict(
                x=cost_mean,
                y=untransformed_score_mean,
                order=list(range(len(cost_mean)))  # index/order for each point
            ))

            # Define a color mapper across the range of point indices
            mapper = LinearColorMapper(
                palette=Turbo256,
                low=0,
                high=len(costs)
            )

            # Set up the figure
            p = figure(title='Synthetic Hyperparam Test', 
                       x_axis_label='Cost', 
                       y_axis_label='Score')

            p.scatter(x='x', 
                      y='y',
                      color='purple',
                      size=10, 
                      source=test_source)

            # Use the 'order' field for color -> mapped by 'mapper'
            p.scatter(x='x', 
                      y='y', 
                      color={'field': 'order', 'transform': mapper}, 
                      size=10, 
                      source=source)

            show(p)

            exit()
            '''


        self.suggestion = best
        params = deepcopy(self.spaces)
        fill(params, best)
        return params, info


    def observe(self, score, cost, is_failure=False):
        self.success_observations.append(dict(
            input=self.suggestion,
            output=score,
            cost=cost,
            is_failure=is_failure,
        ))

def log_normal(min, max, mean, scale):
    '''Samples normally spaced points on a log 10 scale.
    mean: Your center sample point
    scale: standard deviation in base 10 orders of magnitude
    clip: maximum standard deviations

    Example: mean=0.001, scale=1, clip=2 will produce data from
    0.1 to 0.00001 with most of it between 0.01 and 0.0001
    '''
    return np.clip(
        10**np.random.normal(
            np.log10(mean),
            scale,
        ),
        a_min = min,
        a_max = max,
    )

def logit_normal(min, max, mean, scale):
    '''log normal but for logit data like gamma and gae_lambda'''
    return 1 - log_normal(1-max, 1-min, 1-mean, scale)

def uniform_pow2(min, max):
    '''Uniform distribution over powers of 2 between min and max inclusive'''
    min_base = np.log2(min)
    max_base = np.log2(max)
    return 2**np.random.randint(min_base, max_base+1)

def uniform(min, max):
    '''Uniform distribution between min and max inclusive'''
    return np.random.uniform(min, max)

def int_uniform(min, max):
    '''Uniform distribution between min and max inclusive'''
    return np.random.randint(min, max+1)

def sample_hyperparameters(sweep_config):
    samples = {}
    for name, param in sweep_config.items():
        if name in ('method', 'name', 'metric'):
            continue

        assert isinstance(param, dict)
        if any(isinstance(param[k], dict) for k in param):
            samples[name] = sample_hyperparameters(param)
        elif 'values' in param:
            assert 'distribution' not in param
            samples[name] = random.choice(param['values'])
        elif 'distribution' in param:
            if param['distribution'] == 'uniform':
                samples[name] = uniform(param['min'], param['max'])
            elif param['distribution'] == 'int_uniform':
                samples[name] = int_uniform(param['min'], param['max'])
            elif param['distribution'] == 'uniform_pow2':
                samples[name] = uniform_pow2(param['min'], param['max'])
            elif param['distribution'] == 'log_normal':
                samples[name] = log_normal(
                    param['min'], param['max'], param['mean'], param['scale'])
            elif param['distribution'] == 'logit_normal':
                samples[name] = logit_normal(
                    param['min'], param['max'], param['mean'], param['scale'])
            else:
                raise ValueError(f'Invalid distribution: {param["distribution"]}')
        else:
            raise ValueError('Must specify either values or distribution')

    return samples


