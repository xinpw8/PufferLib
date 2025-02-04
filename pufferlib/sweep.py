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
        if name in ('method', 'name', 'metric'):
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

def sample(mu, sigma, num_samples):
    n_input, n_dim = mu.shape
    mu_idxs = np.random.randint(0, n_input, num_samples)
    return sigma*np.random.randn(num_samples, n_dim) + mu[mu_idxs]

def fill(spaces, flat_sample, idx=0):
    for name, space in spaces.items():
        if isinstance(space, dict):
            fill(spaces[name], flat_sample, idx=idx)
        else:
            spaces[name] = spaces[name].unnormalize(flat_sample[idx])
            idx += 1

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
    linear_kernel = gp.kernels.Linear(x_dim)
    kernel = gp.kernels.Sum(linear_kernel, matern_kernel)

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
            num_random_samples = 900,
            seed = 0,
            initial_search_radius = 0.3,
            global_search_scale = 0.1,
            num_suggestion_candidates = 2048,
        ):
        self.spaces = _carbs_params_from_puffer_sweep(sweep_config)
        self.flat_spaces = dict(pufferlib.utils.unroll_nested_dict(self.spaces))
        self.num_params = len(self.flat_spaces)

        self.metric = sweep_config['metric']

        assert self.metric['goal'] in ['maximize', 'minimize']
        self.optimize_direction = 1 if self.metric['goal'] == 'maximize' else -1

        self.seed = seed
        self.num_random_samples = num_random_samples
        self.initial_search_radius = initial_search_radius
        self.global_search_scale = global_search_scale
        self.num_suggestion_candidates = num_suggestion_candidates
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

        self.gp_score, self.score_opt = create_gp(self.num_params)
        self.gp_cost, self.cost_opt = create_gp(self.num_params)

    def suggest(self):
        self.suggestion_idx += 1
        if self.suggestion_idx <= self.num_random_samples:
            suggestions = sample(self.search_centers[None, :], 10*self.search_scales, self.num_suggestion_candidates)
            suggestions = np.clip(suggestions, self.min_bounds, self.max_bounds)
        elif self.resample_frequency and self.suggestion_idx % self.resample_frequency == 0:
            candidates, _ = pareto_points(self.success_observations)
            suggestions = np.stack([e['input'] for e in candidates])
        else:
            params = np.array([e['input'] for e in self.success_observations])
            params = torch.from_numpy(params)

            # Quantile transform. Using n_quantiles < n samples preserves distance information
            raw_scores = self.optimize_direction*np.array([e['output'] for e in self.success_observations])
            #percentile_scores = scipy.stats.rankdata(raw_scores) / (len(raw_scores) + 1)
            #percentile_mean = np.mean(percentile_scores)
            #percentile_std = np.std(percentile_scores)
            #normalized_scores = (percentile_scores - percentile_mean) / percentile_std
            raw_score_max = np.max(raw_scores)
            raw_score_mean = np.mean(raw_scores)
            raw_score_std = np.std(raw_scores)
            normalized_scores = (raw_scores - raw_score_mean) / raw_score_std
            '''
            n_quantiles = int(np.sqrt(len(self.success_observations)))
            q = np.linspace(0, 1, n_quantiles, endpoint=True)
            quantiles = np.quantile(raw_scores, q)
            p = np.interp(raw_scores, quantiles, q)
            p = np.clip(p, 0.01, 0.99)
            normalized_scores = np.sqrt(2) * scipy.special.erfinv(2*p - 1)
            '''

            normalized_scores = torch.from_numpy(normalized_scores)
            self.gp_score.set_data(params, normalized_scores)
            self.gp_score.train()
            gp.util.train(self.gp_score, self.score_opt)
            self.gp_score.eval()

            # Min-max scale
            costs = np.array([e['cost'] for e in self.success_observations])
            log_costs = np.log(costs)
            min_log_cost = log_costs.min(axis=0)
            max_log_cost = log_costs.max(axis=0)
            log_cost_std = (log_costs - min_log_cost) / (max_log_cost - min_log_cost)
            normalized_log_costs = log_cost_std*(max_log_cost - min_log_cost) + min_log_cost
            normalized_log_costs = torch.from_numpy(normalized_log_costs)
            self.gp_cost.set_data(params, normalized_log_costs)
            self.gp_cost.train()
            gp.util.train(self.gp_cost, self.cost_opt)
            self.gp_cost.eval()

            ### Sample suggestions
            candidates, pareto_idxs = pareto_points(self.success_observations)
            search_centers = np.stack([e['input'] for e in candidates])
            suggestions = sample(search_centers, self.search_scales, self.num_suggestion_candidates)
            suggestions = np.clip(suggestions, self.min_bounds, self.max_bounds)

            ### Predict scores and costs
            suggestions = torch.from_numpy(suggestions)
            with torch.no_grad():
                normalized_score_mean, normalized_score_var = self.gp_score(suggestions)
                normalized_log_cost_mean, normalized_log_cost_var = self.gp_cost(suggestions)

            log_cost_std_est = (normalized_log_cost_mean - min_log_cost) / (max_log_cost - min_log_cost)
            log_cost_est = log_cost_std_est*(max_log_cost - min_log_cost) + min_log_cost
            cost_mean = np.exp(log_cost_est)

            normalized_conservative_log_cost_mean = normalized_log_cost_mean + normalized_log_cost_var
            normalized_conservative_log_cost_std_est = (normalized_conservative_log_cost_mean - min_log_cost) / (max_log_cost - min_log_cost)
            normalized_conservative_log_cost_est = normalized_conservative_log_cost_std_est*(max_log_cost - min_log_cost) + min_log_cost
            conservative_cost_mean = np.exp(normalized_conservative_log_cost_est)

            normalized_pareto_scores = normalized_scores[pareto_idxs]
            pareto_costs = torch.from_numpy(costs[pareto_idxs])
            cost_diff = cost_mean.unsqueeze(1) - pareto_costs.unsqueeze(0)
            cost_dist = torch.abs(cost_diff)
            dist_to_nearest_pareto = torch.min(cost_dist, dim=1)[0]

            cost_diff[cost_diff < 0] = torch.inf
            closest_pareto_idx = torch.argmin(cost_diff, dim=1)
            normalized_nearest_pareto_score = normalized_pareto_scores[closest_pareto_idx]

            unnormalized_score_mean = (normalized_score_mean * raw_score_std) + raw_score_mean
            unnormalized_nearest_pareto_score = (normalized_nearest_pareto_score * raw_score_std) + raw_score_mean

            conservative_normalized_score_mean = normalized_score_mean - normalized_score_var
            conservative_unnormalized_score_mean = (conservative_normalized_score_mean * raw_score_std) + raw_score_mean
            unnormalized_score_var = unnormalized_score_mean - conservative_unnormalized_score_mean

            max_cost_mask = conservative_cost_mean < self.max_suggestion_cost
            #suggestion_scores = max_cost_mask * (unnormalized_score_mean - unnormalized_nearest_pareto_score)# / cost_mean
            #suggestion_scores = max_cost_mask * unnormalized_score_mean * dist_to_nearest_pareto / cost_mean
            #suggestion_scores = max_cost_mask * (unnormalized_score_mean - unnormalized_nearest_pareto_score) * dist_to_nearest_pareto / cost_mean
            #suggestion_scores = max_cost_mask * (unnormalized_score_mean - unnormalized_nearest_pareto_score) * dist_to_nearest_pareto / conservative_cost_mean


            #suggestion_scores = max_cost_mask * (conservative_unnormalized_score_mean - unnormalized_nearest_pareto_score) * dist_to_nearest_pareto / (conservative_cost_mean * unnormalized_score_var)

            #suggestion_scores = max_cost_mask * (unnormalized_score_mean - unnormalized_nearest_pareto_score) * dist_to_nearest_pareto / cost_mean
            #suggestion_scores = max_cost_mask * dist_to_nearest_pareto
            suggestion_scores = max_cost_mask * (unnormalized_score_mean - unnormalized_nearest_pareto_score) * dist_to_nearest_pareto

            # This works and uncovers approximate binary search when the GP is perfect
            # Can't include cost in denom because it biases this case
            # Instead, use conservative score and/or cost estimates
            # Just need to figure out why the GP is overconfident

            best_idx = np.argmax(suggestion_scores)
            score = unnormalized_score_mean[best_idx].item()
            nearby = unnormalized_nearest_pareto_score[best_idx].item()
            dist = dist_to_nearest_pareto[best_idx].item()
            cost = cost_mean[best_idx].item()
            rating = suggestion_scores[best_idx].item()
            var = unnormalized_score_var[best_idx].item()
            print('Predicted -- ',
                f'Score: {score:.3f}',
                f'Nearby: {nearby:.3f}',
                f'Dist: {dist:.3f}',
                f'Cost: {cost:.3f}',
                f'Rating: {rating:.3f}',
                f'Var: {var:.3f}',
            )

            breakpoint()
            suggestions = suggestions[best_idx:best_idx+1].numpy()

        best = suggestions[0]
        self.suggestion = best

        params = deepcopy(self.spaces)
        fill(params, best)
        return params

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


