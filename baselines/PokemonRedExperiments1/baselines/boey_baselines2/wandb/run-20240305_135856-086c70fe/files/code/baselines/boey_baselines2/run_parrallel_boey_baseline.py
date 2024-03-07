from os.path import exists
from pathlib import Path
import uuid
# from baselines.boey_baselines2.red_gym_env import RedGymEnvV3 as RedGymEnv
# from stable_baselines3 import PPO
# from stable_baselines3.common import env_checker
# from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# from stable_baselines3.common.utils import set_random_seed
# from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
# from baselines.boey_baselines2.custom_network import CustomCombinedExtractorV2
# from baselines.boey_baselines2.tensorboard_callback import TensorboardCallback, GammaScheduleCallback

from red_gym_env import RedGymEnvV3 as RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from custom_network import CustomCombinedExtractorV2
from tensorboard_callback import TensorboardCallback, GammaScheduleCallback
from typing import Callable

from pdb import set_trace as T
import argparse
import shutil
import sys
import os

import importlib
import inspect
import yaml

import pufferlib
import pufferlib.utils

# import clean_pufferl
import torch
from stream_wrapper import StreamWrapper


def make_env(rank, env_conf, seed=0, es_min_reward_list=None):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env_config['env_id'] = rank
        if es_min_reward_list:
            env_config['early_stopping_min_reward'] = es_min_reward_list[rank]
        env = RedGymEnv(env_conf)
        env = StreamWrapper(env, stream_metadata={"user": "PUFFERBOX3 |BET| \n=BOEY=\n"})
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

def create_callbacks(use_wandb_logging=False, save_state_dir=None):
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=sess_path,
                                     name_prefix='poke')
    # gamma_schedule_callback = GammaScheduleCallback(init_gamma=0.9996, target_gamma=0.9999, given_timesteps=60_000_000, start_from=9_830_400)
    
    # callbacks = [checkpoint_callback, TensorboardCallback(save_state_dir=save_state_dir), gamma_schedule_callback]
    callbacks = [checkpoint_callback, TensorboardCallback(save_state_dir=save_state_dir)]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            config=env_config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
        )
        callbacks.append(WandbCallback())
    else:
        run = None
    return callbacks, run

if __name__ == '__main__':

    use_wandb_logging = True
    cpu_multiplier = 0.5  # For R9 7950x: 1.0 for 32 cpu, 1.25 for 40 cpu, 1.5 for 48 cpu
    ep_length = 1024 * 1000 * 30  # 30m steps
    save_freq = 2048 * 10 * 2
    n_steps = int(5120 // cpu_multiplier) * 1
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'/bet_adsorption_xinpw8/PufferLib/baselines/PokemonRedExperiments1/baselines/boey_baselines2/running/session_{sess_id}_env19_lr3e-4_ent01_bs2048_ep3_5120_vf05_release')
    num_cpu = int(32 * cpu_multiplier)  # Also sets the number of episodes per training iteration
    state_dir = Path('/bet_adsorption_xinpw8/PufferLib/baselines/PokemonRedExperiments1/baselines/boey_baselines2/states/env19_release')
    env_config = {
                'headless': True, 'save_final_state': True, 
                'early_stop': True,  # resumed early stopping to ensure reward signal
                'action_freq': 24, 'init_state': 'has_pokedex_nballs_noanim.state', 'max_steps': ep_length, 
                # 'env_max_steps': env_max_steps,
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': 'PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
                'use_screen_explore': False, 'reward_scale': 4, 
                'extra_buttons': False, 'restricted_start_menu': False, 
                'noop_button': True,
                'swap_button': True,
                'enable_item_manager': True,
                'level_reward_badge_scale': 1.0,
                # 'randomize_first_ep_split_cnt': num_cpu,
                # 'start_from_state_dir': state_dir, 
                'save_state_dir': state_dir,
                'explore_weight': 1.5, # 3
                'special_exploration_scale': 1.0,  # double the exploration for special maps (caverns)
                'enable_stage_manager': True,
                'enable_item_purchaser': True,
                'auto_skip_anim': True,
                'auto_skip_anim_frames': 8,
                'early_stopping_min_reward': 2.0,
                'total_envs': num_cpu,
                'level_manager_eval_mode': False,  # True = full run
                # 'randomization': 0.3,
            }
    
    print(env_config)

    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)], start_method='spawn')
    

    learn_steps = 1
    # put a checkpoint here you want to start from
    file_name = ''
    if file_name and not exists(file_name + '.zip'):
        raise Exception(f'File {file_name} does not exist!')
    
    def warmup_schedule(initial_value: float) -> Callable[[float], float]:
        """
        Linear learning rate schedule.

        :param initial_value: Initial learning rate.
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            one_update = 0.000125
            n_update = 2
            if progress_remaining > (1 - (one_update * n_update)):  # was warmup for 16 updates 81920 steps, 2.6m total steps.
                return 0.0
            else:
                return initial_value

        return func

    if exists(file_name + '.zip'):
        print(f'\nloading checkpoint: {file_name}')
        new_gamma = 0.9996
        model = PPO.load(file_name, env=env, ent_coef=0.01, n_epochs=1, gamma=new_gamma)  # , learning_rate=warmup_schedule(0.0003)
        print(f'Loaded model1 --- LR: {model.learning_rate} OptimizerLR: {model.policy.optimizer.param_groups[0]["lr"]}, ent_coef: {model.ent_coef}, n_epochs: {model.n_epochs}, n_steps: {model.n_steps}, batch_size: {model.batch_size}, gamma: {model.gamma}, rollout_buffer.gamma: {model.rollout_buffer.gamma}')
        model.gamma = new_gamma
        model.rollout_buffer.gamma = new_gamma
        print(model.policy)
        print(f'Loaded model3 --- LR: {model.learning_rate} OptimizerLR: {model.policy.optimizer.param_groups[0]["lr"]}, ent_coef: {model.ent_coef}, gamma: {model.gamma}, rollout_buffer.gamma: {model.rollout_buffer.gamma}')
    else:
        print('\ncreating new model with [512, 512] fully shared layer')
        import torch
        policy_kwargs = dict(
            features_extractor_class=CustomCombinedExtractorV2,
            share_features_extractor=True,
            net_arch=[1024, 1024],  # dict(pi=[256, 256], vf=[256, 256])
            activation_fn=torch.nn.ReLU,
        )
        model = PPO('MultiInputPolicy', env, verbose=1, n_steps=n_steps, batch_size=2048, n_epochs=3, gamma=0.999, tensorboard_log=sess_path,
                    ent_coef=0.01, learning_rate=0.0003, vf_coef=0.5,  # target_kl=0.01,
                    policy_kwargs=policy_kwargs)
         # , policy_kwargs={'net_arch': dict(pi=[1024, 1024], vf=[1024, 1024])}
        
        print(model.policy)

        print(f'start training --- LR: {model.learning_rate} OptimizerLR: {model.policy.optimizer.param_groups[0]["lr"]}, ent_coef: {model.ent_coef}')
    
    callbacks, run = create_callbacks(use_wandb_logging, save_state_dir=state_dir)
    
    for i in range(learn_steps):
        model.learn(total_timesteps=(81_920)*num_cpu*1000*10, callback=CallbackList(callbacks), reset_num_timesteps=True)

        
    if run:
        run.finish()

# BET ADDED begin demo.py
def load_from_config(env):
    with open('config_jsuarez.yaml') as f:
        config = yaml.safe_load(f)

    assert env in config, f'"{env}" not found in config.yaml. Uncommon environments that are part of larger packages may not have their own config. Specify these manually using the parent package, e.g. --config atari --env MontezumasRevengeNoFrameskip-v4.'

    default_keys = 'env train policy recurrent sweep_metadata sweep_metric sweep'.split()
    defaults = {key: config.get(key, {}) for key in default_keys}

    # Package and subpackage (environment) configs
    env_config = config[env]
    pkg = env_config['package']
    pkg_config = config[pkg]

    combined_config = {}
    for key in default_keys:
        env_subconfig = env_config.get(key, {})
        pkg_subconfig = pkg_config.get(key, {})
    

        # Override first with pkg then with env configs
        try:
            combined_config[key] = {**defaults[key], **pkg_subconfig, **env_subconfig}
            # print(f'combo_config: {combined_config[key]}')
        except TypeError as e:
            pass
            # print(f'combined_config={combined_config}')
            # print(f' {type(e)} ')
            # print(f'key={type(key)}; combined_config[{key}]=sad')
        finally:
            # print(f'{key} has caused its last problem.')
            pass

    return pkg, pufferlib.namespace(**combined_config)
   
def make_policy(env, env_module, args):
    policy = env_module.Policy(env, **args.policy)
    if args.force_recurrence or env_module.Recurrent is not None:
        policy = env_module.Recurrent(env, policy, **args.recurrent)
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)
    
    # BET ADDED 1
    # mode = "default"
    # if args.train.device == "cuda":
    #     mode = "reduce-overhead"
    #     policy = policy.to(args.train.device, non_blocking=True)
    #     policy.get_value = torch.compile(policy.get_value, mode=mode)
    #     policy.get_action_and_value = torch.compile(policy.get_action_and_value, mode=mode)

    return policy.to(args.train.device)

def init_wandb(args, env_module, name=None, resume=True):
    #os.environ["WANDB_SILENT"] = "true"

    import wandb
    return wandb.init(
        id=args.exp_name or wandb.util.generate_id(),
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config={
            'cleanrl': args.train,
            'env': args.env,
            'policy': args.policy,
            'recurrent': args.recurrent,
        },
        name=name or args.config,
        monitor_gym=True,
        save_code=True,
        resume=resume,
    )

def sweep(args, env_module, make_env):
    import wandb
    sweep_id = wandb.sweep(sweep=args.sweep, project="pufferlib")

    def main():
        try:
            args.exp_name = init_wandb(args, env_module)
            if hasattr(wandb.config, 'train'):
                # TODO: Add update method to namespace
                print(args.train.__dict__)
                print(wandb.config.train)
                args.train.__dict__.update(dict(wandb.config.train))
            train(args, env_module, make_env)
        except Exception as e:
            import traceback
            traceback.print_exc()

    wandb.agent(sweep_id, main, count=20)

def get_init_args(fn):
    if fn is None:
        return {}

    sig = inspect.signature(fn)
    args = {}
    for name, param in sig.parameters.items():
        if name in ('self', 'env', 'policy'):
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        else:
            args[name] = param.default if param.default is not inspect.Parameter.empty else None
    # print(f'ARGS LINE116 DEMO.PY: {args}\n\n')
    return args

def train(args, env_module, make_env):
    if args.backend == 'clean_pufferl':
        data = clean_pufferl.create(
            config=args.train,
            agent_creator=make_policy,
            agent_kwargs={'env_module': env_module, 'args': args},
            env_creator=make_env,
            env_creator_kwargs=args.env,
            vectorization=args.vectorization,
            exp_name=args.exp_name,
            track=args.track,
        )

        while not clean_pufferl.done_training(data):
            clean_pufferl.evaluate(data)
            clean_pufferl.train(data)

        print('Done training. Saving data...')
        clean_pufferl.close(data)
        print('Run complete')
    elif args.backend == 'sb3':
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.env_util import make_vec_env
        from sb3_contrib import RecurrentPPO

        envs = make_vec_env(lambda: make_env(**args.env),
            n_envs=args.train.num_envs, seed=args.train.seed, vec_env_cls=DummyVecEnv)

        model = RecurrentPPO("CnnLstmPolicy", envs, verbose=1,
            n_steps=args.train.batch_rows*args.train.bptt_horizon,
            batch_size=args.train.batch_size, n_epochs=args.train.update_epochs,
            gamma=args.train.gamma
        )

        model.learn(total_timesteps=args.train.total_timesteps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse environment argument', add_help=False)
    parser.add_argument('--backend', type=str, default='sb3', help='Train backend (clean_pufferl, sb3)')
    parser.add_argument('--config', type=str, default='pokemon_red', help='Configuration in config.yaml to use')
    parser.add_argument('--env', type=str, default=None, help='Name of specific environment to run')
    parser.add_argument('--mode', type=str, default='train', choices='train sweep evaluate'.split())
    parser.add_argument('--eval-model-path', type=str, default=None, help='Path to model to evaluate')
    parser.add_argument('--baseline', action='store_true', help='Baseline run')
    parser.add_argument('--no-render', action='store_true', help='Disable render during evaluate')
    parser.add_argument('--exp-name', type=str, default=None, help="Resume from experiment")
    parser.add_argument('--vectorization', type=str, default='multiprocessing', choices='serial multiprocessing ray'.split())
    parser.add_argument('--wandb-entity', type=str, default='xinpw8', help='WandB entity')
    parser.add_argument('--wandb-project', type=str, default='pufferlib', help='WandB project')
    parser.add_argument('--wandb-group', type=str, default='debug', help='WandB group')
    parser.add_argument('--track', action='store_true', help='Track on WandB')
    parser.add_argument('--force-recurrence', action='store_true', help='Force model to be recurrent, regardless of defaults')

    clean_parser = argparse.ArgumentParser(parents=[parser])
    args = parser.parse_known_args()[0].__dict__
    pkg, config = load_from_config(args['config'])

    try:
        env_module = importlib.import_module(f'pufferlib.environments.{pkg}')
    except:
        pufferlib.utils.install_requirements(pkg)
        env_module = importlib.import_module(f'pufferlib.environments.{pkg}')

    # Get the make function for the environment
    env_name = args['env'] or config.env.pop('name')
    make_env = env_module.env_creator(env_name)

    # Update config with environment defaults
    config.env = {**get_init_args(make_env), **config.env}
    # print(f'config.env={config.env}')
    config.policy = {**get_init_args(env_module.Policy.__init__), **config.policy}
    # print(f'config.policy={config.policy}')
    config.recurrent = {**get_init_args(env_module.Recurrent.__init__), **config.recurrent}
    # print(f'config.recurrent={config.recurrent}')

    # Generate argparse menu from config
    for name, sub_config in config.items():
        args[name] = {}
        for key, value in sub_config.items():
            data_key = f'{name}.{key}'
            cli_key = f'--{data_key}'.replace('_', '-')
            if isinstance(value, bool) and value is False:
                action = 'store_false'
                parser.add_argument(cli_key, default=value, action='store_true')
                clean_parser.add_argument(cli_key, default=value, action='store_true')
            elif isinstance(value, bool) and value is True:
                data_key = f'{name}.no_{key}'
                cli_key = f'--{data_key}'.replace('_', '-')
                parser.add_argument(cli_key, default=value, action='store_false')
                clean_parser.add_argument(cli_key, default=value, action='store_false')
            else:
                parser.add_argument(cli_key, default=value, type=type(value))
                clean_parser.add_argument(cli_key, default=value, metavar='', type=type(value))

            args[name][key] = getattr(parser.parse_known_args()[0], data_key)
        args[name] = pufferlib.namespace(**args[name])

    clean_parser.parse_args(sys.argv[1:])
    args = pufferlib.namespace(**args)

    vec = args.vectorization
    if vec == 'serial':
        args.vectorization = pufferlib.vectorization.Serial
    elif vec == 'multiprocessing':
        args.vectorization = pufferlib.vectorization.Multiprocessing
    elif vec == 'ray':
        args.vectorization = pufferlib.vectorization.Ray
    else:
        raise ValueError(f'Invalid --vectorization (serial/multiprocessing/ray).')

    if args.mode == 'sweep':
        args.track = True
    elif args.track:
        args.exp_name = init_wandb(args, env_module).id
    elif args.baseline:
        args.track = True
        version = '.'.join(pufferlib.__version__.split('.')[:2])
        args.exp_name = f'puf-{version}-{args.config}'
        args.wandb_group = f'puf-{version}-baseline'
        shutil.rmtree(f'experiments/{args.exp_name}', ignore_errors=True)
        run = init_wandb(args, env_module, name=args.exp_name, resume=False)
        if args.mode == 'evaluate':
            model_name = f'puf-{version}-{args.config}_model:latest'
            artifact = run.use_artifact(model_name)
            data_dir = artifact.download()
            model_file = max(os.listdir(data_dir))
            args.eval_model_path = os.path.join(data_dir, model_file)

    if args.mode == 'train':
        train(args, env_module, make_env)
        # exit(0)
    elif args.mode == 'sweep':
        sweep(args, env_module, make_env)
        # exit(0)
    elif args.mode == 'evaluate' and pkg != 'pokemon_red':
        rollout(
            make_env,
            args.env,
            agent_creator=make_policy,
            agent_kwargs={'env_module': env_module, 'args': args},
            model_path=args.eval_model_path,
            device=args.train.device
        )
    elif args.mode == 'evaluate' and pkg == 'pokemon_red':
        import pokemon_red_eval
        pokemon_red_eval.rollout(
            make_env,
            args.env,
            agent_creator=make_policy,
            agent_kwargs={'env_module': env_module, 'args': args},
            model_path=args.eval_model_path,
            device=args.train.device,
        )
    elif pkg != 'pokemon_red':
        raise ValueError('Mode must be one of train, sweep, or evaluate')