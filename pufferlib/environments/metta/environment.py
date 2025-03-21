import functools

def env_creator(name='metta'):
    return functools.partial(make, name)

def make(name, config='metta.yaml', render_mode='human', buf=None):
    '''Crafter creation function'''
    import mettagrid.mettagrid_env
    return mettagrid.mettagrid_env.make_env_from_cfg(config, render_mode)
