# from .cy_environment import env_creator, make # if you want to use the cython version, use this instead
from .py_environment import env_creator, make

try:
    import torch
except ImportError:
    pass
else:
    # from .torch import Policy
    try:
        from .torch import Recurrent
    except:
        Recurrent = None
