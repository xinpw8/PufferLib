# from .old_environment import env_creator, make
from .environment import env_creator, make

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
