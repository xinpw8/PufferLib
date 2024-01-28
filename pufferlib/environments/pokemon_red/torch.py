import pufferlib.models

# Unaltered code through line 20
class Recurrent(pufferlib.models.RecurrentWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

class Policy(pufferlib.models.Convolutional):
    def __init__(self, env, input_size=512, hidden_size=512, output_size=512,
            framestack=4, flat_size=64*5*6):
        super().__init__(
            env=env,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            framestack=framestack,
            flat_size=flat_size,
            channels_last=True,
        )

# class Policy(pufferlib.models.ProcgenBET):
#     def __init__(self, env, input_size=64, hidden_size=64, output_size=64, flat_size=16*5*9
#        ): # framestack=4, hidden_size=512, flat_size=64*5*6
#         super().__init__(
#             env=env,
#             input_size=input_size,
#             hidden_size=hidden_size,
#             output_size=output_size,
#             # framestack=framestack,
#             flat_size=flat_size,
#             channels_last=True,
#         )

# class Recurrent(pufferlib.models.RecurrentWrapper):
#     def __init__(self, env, policy, input_size=64, hidden_size=64, num_layers=1):
#         super().__init__(env, policy, input_size, hidden_size, num_layers)


