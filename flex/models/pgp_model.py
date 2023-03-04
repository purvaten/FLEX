from flex.tools.registry import registry
from torch import nn as nn


@registry.register_class(name='Regress')
class Regress(nn.Module):
    '''
    Description:
    - Data
        Input: body_pose
        Output: body pitch, roll and z-transl
    - Model
        2-layer MLP with ReLU
    '''
    def __init__(self, cfg):

        super(Regress, self).__init__()

        # Initialization.
        self.cfg = cfg
        self.out_size = 3

        # Model.
        self.fc1 = nn.Linear(63, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 3)


    def forward(self, X, **kwargs):
        '''
        :param X            (torch.Tensor) -- (b, 63)
        :return out_dict    (dict) -- of tensors of size (b,) each.
        '''
        x = self.fc1(X)    # (b, 32)
        x = self.relu(x)   # (b, 32)
        out = self.fc2(x)  # (b, 3)
        out_dict = {self.cfg.height: out[:,0], 'pitch': out[:,1], 'roll': out[:,2]}
        return out_dict