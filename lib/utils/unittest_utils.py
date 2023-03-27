# @Author  : Junting Dong
# @Mail    : jtdong@zju.edu.cn
from termcolor import colored
import numpy as np
import torch

def compare(current: dict, original: dict):
    for key in current.keys():
        current_value = current[key]
        original_value = original[key]
        if isinstance(current_value, torch.Tensor):
            condition = current_value.equal(original_value)
        elif isinstance(current_value, dict):
            condition = (current_value == original_value)
        else:
            raise ValueError
        if not condition:
            import ipdb; ipdb.set_trace(context=11)
            assert condition , colored('key: {} is not equal!'.format(key), 'red')
    print(colored('unit test pass!', 'green'))

# Function for setting the seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
