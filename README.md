# PySodium V0.0.1

## Usage

1. Install the PySodium Library

`pip install git+https://github.com/satyajitghana/PySodium.git#egg=sodium`

2. Create a config.yml

```yaml
name: CIFAR10_MyNet
save_dir: saved/
seed: 1
target_device: 0

arch:
    type: CIFAR10Model
    args: {}

augmentation:
    type: CIFAR10Transforms
    args: {}

data_loader:
    type: CIFAR10DataLoader
    args:
        batch_size: 64
        data_dir: data/
        nworkers: 4
        shuffle: True

loss: nll_loss

# lr_scheduler:
#     type: OneCycleLR
#     args:
#         gamma: 0.1
#         step_size: 6

optimizer:
    type: SGD
    args:
        lr: 0.008
        momentum: 0.95

training:
    epochs: 50
```

3. Run the Model !

```python
from sodium.utils import load_config
import sodium.runner as runner

config = load_config('config.yml')
runner.train(config)
```
