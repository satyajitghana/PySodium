# PySodium V0.0.2

![release](https://img.shields.io/github/v/release/satyajitghana/PySodium?include_prereleases)
![PyPI-Python-version](https://img.shields.io/pypi/pyversions/py-sodium)
![PyPI-wheel](https://img.shields.io/pypi/wheel/py-sodium)
![release-date](https://img.shields.io/github/release-date-pre/satyajitghana/PySodium)
![license](https://img.shields.io/github/license/satyajitghana/pysodium)
![maintainer](https://img.shields.io/badge/maintainer-shadowleaf-blue)

## Usage

1. Install the PySodium Library

`pip install git+https://github.com/satyajitghana/PySodium.git#egg=sodium`

2. Create a config.yml

```yaml
name: CIFAR10_V2
save_dir: saved/
seed: 1
target_device: 0

arch:
    type: CIFAR10S8Model
    args: {}

augmentation:
    type: CIFAR10Albumentations
    args: {}

data_loader:
    type: CIFAR10DataLoader
    args:
        batch_size: 128
        data_dir: data/
        nworkers: 4
        shuffle: True

criterion: cross_entropy_loss

lr_scheduler:
    type: ReduceLROnPlateau
    args:
        mode: 'min'
        batch_scheduler: False

optimizer:
    type: SGD
    args:
        lr: 0.001
        momentum: 0.95
        weight_decay: 0.0005

training:
    epochs: 10
```

3. Run the Model !

```python
# import my baby-library
from sodium.utils import load_config
import sodium.runner as runner

# create a runner
config = load_config('config.yml', tsai_mode=True)

# setup trainer
runner.setup_train(tsai_mode=True)

# find best lr
runner.find_lr()

# train the network using the best lr
runner.train(use_bestlr=True)

# plot metrics
runner.plot_metrics()

# plot grad cam
target_layers = ["layer1", "layer2", "layer3", "layer4"]
runner.plot_gradcam(target_layers=target_layers)

# plot misclassifications
runner.plot_misclassifications(target_layers=target_layers)
```

## NOTE

if you are using the library on a terminal, you can use the main.py and call

`python main.py --config=config.yml`


---

Made with ❤ by shadowleaf.satyajit
