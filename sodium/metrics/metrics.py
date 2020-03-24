class Metrics:
    def __init__(self):
        self.train_metric = {'train_loss': [], 'train_accuracy': []}
        self.test_metric = {'test_loss': [], 'test_accuracy': []}
        self.lr_metric = []
