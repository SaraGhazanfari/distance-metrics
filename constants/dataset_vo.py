class DatasetConstants:

    def __init__(self, dataset_name):
        self.PER_CLASS_TRAIN_SIZE, self.PER_CLASS_TEST_SIZE = None, None
        if dataset_name in ['cifar-10', 'cifar-100']:
            self.CLASS_NUMBER = 10
            self.TRAIN_CLASS_SIZE = int(50_000 / self.CLASS_NUMBER)
            self.TEST_CLASS_SIZE = int(10_000 / self.CLASS_NUMBER)

        elif dataset_name == 'svhn':
            self.CLASS_NUMBER = 10
            self.TRAIN_CLASS_SIZE = 73_257
            self.TEST_CLASS_SIZE = 26_032
            self.PER_CLASS_TRAIN_SIZE = {0: 4948, 1: 13861, 2: 10585, 3: 8497, 4: 7458, 5: 6882, 6: 5727,
                                         7: 5595, 8: 5045, 9: 4659}
            self.PER_CLASS_TEST_SIZE = {}  # todo

        elif dataset_name == 'imagenet-100':
            self.CLASS_NUMBER = 100
            self.TRAIN_CLASS_SIZE = 130_000
            self.TEST_CLASS_SIZE = 5_000

        if not self.PER_CLASS_TRAIN_SIZE:
            self.PER_CLASS_TRAIN_SIZE = {class_number: self.TRAIN_CLASS_SIZE for class_number in
                                         range(self.CLASS_NUMBER)}
            self.PER_CLASS_TEST_SIZE = {class_number: self.TEST_CLASS_SIZE for class_number in
                                        range(self.CLASS_NUMBER)}

    def get_per_class_size(self, split):
        if split == 'train':
            return self.PER_CLASS_TRAIN_SIZE
        return self.PER_CLASS_TEST_SIZE
