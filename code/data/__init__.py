from importlib import import_module
from torch.utils.data import Dataset, DataLoader

class Data:
    def __init__(self, args):
        self.loader_train = None

        if not args.test:
            module_train = import_module('data.benchmark')
            train_set = getattr(module_train, 'Benchmark')(args)
            self.loader_train = DataLoader(
                dataset=train_set, 
                num_workers=args.threads, 
                batch_size=args.train_batch, 
                shuffle=True
            )
        
        if args.data_test in['Set5', 'Set14', 'B100']:
            module_test = import_module('data.benchmark')
            test_set = getattr(module_test, 'Benchmark')(args)
        
        # test without HR image
        else:
            module_test = import_module('data.' + args.data_test.lower())
            test_set = getattr(module_test, args.data_test)(args)

        self.loader_test = DataLoader(
                dataset=test_set, 
                num_workers=args.threads, 
                batch_size=1, 
                shuffle=False
            )



