from importlib import import_module
from torch.utils.data import Dataset, DataLoader

class Data:
    def __init__(self, args):
        pin_memory = False
        if args.cuda:
            pin_memory = True

        self.loader_train = None
        
        if not args.test:
            module_train = import_module('data.div2k')
            train_set = getattr(module_train, 'DIV2K')(args, train=True)
            self.loader_train = DataLoader(
                dataset=train_set, batch_size=args.train_batch, shuffle=True,
                num_workers=args.threads, pin_memory=pin_memory
            )

        if args.data_test == 'DIV2K':
            module_test = import_module('data.div2k')
            test_set = getattr(module_test, 'DIV2K')(args, train=False)

        elif args.data_test in['Set5', 'Set14', 'B100']:
            module_test = import_module('data.testset')
            test_set = getattr(module_test, 'Testset')(args)
        
        # test without HR image
        else:
            module_test = import_module('data.' + args.data_test.lower())
            test_set = getattr(module_test, args.data_test)(args)

        self.loader_test = DataLoader(
                dataset=test_set, batch_size=1, shuffle=False,
                num_workers=args.threads, pin_memory=pin_memory
            )



