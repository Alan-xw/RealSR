from .SRdataset import SRdatasets
import torch.utils.data.dataloader as DataLoader
from torch.utils.data import ConcatDataset
from .demo import Demo

class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)


class Data:
    def __init__(self, args):
        self.loader_train = None
        
        if not args.test_only and not args.demo:
            data_train = SRdatasets(args, name=args.data_name, train=True)
            self.loader_train = DataLoader.DataLoader(
                                data_train,
                                batch_size=args.batch_size,
                                num_workers=8,
                                shuffle=True)
        
        if args.demo:
            data_demo = Demo(args,name=args.demo_name,train=False)
            self.loader_demo = DataLoader.DataLoader(
                                data_demo,
                                batch_size=1,
                                shuffle=False)
        
        
        self.loader_test = []
        # Only Test one dataset
        if not args.demo:
            for test_data in args.test_name:
                data_test = SRdatasets(args, name=test_data, train=False)
                self.loader_test.append(DataLoader.DataLoader(
                    data_test,
                    batch_size=1,
                    num_workers=4,
                    shuffle=False))

        

        

            