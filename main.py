import torch
from config import args
import data
import model
import loss
import utility
from trainer import Trainer
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model

    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        _loss = loss.Loss(args, checkpoint) if not args.test_only and not args.demo else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()
            
        checkpoint.done()

if __name__ == '__main__':
    main()
    