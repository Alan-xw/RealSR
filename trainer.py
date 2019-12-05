import os
import math
from decimal import Decimal
import utility
import torch
import torch.nn.utils as utils
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        if args.demo:
            self.loader_demo = loader.loader_demo
        
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8


    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 2
        lr = self.optimizer.get_lr()
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _) in enumerate(self.loader_train):

            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            self.optimizer.zero_grad()
            sr = self.model(lr)
            loss = self.loss(sr, hr)
            loss.backward()
            self.optimizer.step()
            timer_model.hold()

            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )

            if (batch + 1) % self.args.print_every == 0:
                
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            
            timer_data.tic()
            
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        
#         # save the lastest model 
#         self.ckp.save_in_training(self, epoch, is_best=False)
                



    def test(self):
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()+1
        print('Test at epoch', epoch)
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), 3)
        )
        self.model.eval()
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for lr, hr, filename in tqdm(d, ncols=80):
                lr, hr = self.prepare(lr, hr)
                sr = self.model(lr)
                
                sr = utility.quantize(sr, self.args.rgb_range)
                
                save_list = [sr]
                # compute PSNR of test images
                self.ckp.log[-1, idx_data, self.args.scale-2] += utility.calc_psnr(
                        sr, hr, self.args.scale, self.args.rgb_range, if_benchmark=True)
                # whether to save gt or sr results
                if self.args.save_gt:
                    save_list.extend([lr, hr])
                if self.args.save_results:
                    self.ckp.save_results(d, filename[0], save_list, self.args.scale)
                    
            # compute average_PSNR        
            self.ckp.log[-1, idx_data, self.args.scale-2] /= len(d)
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    d.dataset.name,
                    self.args.scale,
                    self.ckp.log[-1, idx_data, self.args.scale-2],
                    best[0][idx_data, self.args.scale-2],
                    best[1][idx_data, self.args.scale-2] + 1
                )
            )


        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            #以Urban100的PSNR值为准，进行保存
            self.ckp.save(self, epoch, is_best=(best[1][-1, self.args.scale-2] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)
        
    def test_bench(self):
        torch.set_grad_enabled(False)
        print('Test the Benchmwark Image')
        self.model.eval()
        psnr_list = torch.zeros(1, len(self.loader_test), 3)
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for lr, hr, filename in tqdm(d, ncols=80):
                lr, hr = self.prepare(lr, hr)
                sr = self.model(lr)
                sr = utility.quantize(sr, self.args.rgb_range)
                save_list = [sr]
                
                psnr_list[-1, idx_data, self.args.scale-2] += utility.calc_psnr(
                    sr, hr, self.args.scale, self.args.rgb_range, if_benchmark=True)
                
                if self.args.save_gt:
                    save_list.extend([lr, hr])

                if self.args.save_results:
                    self.ckp.save_results(d, filename[0], save_list, self.args.scale)

            psnr_list[-1, idx_data, self.args.scale-2] /= len(d)
            print(
                '[{} x{}] \t PSNR: {:.3f})\n'.format(
                    d.dataset.name,
                    self.args.scale,
                    psnr_list[-1, idx_data, self.args.scale-2].numpy()))
        torch.set_grad_enabled(True)
    
    def test_demo(self):
        torch.set_grad_enabled(False)
        print('Test the Demo Image')
        if self.args.save_results:
            save_dir = os.path.join('.', 'experiment', self.args.save)
            sub_dir = os.path.join(save_dir,'results-{}'.format(self.args.demo_name))
            os.makedirs(sub_dir, exist_ok=True)
        self.model.eval()
        if self.args.save_results: self.ckp.begin_background()
        data_demo = self.loader_demo
        for lr, _, filename in tqdm(data_demo, ncols=80):
            lr,= self.prepare(lr)
            sr = self.model(lr)
            sr = utility.quantize(sr, self.args.rgb_range)
            save_list = [sr]
            self.ckp.save_results(data_demo, filename[0], save_list, self.args.scale)
        
        self.ckp.end_background()
        torch.set_grad_enabled(True)
            
            
    
    
    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.demo:
            self.test_demo()
            return True
        elif self.args.test_only:
            self.test_bench()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs