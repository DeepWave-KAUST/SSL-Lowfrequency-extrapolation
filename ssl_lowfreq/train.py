import  torch, os
import torch.nn.functional as F
import  numpy as np
import  argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from  trainer import Base_trainer, Base_fastIDR
import random
from dataset import Basicdataset, CUDAPrefetcher
from torch.utils.data import DataLoader
import yaml

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    dataset = Basicdataset(args.dir_train)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.num_workers, pin_memory=True)

    prefetcher = CUDAPrefetcher(train_loader)

    print(f'''Starting training:
        Data type:            {args.data_type}
        Total Epoch:          {args.total_epoch}
        Warmup Epoch:         {args.warmup_epoch}
        Batch size:           {args.batch_size}
        Learning rate:        {args.lr}
        Loss type:            {args.loss_type}
        Optimizer type        {args.optimizer}
        Schedule type         {args.schedule}
    ''')

    writer = SummaryWriter(comment=f'LR_{args.lr}_BS_{args.batch_size}_Epoch_{args.total_epoch}')

    
    model = Base_fastIDR(args) if args.train_mode == 'SSL' else Base_trainer(args)

    print('------------ Training start ------------')

    while True:
        prefetcher.reset()

        data = prefetcher.next()

        epoch_loss = 0
        batch_train_idx = 0

        while data is not None:
            batch_train_idx = batch_train_idx + 1

            model.preprocess(data)

            loss = model.optimize_parameters()

            writer.add_scalar('Loss/loss_iter', loss.item(), model.cur_iter)

            data = prefetcher.next()

        epoch_loss = epoch_loss/batch_train_idx
        writer.add_scalar('Loss/loss_epoch', epoch_loss, model.epoch)

        model.epoch += 1
        model.update_lr_scheme()

        model.save_model()

        if model.epoch >= args.total_epoch:
            print(f'End training, Save the latest epoch model, total iterations {args.total_iter}')
            exit(0)

    writer.close()

    print('------------ Training ending ------------')

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

def load_config(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(config_dict)


if __name__ == '__main__':

    args = load_config('config.yaml')
    main(args)
