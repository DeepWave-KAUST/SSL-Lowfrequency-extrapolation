import  torch.utils.data as data
import  os
import torch
import scipy.io as sio
import numpy as np

class Basicdataset(data.Dataset):
    '''
    The items are (datapath).
    Args:
    - dir: the directory where the dataset will be stored
    '''

    def __init__(self, dir):
        self.dir = dir

        self.ids = [os.path.splitext(file)[0] for file in os.listdir(dir) 
                if not file.startswith('.')]

    def __getitem__(self, index):
        idx_file = self.ids[index]

        file = os.path.join(self.dir, idx_file)

        dict = sio.loadmat(file)
        raw = dict['label']

        return {'raw': torch.from_numpy(raw).unsqueeze(0).type(torch.FloatTensor)}

    def __len__(self):
        return len(self.ids)

class CUDAPrefetcher():
    """CUDA prefetcher.
    Ref:
    https://github.com/NVIDIA/apex/issues/304#
    It may consums more GPU memory.
    Args:
        loader: Dataloader.
        opt (dict): Options.
    """

    def __init__(self, loader, opt=None):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device('cuda')
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return None
        # put tensors to gpu
        with torch.cuda.stream(self.stream):
            if type(self.batch) == dict:
                for k, v in self.batch.items():
                    if torch.is_tensor(v):
                        self.batch[k] = self.batch[k].to(
                            device=self.device, non_blocking=True)
            elif type(self.batch) == list:
                for k in range(len(self.batch)):
                    if torch.is_tensor(self.batch[k]):
                        self.batch[k] = self.batch[k].to(
                            device=self.device, non_blocking=True)
            else:
                assert NotImplementedError

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()