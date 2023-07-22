from torchvision import datasets
import torch

class Cifar10SearchDataset(datasets.CIFAR10):

    def __init__(self, root="~/data", train=True, download=True, transform=None):

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):

        image, label = self.data[index], self.targets[index]

        if self.transform is not None:

            transformed = self.transform(image=image)

            image = transformed["image"]

        return image, label


class data_loader:
    
    def __init__(self):
        self.batch_size = 512
        self.cuda = torch.cuda.is_available()

    def train_loader(self,train):

        dataloader_args = dict(shuffle=True, batch_size=self.batch_size, num_workers=0, pin_memory=True) if self.cuda == torch.cuda.is_available() else dict(shuffle=True, batch_size=64)
        # train dataloader
        return(torch.utils.data.DataLoader(train, **dataloader_args))
    
    def test_loader(self,test):
        dataloader_args = dict(shuffle=True, batch_size=self.batch_size, num_workers=0, pin_memory=True) if self.cuda == torch.cuda.is_available() else dict(shuffle=True, batch_size=64)
        # test dataloader
        return(torch.utils.data.DataLoader(test, **dataloader_args))


