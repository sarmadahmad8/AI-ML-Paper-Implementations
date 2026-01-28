from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms

img_transforms = transforms.Compose([transforms.Resize(size=(32, 32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])

train_dataset = CIFAR10(root= "../data/CIFAR10",
                         train= True,
                         transform= img_transforms,
                         download= True)

test_dataset = CIFAR10(root= "../data/CIFAR10",
                        train= False,
                        transform= img_transforms,
                        download= True)


train_dataloader = DataLoader(dataset= train_dataset,
                              batch_size= 96,
                              num_workers= 8,
                              shuffle= True)

test_dataloader = DataLoader(dataset= test_dataset,
                             batch_size= 96,
                             num_workers= 8,
                             shuffle= False)

len(train_dataloader), len(test_dataloader), len(train_dataset), len(test_dataset)
