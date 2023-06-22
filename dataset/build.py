import os, sys
from torch.utils.data import DataLoader
from .dataset import material_classification_dataset


def build(args):
    train_dataset = material_classification_dataset(args, 'train')
    val_dataset = material_classification_dataset(args, 'val')
    test_dataset = material_classification_dataset(args, 'test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=train_dataset.collate, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=val_dataset.collate, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=test_dataset.collate, drop_last=False)
    print("Dataset Loaded, train: {}, val: {}, test: {}".format(len(train_dataset)//args.batch_size,len(val_dataset)//args.batch_size,len(test_dataset)//args.batch_size))
    return train_loader, val_loader, test_loader