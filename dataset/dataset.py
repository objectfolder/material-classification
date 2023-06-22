# Datasets of material classification
# Yiming Dou (yimingdou@cs.stanford.edu)
# July 2022

import os
import os.path as osp
import json
from tqdm import tqdm
import random

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader

np.random.seed(42)

class material_classification_dataset(object):
    def __init__(self, args, set_type='train'):
        self.args = args
        self.use_vision = 'vision' in args.modality_list
        self.use_touch = 'touch' in args.modality_list
        self.use_audio = 'audio' in args.modality_list
        self.set_type = set_type  # 'train' or 'val' or 'test'
        self.visual_images_location = osp.join(self.args.data_location, 'vision')
        self.tactile_images_location = osp.join(self.args.data_location, 'touch')
        self.audio_spectrogram_location = osp.join(self.args.data_location, 'audio_examples')
        self.label_location = osp.join(self.args.data_location, 'label.json')
        
        self.preprocess = {
            'vision': T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ]),
            'touch': T.Compose([
                T.CenterCrop(320),
                T.Resize((224, 224)),
                T.ToTensor(),
            ]),
        }
        
        with open(self.label_location) as f:
            self.label_dict = json.load(f)

        with open(self.args.split_location) as f:
            self.cand = json.load(f)[self.set_type]  # [[obj, inst],]
            
    def __len__(self):
        return len(self.cand)
    
    # load the #instance visual RGB image of obj
    def load_visual_image(self, obj, instance):
        visual_image = Image.open(
            osp.join(self.visual_images_location, obj, f'{instance}.png')
        ).convert('RGB')
        visual_image = self.preprocess['vision'](visual_image)
        return torch.FloatTensor(visual_image)
    
    # load the #instance tactile image of obj
    def load_tactile_image(self, obj, instance):
        tactile_image = Image.open(
            osp.join(self.tactile_images_location, obj, f'{instance}.png')
        ).convert('RGB')
        tactile_image = self.preprocess['touch'](tactile_image)
        return torch.FloatTensor(tactile_image)
    
    def load_audio_spectrogram(self, obj, instance):
        audio_spectrogram_path = osp.join(self.audio_spectrogram_location, obj, f'{instance}.npy')
        audio_spectrogram = torch.tensor(np.load(audio_spectrogram_path)).float()
        return torch.FloatTensor(audio_spectrogram)
    
    def __getitem__(self, index):
        obj, instance = self.cand[index]
        data = {}
        data['names'] = (obj, instance)
        data['label'] = int(self.label_dict[obj])
        if self.use_vision:
            data['visual_image'] = self.load_visual_image(obj, instance)
        if self.use_touch:
            data['tactile_image'] = self.load_tactile_image(obj, instance)
        if self.use_audio:
            data['audio_spectrogram'] = self.load_audio_spectrogram(obj, instance)
        
        return data
    
    def collate(self, data):
        batch = {}
        batch['names'] = [item['names'] for item in data]
        batch['label'] = torch.tensor([item['label'] for item in data])
        if self.use_vision:
            batch['visual_image'] = torch.stack([item['visual_image'] for item in data])
        if self.use_touch:
            batch['tactile_image'] = torch.stack([item['tactile_image'] for item in data])
        if self.use_audio:
            batch['audio_spectrogram'] = torch.stack([item['audio_spectrogram'].unsqueeze(0) for item in data])
        
        return batch
            