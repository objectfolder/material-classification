from email.policy import strict
import torch.optim as optim
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def build(args, cfg):
    print("Building model: {}".format(args.model))
    if args.model == 'SpectroVision':
        from SpectroVision import SpectroVision
        model = SpectroVision.SpectroVision(args, cfg)
        if cfg.get('vision_backbone_checkpoint', False):
            print(
                f"loading vision_backbone_checkpoint from {cfg.get('vision_backbone_checkpoint')}")
            state_dict = torch.load(cfg.get('vision_backbone_checkpoint'),map_location='cpu')
            model.vision_backbone.load_state_dict(state_dict)
        if cfg.get('audio_backbone_checkpoint', False):
            print(
                f"loading audio_backbone_checkpoint from {cfg.get('audio_backbone_checkpoint')}")
            state_dict = torch.load(cfg.get('audio_backbone_checkpoint'),map_location='cpu')
            model.audio_backbone.load_state_dict(state_dict)
        if cfg.get('touch_backbone_checkpoint', False):
            state_dict = torch.load(cfg.get('touch_backbone_checkpoint'),map_location='cpu')
            print(
                f"loading touch_backbone_checkpoint from {cfg.get('touch_backbone_checkpoint')}")
            model.touch_backbone.load_state_dict(state_dict)
        if cfg.get('vision_mlp_checkpoint', False):
            print(
                f"loading vision_mlp_checkpoint from {cfg.get('vision_mlp_checkpoint')}")
            state_dict = torch.load(cfg.get('vision_mlp_checkpoint'),map_location='cpu')
            model.vision_mlp.load_state_dict(state_dict)
        if cfg.get('audio_mlp_checkpoint', False):
            print(
                f"loading audio_mlp_checkpoint from {cfg.get('audio_mlp_checkpoint')}")
            state_dict = torch.load(cfg.get('audio_mlp_checkpoint'),map_location='cpu')
            model.audio_mlp.load_state_dict(state_dict)
        if cfg.get('touch_mlp_checkpoint', False):
            print(
                f"loading touch_mlp_checkpoint from {cfg.get('touch_mlp_checkpoint')}")
            state_dict = torch.load(cfg.get('touch_mlp_checkpoint'),map_location='cpu')
            model.touch_mlp.load_state_dict(state_dict)
        if cfg.get('classifier_checkpoint', False):
            print(
                f"loading classifier_checkpoint from {cfg.get('classifier_checkpoint')}")
            state_dict = torch.load(cfg.get('classifier_checkpoint'),map_location='cpu')
            model.classifier.load_state_dict(state_dict)
        if cfg.get('all_checkpoint', False):
            print(f"loading all_checkpoint from {cfg.get('all_checkpoint')}")
            state_dict = torch.load(cfg.get('all_checkpoint'),map_location='cpu')
            model.load_state_dict(state_dict, strict=True)

        optim_params = []
        if not args.finetune:
            if model.use_vision:
                optim_params.append({'params': model.vision_backbone.parameters(
                ), 'lr': args.lr*1e-2, 'weight_decay': args.weight_decay*1e-2})
                optim_params.append({'params': model.vision_mlp.parameters(
                ), 'lr': args.lr, 'weight_decay': args.weight_decay})
            if model.use_touch:
                optim_params.append({'params': model.touch_backbone.parameters(
                ), 'lr': args.lr, 'weight_decay': args.weight_decay})
                optim_params.append({'params': model.touch_mlp.parameters(
                ), 'lr': args.lr, 'weight_decay': args.weight_decay})
            if model.use_audio:
                optim_params.append({'params': model.audio_backbone.parameters(
                ), 'lr': args.lr*1e-2, 'weight_decay': args.weight_decay*1e-2})
                optim_params.append({'params': model.audio_mlp.parameters(
                ), 'lr': args.lr, 'weight_decay': args.weight_decay})
            optim_params.append({'params': model.classifier.parameters(
            ), 'lr': args.lr, 'weight_decay': args.weight_decay})
        else:
            if model.use_vision:
                optim_params.append({'params': model.vision_backbone.parameters(
                ), 'lr': args.lr*1e-4, 'weight_decay': args.weight_decay*1e-4})
                optim_params.append({'params': model.vision_mlp.parameters(
                ), 'lr': args.lr*1e-4, 'weight_decay': args.weight_decay*1e-4})
            if model.use_touch:
                optim_params.append({'params': model.touch_backbone.parameters(
                ), 'lr': args.lr*1e-4, 'weight_decay': args.weight_decay*1e-4})
                optim_params.append({'params': model.touch_mlp.parameters(
                ), 'lr': args.lr*1e-4, 'weight_decay': args.weight_decay*1e-4})
            if model.use_audio:
                optim_params.append({'params': model.audio_backbone.parameters(
                ), 'lr': args.lr*1e-4, 'weight_decay': args.weight_decay*1e-4})
                optim_params.append({'params': model.audio_mlp.parameters(
                ), 'lr': args.lr*1e-4, 'weight_decay': args.weight_decay*1e-4})
            optim_params.append({'params': model.classifier.parameters(
            ), 'lr': args.lr, 'weight_decay': args.weight_decay})

        optimizer = optim.AdamW(optim_params)
        
    elif args.model == 'FENet':
        from FENet import FENet
        model = FENet.FENet(args, cfg)
        if cfg.get('vision_checkpoint', False):
            print(f"loading vision_checkpoint from {cfg.get('vision_checkpoint')}")
            state_dict = torch.load(cfg.get('vision_checkpoint'),map_location='cpu')
            state_dict = {k:v for k,v in state_dict.items() if 'vision' in k}
            model.load_state_dict(state_dict, strict=False)
        if cfg.get('touch_checkpoint', False):
            print(f"loading touch_checkpoint from {cfg.get('touch_checkpoint')}")
            state_dict = torch.load(cfg.get('touch_checkpoint'),map_location='cpu')
            state_dict = {k:v for k,v in state_dict.items() if 'touch' in k}
            model.load_state_dict(state_dict, strict=False)
        if cfg.get('audio_checkpoint', False):
            print(f"loading audio_checkpoint from {cfg.get('audio_checkpoint')}")
            state_dict = torch.load(cfg.get('audio_checkpoint'),map_location='cpu')
            state_dict = {k:v for k,v in state_dict.items() if 'audio' in k}
            model.load_state_dict(state_dict, strict=False)
        if cfg.get('all_checkpoint', False):
            print(f"loading all_checkpoint from {cfg.get('all_checkpoint')}")
            state_dict = torch.load(cfg.get('all_checkpoint'),map_location='cpu')
            model.load_state_dict(state_dict, strict=True)
            
        optim_params = []
        if model.use_vision:
            optim_params.append({'params': model.vision_backbone.parameters(), 'lr': args.lr*1e-2, 'weight_decay': args.weight_decay*1e-1})
            # optim_params.append({'params': model.vision_backbone.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay*1e-1})
            optim_params.append({'params': model.pool_vision.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
            optim_params.append({'params': model.UP_vision.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
            optim_params.append({'params': model.conv_before_mfs_vision.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
            optim_params.append({'params': model.mfs_vision.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
            optim_params.append({'params': model.fc_vision.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
            optim_params.append({'params': model.mlp_vision.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
        if model.use_touch:
            optim_params.append({'params': model.touch_backbone.parameters(), 'lr': args.lr*1e-2, 'weight_decay': args.weight_decay})
            # optim_params.append({'params': model.touch_backbone.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
            optim_params.append({'params': model.mlp_touch.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
        if model.use_audio:
            optim_params.append({'params': model.audio_backbone.parameters(), 'lr': args.lr*1e-2, 'weight_decay': args.weight_decay*1e-1})
            # optim_params.append({'params': model.audio_backbone.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay*1e-1})
            optim_params.append({'params': model.mlp_audio.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
        if args.finetune:
            optim_params.append({'params': model.classifier.parameters(), 'lr': args.lr*10, 'weight_decay': args.weight_decay*10})
        else:
            optim_params.append({'params': model.classifier.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
        optimizer = optim.AdamW(optim_params)
        
    return model, optimizer
