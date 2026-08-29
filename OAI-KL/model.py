from torch import nn
from torchvision import models

def model_return(args):
    # --- CNN Architectures ---
    if args.model_type == 'resnet_101':
        model_ft = models.resnet101(weights='DEFAULT')
        in_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_ftrs, 5)
        
        for param in model_ft.parameters():
            param.requires_grad = False
        for param in model_ft.fc.parameters():
            param.requires_grad = True
                
    elif args.model_type == 'resnext_50_32x4d':
        model_ft = models.resnext50_32x4d(weights='DEFAULT')
        in_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_ftrs, 5)
        
        for param in model_ft.parameters():
            param.requires_grad = False
        for param in model_ft.fc.parameters():
            param.requires_grad = True
        
    elif args.model_type == 'wide_resnet_50_2':
        model_ft = models.wide_resnet50_2(weights='DEFAULT')
        in_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_ftrs, 5)
        
        for param in model_ft.parameters():
            param.requires_grad = False
        for param in model_ft.fc.parameters():
            param.requires_grad = True
        
    elif args.model_type == 'densenet_161':
        model_ft = models.densenet161(weights='DEFAULT')
        in_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(in_ftrs, 5)
        
        for param in model_ft.parameters():
            param.requires_grad = False
        for param in model_ft.classifier.parameters():
            param.requires_grad = True
        
    elif args.model_type == 'efficientnet_b5':
        model_ft = models.efficientnet_b5(weights='DEFAULT')
        in_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(in_ftrs, 5)
        
        for param in model_ft.parameters():
            param.requires_grad = False
        for param in model_ft.classifier[1].parameters():
            param.requires_grad = True
        
    elif args.model_type == 'efficientnet_v2_s':
        model_ft = models.efficientnet_v2_s(weights='DEFAULT')
        in_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(in_ftrs, 5)
        
        for param in model_ft.parameters():
            param.requires_grad = False
        for param in model_ft.classifier[1].parameters():
            param.requires_grad = True

    elif args.model_type == 'regnet_y_8gf':
        model_ft = models.regnet_y_8gf(weights='DEFAULT')
        in_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_ftrs, 5)
        
        for param in model_ft.parameters():
            param.requires_grad = False
        for param in model_ft.fc.parameters():
            param.requires_grad = True
        
    elif args.model_type == 'shufflenet_v2_x2_0':
        model_ft = models.shufflenet_v2_x2_0(weights='DEFAULT')
        in_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_ftrs, 5)
        
        for param in model_ft.parameters():
            param.requires_grad = False
        for param in model_ft.fc.parameters():
            param.requires_grad = True

    # --- Vision Transformer Architectures ---
    elif args.model_type == 'vit_b_16':
        model_ft = models.vit_b_16(weights='DEFAULT')
        in_ftrs = model_ft.heads.head.in_features
        model_ft.heads.head = nn.Linear(in_ftrs, 5)
        
        for param in model_ft.parameters():
            param.requires_grad = False
        for param in model_ft.heads.head.parameters():
            param.requires_grad = True

    elif args.model_type == 'swin_s':
        model_ft = models.swin_s(weights='DEFAULT')
        in_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(in_ftrs, 5)
        
        for param in model_ft.parameters():
            param.requires_grad = False
        for param in model_ft.head.parameters():
            param.requires_grad = True

    elif args.model_type == 'swin_v2_s':
        model_ft = models.swin_v2_s(weights='DEFAULT')
        in_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(in_ftrs, 5)
        
        for param in model_ft.parameters():
            param.requires_grad = False
        for param in model_ft.head.parameters():
            param.requires_grad = True
        
    return model_ft