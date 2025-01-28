"""
Eval utils.
"""
from datetime import datetime
import datasets
import models as m
import os
import torch
import argparse
import configs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALL_MODELS = [
    'slbr', 
    'denet-g', 
    'denet-l', 
    'denet-h', 
    'wdnet', 
    'splitnet', 
    'slbr-ft',
    'refine',
    'morphomod_slbr',
    'morphomod_denet-g',
    'morphomod_denet-l',
    'morphomod_denet-h',
    'morphomod_wdnet',
    'morphomod_splitnet',
]

ALL_DATASETS = ['lvw', 'clwd', 'logo-g', 'logo-l', 'logo-h']


def pprint_args(args: argparse.Namespace):
    """Pretty print args for an experiment"""
    print("\n\n================================")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    print("================================\n\n")

# G.load_state_dict(torch.load('/Users/probinette/Documents/PROJECTS/hydra_steg/models/wdnet_g.pkl', map_location=torch.device('cpu')))

def load_checkpoint(model, checkpoint: str):
    """Load Checkpoint
    
    Args:
        model: the model to load checkpoint.
        checkpoint: path of the checkpoint to load.
    """
    if not os.path.exists(checkpoint):
        raise Exception("=> no checkpoint found at '{}'".format(checkpoint))

    current_checkpoint = torch.load(checkpoint, map_location=DEVICE, weights_only=True)

    # ---------------- Load Model Weights --------------------------------------
    try:
        model.load_state_dict(current_checkpoint['state_dict'], strict=True)
        print("normal load")
    except:
        if 'state_dict' in current_checkpoint:
            new_state_dict = {key.replace("module.", ""): value for key, value in current_checkpoint['state_dict'].items()}
            model.load_state_dict(new_state_dict, strict=True)
        else:
            model.load_state_dict(current_checkpoint, strict=True)
        print("alt load")
        
    print("=> loaded checkpoint '{}'".format(checkpoint))
    return model


def get_save_name(args):
    """Get save name with datetime"""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S.pkl")  # Format: YYYYMMDD_HHMMSS
    save_name = f"{args.dataset}_{current_time}"
    res = os.path.join(args.save_path, save_name)
    os.makedirs(args.save_path, exist_ok=True)
    
    return res


def load_model(model_type: str, checkpoint: str = None, refine_checkpoint: str = None, inpaint: str = 'SD2'):
    """
    Load a model from a checkpoint.
    """
    checkpoint_path = None
    if model_type == "morphomod_slbr":
        mask_model = load_model("slbr_mask", checkpoint)
        refine_model = load_model("refine", refine_checkpoint)
        model = m.MorphoModel(mask_model, refine_model, inpaint)
        return model
    elif model_type == "morphomod_splitnet":
        mask_model = load_model("splitnet_mask", checkpoint)
        refine_model = load_model("refine", refine_checkpoint)
        model = m.MorphoModel(mask_model, refine_model, inpaint)
        return model
    elif model_type == "morphomod_denet-g":
        mask_model = load_model("denet-g_mask", checkpoint)
        refine_model = load_model("refine", refine_checkpoint)
        model = m.MorphoModel(mask_model, refine_model, inpaint)
        return model
    elif model_type == "morphomod_denet-l":
        mask_model = load_model("denet-l_mask", checkpoint)
        refine_model = load_model("refine", refine_checkpoint)
        model = m.MorphoModel(mask_model, refine_model, inpaint)
        return model
    elif model_type == "morphomod_denet-h":
        mask_model = load_model("denet-h_mask", checkpoint)
        refine_model = load_model("refine", refine_checkpoint)
        model = m.MorphoModel(mask_model, refine_model, inpaint)
        return model
    elif model_type == "morphomod_wdnet":
        mask_model = load_model("wdnet_mask", checkpoint)
        refine_model = load_model("refine", refine_checkpoint)
        model = m.MorphoModel(mask_model, refine_model, inpaint)
        return model
    # 
    # Single Models MASK
    #
    elif model_type == "slbr_mask":
        model_args = configs.get_default_slbr_config()
        model = m.slbr_nets.__dict__['slbr_prime_mask'](args=model_args)
        checkpoint_path = model_args.resume if checkpoint == None else checkpoint
    elif model_type == "denet-g_mask":
        print(f"==> Loading {model_type}")
        model_args = configs.get_default_denet_config()
        model = m.denet_nets.__dict__['slbr_mask'](args=model_args)
        checkpoint_path = osp.join(model_args.resume, "denet_logo-gray.pth.tar") if checkpoint == None else checkpoint
    elif model_type == "denet-l_mask":
        print(f"==> Loading {model_type}")
        model_args = configs.get_default_denet_config()
        model = m.denet_nets.__dict__['slbr_mask'](args=model_args)
        checkpoint_path = osp.join(model_args.resume, "denet_logo-l.pth.tar") if checkpoint == None else checkpoint
    elif model_type == "denet-h_mask":
        print(f"==> Loading {model_type}")
        model_args = configs.get_default_denet_config()
        model = m.denet_nets.__dict__['slbr_mask'](args=model_args)
        checkpoint_path = osp.join(model_args.resume, "denet_logo-h.pth.tar") if checkpoint == None else checkpoint
    elif model_type == "splitnet_mask":
        print(f"==> Loading {model_type}")
        model = m.splitnet_nets.__dict__['splitnet_mask']()
        checkpoint_path = osp.join(model_args.resume, "splitnet.pth.tar") if checkpoint == None else checkpoint
    elif model_type == "refine":
        print(f"==> Loading {model_type}")
        model = m.UNetRefineSemseg()
        checkpoint_path = checkpoint
    elif model_type == "wdnet_mask":
        print(f"==> Loading {model_type}")
        model = m.generator_mask(3, 3)
        checkpoint_path = checkpoint
    # 
    # Single Models "OG"
    #
    elif model_type == "slbr":
        print(f"==> Loading {model_type}")
        model_args = configs.get_default_slbr_config()
        model = m.slbr_nets.__dict__['slbr'](args=model_args)
        checkpoint_path = model_args.resume if checkpoint == None else checkpoint
    elif model_type == "denet-g":
        print(f"==> Loading {model_type}")
        model_args = configs.get_default_denet_config()
        model = m.denet_nets.__dict__['slbr'](args=model_args)
        checkpoint_path = osp.join(model_args.resume, "denet_logo-gray.pth.tar") if checkpoint == None else checkpoint
    elif model_type == "denet-l":
        print(f"==> Loading {model_type}")
        model_args = configs.get_default_denet_config()
        model = m.denet_nets.__dict__['slbr'](args=model_args)
        checkpoint_path = osp.join(model_args.resume, "denet_logo-l.pth.tar") if checkpoint == None else checkpoint
    elif model_type == "denet-h":
        print(f"==> Loading {model_type}")
        model_args = configs.get_default_denet_config()
        model = m.denet_nets.__dict__['slbr'](args=model_args)
        checkpoint_path = osp.join(model_args.resume, "denet_logo-h.pth.tar") if checkpoint == None else checkpoint
    elif model_type == "splitnet":
        print(f"==> Loading {model_type}")
        model = m.splitnet_nets.__dict__['vvv4n']()
        checkpoint_path = osp.join(model_args.resume, "splitnet.pth.tar") if checkpoint == None else checkpoint
    elif model_type == "wdnet":
        print(f"==> Loading {model_type}")
        model = m.generator(3, 3)
        checkpoint_path = checkpoint
    else:
        raise ValueError("Not implemented")

    model = load_checkpoint(model, checkpoint_path)
    return model
    

def get_models(args):
    """ Get the models to be evaluated."""
    get_m_path = lambda x: os.path.join(args.model_path, x)
    models = {}
    #
    # Update model names
    #
    names = args.model if args.model else ALL_MODELS
    for name in names:
        assert name in ALL_MODELS
    #
    # Load models
    #
    try:
        for model_name in names:
            if model_name == "slbr":
                models[model_name] = m.load_slbr(
                    checkpoint=get_m_path('SLBR.pth.tar')
                )
            elif model_name == "denet-g":
                models[model_name] = m.load_denet(
                    mode='gray', 
                    checkpoint=get_m_path('denet_logo-gray.pth.tar')
                )
            elif model_name == "denet-l":
                models[model_name] = m.load_denet(
                    mode='l', 
                    checkpoint=get_m_path('denet_logo-l.pth.tar')
                )
            elif model_name == "denet-h":
                models[model_name] = m.load_denet(
                    mode='h',
                    checkpoint=get_m_path('denet_logo-h.pth.tar')
                )
            elif model_name == "wdnet":
                models[model_name] = m.load_wdnet(
                    checkpoint=get_m_path('wdnet_g.pkl')
                )
            elif model_name == "splitnet":
                models[model_name] = m.load_splitnet(
                    checkpoint=get_m_path('splitnet.pth.tar')
                )
            elif model_name == "slbr-ft":
                models[model_name] = m.load_slbr(
                    checkpoint=get_m_path('slbr_prime_ft.pth.tar')
                )
            else:
                raise ValueError(f"{model_name} does not exist.")

    except:
        raise ValueError(f"{model_name} was not loaded.")

    return models


def load_data(args):
    """Load the dataset"""
    assert args.dataset in ALL_DATASETS, f"{args.dataset} not allowed. Please update"
    if args.dataset == "lvw":
        # TODO: update this
        dataloader = None
    elif args.dataset == "clwd":
        dataloader = datasets.load_clwd(
            batch_size=args.batch_size,
            path=args.data_path
        )
    elif args.dataset == 'logo-g':
        dataloader = datasets.load_10k(
            mode='gray',
            batch_size=args.batch_size,
            path=args.data_path,
        )
    elif args.dataset == 'logo-l':
        dataloader = datasets.load_10k(
            mode='mid',
            batch_size=args.batch_size,
            path=args.data_path
        )
    elif args.dataset == 'logo-h':
        dataloader = datasets.load_10k(
            mode='high',
            batch_size=args.batch_size,
            path=args.data_path
        )
    else:
        raise ValueError(f"{args.dataset} is not valid.")

    return dataloader