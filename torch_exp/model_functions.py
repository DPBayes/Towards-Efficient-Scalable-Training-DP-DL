import timm
from opacus.validators import ModuleValidator
import models
import numpy as np
import os


def print_param_shapes(model, prefix=""):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{prefix}{name}: {param.shape}")

    for name, module in model.named_children():
        print(f"{prefix}{name}:")
        print_param_shapes(module, prefix + "  ")


def prepare_vision_model(model, model_name):

    pre_total, pre_train = count_params(model)

    print("Preparing vision model pre total parameters {} pre trained parameters {}".format(pre_total, pre_train))

    if "xcit" in model_name:
        for name, param in model.named_parameters():
            if "gamma" in name or "attn.temperature" in name:
                param.requires_grad = False

    if "cait" in model_name:
        for name, param in model.named_parameters():
            if "gamma_" in name:
                param.requires_grad = False

    if "convnext" in model_name:
        for name, param in model.named_parameters():
            if ".gamma" in name or "head.norm." in name or "downsample.0" in name or "stem.1" in name:
                param.requires_grad = False

    if "convit" in model_name:
        for name, param in model.named_parameters():
            if "attn.gating_param" in name:
                param.requires_grad = False

    if "beit" in model_name:
        for name, param in model.named_parameters():
            if (
                "gamma_" in name
                or "relative_position_bias_table" in name
                or "attn.qkv.weight" in name
                or "attn.q_bias" in name
                or "attn.v_bias" in name
            ):
                param.requires_grad = False

    for name, param in model.named_parameters():
        if "cls_token" in name or "pos_embed" in name:
            param.requires_grad = False

    pos_total, pos_train = count_params(model)
    print("Preparing vision model post total parameters {} post trained parameters {}".format(pos_total, pos_train))
    return model


def count_params(model):
    n_params = sum([p.numel() for p in model.parameters()])
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_params, trainable_params


# Load model from timm
def load_model(model_name, n_classes, lib):
    print("Path", os.getcwd())
    print("==> Building model..", model_name, "with n_classes", n_classes)
    model = None
    # Model
    if "vit_base_patch16_224" in model_name:
        # model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k',pretrained=True,num_classes=int(n_classes))
        model = timm.create_model(model_name, pretrained=True, num_classes=int(n_classes))
        pre_total, pre_train = count_params(model)
        print("pre total parameters {} pre trained parameters {}".format(pre_total, pre_train))
        # model = ModuleValidator.fix(model)
        pos_total, pos_train = count_params(model)
        print("post total parameters {} post trained parameters {}".format(pos_total, pos_train))
    elif "BiT-M-R" in model_name:
        std = False
        if lib == "non" or lib == "opacus":
            std = True
        model = models.KNOWN_MODELS[model_name](head_size=100, zero_head=True, std=std)
        model.load_from(np.load(f"/models_files/{model_name}.npz"))
        pos_total, pos_train = count_params(model)
        print("post total parameters {} post trained parameters {}".format(pos_total, pos_train))
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=int(n_classes))
        pre_total, pre_train = count_params(model)
        print("pre total parameters {} pre trained parameters {}".format(pre_total, pre_train))
        print(ModuleValidator.validate(model))
        if not ModuleValidator.is_valid(model) and not lib == "non":
            model = ModuleValidator.fix(model)
        model = ModuleValidator.fix(model)
        print("After validation: \n", ModuleValidator.validate(model))

        pos_total, pos_train = count_params(model)
        print("post total parameters {} post trained parameters {}".format(pos_total, pos_train))

    model = models.DpFslLinear(model_name, model, n_classes)

    return model
