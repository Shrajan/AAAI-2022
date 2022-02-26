def get_model(opt=None):
    """
    Retrieves the model architecture based on the requirement.
    Args: model_name: Must be a string from choice list.
          opt: Command line argument.
    """
    # Original UNet
    if opt.model_name == "unet":
        from networks.unet import UNet as model_class

    # Attention-UNet
    elif opt.model_name == "attention_unet":
        from networks.attention_unet import Attention_UNet as model_class

    # Original VNet
    elif opt.model_name == "vnet":
        from networks.vnet import VNet as model_class

    # OOCS UNet k=3
    elif opt.model_name == "oocs_unet_k3":
        from networks.oocs_unet import OOCS_UNet_k3 as model_class

    # OOCS UNet k=5
    elif opt.model_name == "oocs_unet_k5":
        from networks.oocs_unet import OOCS_UNet_k5 as model_class

    # OOCS VNet k=3
    elif opt.model_name == "oocs_vnet_k3":
        from networks.oocs_vnet import OOCS_VNet_k3  as model_class

    # OOCS VNet k=5
    elif opt.model_name == "oocs_vnet_k5":
        from networks.oocs_vnet import OOCS_VNet_k5  as model_class

    # OOCS Attention U-Net K=3
    elif opt.model_name == "oocs_attention_unet_k3":
        from networks.oocs_attention_unet import OOCS_Attention_UNet_k3 as model_class

    # OOCS Attention U-Net K=5
    elif opt.model_name == "oocs_attention_unet_k5":
        from networks.oocs_attention_unet import OOCS_Attention_UNet_k5 as model_class
        
    else:
        raise Exception("Re-check the model name, otherwise the model isn't available.")
    
    model = model_class(opt)
    return model

