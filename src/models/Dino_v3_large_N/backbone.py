import timm


def build_backbone():
    model = timm.create_model(
        "vit_large_patch16_dinov3",
        pretrained=True,
        num_classes=0,  # remove classifier, output is raw embedding [1024]
    )
    return model

