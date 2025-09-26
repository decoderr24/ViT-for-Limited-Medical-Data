# src/model.py
import timm

def get_vit_model(num_classes: int = 2):
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes
    )
    return model
