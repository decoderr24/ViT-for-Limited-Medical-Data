import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# =========================================================
# Focal Loss dengan alpha (class weight) + gamma (focusing)
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        alpha   : list/array/tensor untuk bobot tiap kelas. ex: [1, 1, 2, 1]
        gamma   : focusing parameter, default = 2
        reduction : 'mean' | 'sum' | 'none'
        """
        super(FocalLoss, self).__init__()
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs : prediksi logit model, shape [batch, num_classes]
        targets : label ground truth, shape [batch]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # CE per sample
        pt = torch.exp(-ce_loss)  # probabilitas prediksi benar

        if self.alpha is not None:
            at = self.alpha.to(inputs.device)[targets]  # ambil alpha sesuai kelas target
            focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# =========================================================
# Contoh Model (EfficientNet dari timm)
# =========================================================
class RetinaClassifier(nn.Module):
    def __init__(self, model_name="efficientnet_b0", num_classes=4, pretrained=True):
        super(RetinaClassifier, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)
def create_model(num_classes, model_name="efficientnet_b0", pretrained=True):
    return RetinaClassifier(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
# =========================================================

if __name__ == "__main__":
    # Quick test
    model = RetinaClassifier(num_classes=4)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)  # (2, 4)

    # Test FocalLoss dengan alpha
    criterion = FocalLoss(alpha=[1, 1, 2, 1], gamma=2)
    target = torch.tensor([0, 2])  # contoh label batch
    loss = criterion(y, target)
    print("Loss:", loss.item())
