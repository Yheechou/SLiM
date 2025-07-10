from .resnet_preact_bin import resnet18_preact_bin, resnet34_preact_bin, resnet100_preact_bin
import torch, torch.nn as nn
_model_factory = {
    "resnet18_preact_bin":resnet18_preact_bin,
    "resnet34_preact_bin":resnet34_preact_bin,
    "resnet100_preact_bin":resnet100_preact_bin,
}

class Classifier(torch.nn.Module):
    def __init__(self, bin_backbone,num_classes=None):
        super(Classifier,self).__init__()
        self.bin_backbone = bin_backbone
        self.class_fc_bin = nn.Linear(bin_backbone.fc.in_features, num_classes)

    def forward(self,x):
        out_bin = self.bin_backbone(x)
        class_output_bin = self.class_fc_bin(out_bin[-1])
        return [{"logits":class_output_bin}]

def get_model(arch_name, **kwargs):
    bin_backbone =  _model_factory[arch_name](**kwargs)
    model = Classifier(bin_backbone, num_classes = kwargs["num_classes"])
    return model