from torch import nn


class PromptNet(nn.Module):
    def __init__(self, backbone, prompt, **kwargs):
        super(PromptNet, self).__init__()
        self.backbone = backbone
        self.prompt = prompt

    def forward(self, x, func=None):
        if func == None:
            x_querry = None
        else:
            x_querry = func(x[0], device=x[0].device)
        o1 = self.backbone(x, x_querry, prompt=self.prompt)
        return o1
