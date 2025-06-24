from transformers import CLIPModel
import torch
from typing import Optional
import numpy as np

class Clip(CLIPModel):
    def __init__(self, config):
        super(Clip, self).__init__(config)
        self.vision_model = self.vision_model
        for k, v in self.named_parameters():
            v.requires_grad = False
        self.last = torch.nn.Linear(180, 2)

    def get_image_features(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # b,512
        self.full_feature = vision_outputs[0]  # pooled_output
        pooled_output = vision_outputs[1]  # pooled_output

        self.cls_features = self.visual_projection(pooled_output)
        image_features = self.visual_projection(vision_outputs[0])
        image_features = torch.mean(image_features, dim=1)
        return [self.cls_features, image_features]

    def get_image_cls(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = vision_outputs[1]  # pooled_output
        cls = self.last(self.visual_projection(pooled_output))
        return cls
