""" AttCAT Generator Code """
# adapted from https://arxiv.org/pdf/2401.09972 (https://github.com/LinxinS97/Mask-LRP/blob/main/Transformer_Explanation/ExplanationGenerator.py)

from transformers import (
    EsmTokenizer,
    AutoModelForSequenceClassification,
)

import torch
import pandas as pd
import numpy as np

from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler



# modified implementation for ESM and use on antibody sequences
class Explainer:
    def __init__(self, model, is_start=False, model_name='esm'):
        self.model = model
        self.device = model.device
        self.model.eval()
        
        self.is_start = is_start
        self.model_name = model_name


    def AttCAT(self, input_ids, attention_mask,
               target_class=None, norm=False, cat_only=False,
               start_layer=0, position_ids=None):

        # only output attentions if they will be used for computing AttCAT
        output_attentions = False if cat_only else True

        # inference (model forward pass)
        result = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            output_hidden_states=True,
                            output_attentions=output_attentions
                           )
        
        output, hs, attns = result[0], result[1], result[2]

        # blocks = layers of [attention + feedforward (intermediate) + layernorm]
        if 'esm' in self.model_name:
            blocks = self.model.__dict__['_modules'][self.model_name].encoder.layer 

        # keep the gradients for all the hidden states (attention gradients not needed for AttCAT)
        for blk_id in range(len(blocks)):
            hs[blk_id].retain_grad()

        # class of interest - default behavior is w.r.t. the predicted class (argmax of logits)
        if target_class == None:
            target_class = np.argmax(output.cpu().data.numpy(), axis=-1)

        # one-hot encode target class, pick out only the relevant logits for back propagation (reason for not batching)
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, target_class] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(input_ids.device) * output)

        # backward pass and store the gradients
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        # calculate attentive class activation token maps
        cams = {}
        # for each layer in the model
        for blk_id in range(len(blocks)):
            # token importance weights (gradients w.r.t. target class)
            hs_grads = hs[blk_id].grad

            # element-wise multiply layer output by importance weights, sum over embedding dimension for each token in sequence length
            cat = (hs[blk_id] * hs_grads).sum(dim=-1).squeeze(0)

            # skip for class activation token values only
            if not cat_only:

                # attention weights of i-th token averaged over all heads and sequence length
                att = attns[blk_id].squeeze(0)
                att = att.mean(dim=0).mean(dim=0)
                cat = cat * att
            
            cams[blk_id] = cat

        # sum over all layers
        cat_expln = sum(cams.values())
        cat_expln = cat_expln.detach().cpu().numpy()

        # normalize each row
        if norm:
            scaler = StandardScaler()
            cat_expln = scaler.fit_transform(cat_expln.reshape(-1, 1))
            cat_expln = cat_expln.squeeze(-1)

        return cat_expln