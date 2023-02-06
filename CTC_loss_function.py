import torch.nn.functional as F
import torch.nn as nn
import torch

class CTCLoss(nn.Module):
    def __init__(self):
        super(CTCLoss, self).__init__()
    
    def forward(self, outputs, targets):
        lprobs = F.log_softmax(outputs[0], dim=-1).transpose(0, 1).contiguous()
        input_lengths = outputs[1]

        pad_mask = (targets != 0)
        targets_flat = targets.masked_select(pad_mask)
        target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=0,
                reduction='sum',
                zero_infinity=False,
            )
        return loss