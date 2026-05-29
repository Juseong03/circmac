from torch import Tensor
import torch
import torch.nn as nn
from multimolecule.models import RnaBertModel, RnaErnieModel, RnaFmModel, RnaMsmModel

class PretrainedModel(nn.Module):
    def __init__(
            self, 
            rna_model: str,
            d_model: int,
            trainable: bool = False,
            rc: bool = False
        ) -> None:

        super().__init__()
        self.rc = rc
        self.trainable = trainable
        if rna_model == 'rnabert':
            if trainable:
                self.encoder = RnaBertModel.from_pretrained('multimolecule/rnabert')
            self.to_in = nn.Linear(120, d_model)
        elif rna_model == 'rnaernie':
            if trainable:
                self.encoder = RnaErnieModel.from_pretrained('multimolecule/rnaernie')
            self.to_in = nn.Linear(768, d_model)
        elif rna_model == 'rnafm':
            if trainable:
                self.encoder = RnaFmModel.from_pretrained('multimolecule/rnafm')
            self.to_in = nn.Linear(640, d_model)
        elif rna_model == 'rnamsm':
            if trainable:
                self.encoder = RnaMsmModel.from_pretrained('multimolecule/rnamsm')
            self.to_in = nn.Linear(768, d_model)
        else:
            raise ValueError(f"RNA model '{rna_model}' not recognized.")

        self.norm = nn.LayerNorm(d_model)

    def forward(
            self, 
            x: torch.Tensor, 
            mask: torch.Tensor | None = None,
            x_rc: torch.Tensor | None = None,
            mask_rc: torch.Tensor | None = None
        ) -> tuple[Tensor, Tensor]:
        if self.trainable:
            x_emb = self.encoder(input_ids=x, attention_mask=mask)
            x_emb = x_emb['last_hidden_state']
            if self.rc:
                x_emb_rc = self.encoder(input_ids=x_rc, attention_mask=mask_rc)
                x_emb_rc = x_emb_rc['last_hidden_state']
            else:
                x_emb_rc = x_emb

            x_emb = self.to_in(x_emb)
            x_emb = self.norm(x_emb)
            if self.rc:
                x_emb_rc = self.to_in(x_emb_rc)
                x_emb_rc = self.norm(x_emb_rc)
            else:
                x_emb_rc = x_emb
        else:
            x_emb = x
            if self.rc:
                x_emb_rc = x_rc
            else:
                x_emb_rc = x
            x_emb = self.to_in(x_emb)
            x_emb = self.norm(x_emb)
            if self.rc:
                x_emb_rc = self.to_in(x_emb_rc)
                x_emb_rc = self.norm(x_emb_rc)
            else:
                x_emb_rc = x_emb
        return x_emb, x_emb_rc
