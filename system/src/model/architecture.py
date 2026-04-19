import torch
import torch.nn as nn
import torch.nn.functional as F

class LightAttentionPooling(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=9, dropout=0.2):
        super().__init__()
        pad = kernel // 2
        self.feat_conv = nn.Conv1d(in_dim, in_dim, kernel, padding=pad)
        self.attn_conv = nn.Conv1d(in_dim, in_dim, kernel, padding=pad)
        self.proj = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.LayerNorm(out_dim), nn.Dropout(dropout), nn.GELU()
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask, return_attention=False):
        # x: [B, L, in_dim]  mask: [B, L] bool
        x = x.float()
        xt = x.permute(0, 2, 1)              # [B, C, L]
        feat = self.drop(self.feat_conv(xt))    # [B, C, L]
        attn = self.attn_conv(xt)               # [B, C, L]
        mask_t = mask.unsqueeze(1)              # [B, 1, L]
        attn = attn.masked_fill(~mask_t, -1e4)
        attn = torch.softmax(attn, dim=2)
        weighted = (feat * attn).sum(dim=2)                            # [B, C]
        maxpool = feat.masked_fill(~mask_t, -1e4).max(dim=2).values  # [B, C]
        output = self.proj(torch.cat([weighted, maxpool], dim=1))        # [B, out_dim]
        if return_attention:
            per_residue_attn = attn.mean(dim=1)
            return output, per_residue_attn
        return output

class DrugEncoder(nn.Module):
    def __init__(self, d2_dim, d3_dim, out_dim, dropout=0.2):
        super().__init__()
        self.d2_enc = nn.Sequential(
            nn.Linear(d2_dim, 512), nn.LayerNorm(512), nn.Dropout(dropout), nn.GELU(),
            nn.Linear(512, out_dim), nn.LayerNorm(out_dim), nn.Dropout(dropout), nn.GELU()
        )
        self.d3_enc = nn.Sequential(
            nn.Linear(d3_dim, 512), nn.LayerNorm(512), nn.Dropout(dropout), nn.GELU(),
            nn.Linear(512, out_dim), nn.LayerNorm(out_dim), nn.Dropout(dropout), nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim), nn.Dropout(dropout), nn.GELU()
        )

    def forward(self, d2, d3):
        return self.fusion(torch.cat([self.d2_enc(d2), self.d3_enc(d3)], dim=1))

class DTIModel(nn.Module):
    def __init__(self, d2_dim, d3_dim, p_dim=1024,
                 drug_out=256, prot_out=256, hidden=512, dropout=0.2):
        super().__init__()
        self.drug_enc = DrugEncoder(d2_dim, d3_dim, drug_out, dropout)
        self.prot_enc = LightAttentionPooling(p_dim, prot_out, kernel=9, dropout=dropout)

        ca_dim = 256
        self.drug_proj = nn.Linear(drug_out, ca_dim)
        self.prot_proj = nn.Linear(prot_out, ca_dim)
        self.ca_drug = nn.MultiheadAttention(ca_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.ca_prot = nn.MultiheadAttention(ca_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.ca_norm_d = nn.LayerNorm(ca_dim)
        self.ca_norm_p = nn.LayerNorm(ca_dim)

        self.decoder = nn.Sequential(
            nn.Linear(ca_dim * 2, hidden),
            nn.LayerNorm(hidden), nn.Dropout(dropout), nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2), nn.Dropout(dropout), nn.GELU(),
            nn.Linear(hidden // 2, 2)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, d2, d3, p, mask, return_attention=False):
        drug_h = self.drug_enc(d2, d3)            # [B, drug_out]
        
        if return_attention:
            prot_h, attn_weights = self.prot_enc(p, mask, return_attention=True)
        else:
            prot_h = self.prot_enc(p, mask)            # [B, prot_out]

        dq = self.drug_proj(drug_h).unsqueeze(1)  # [B, 1, ca_dim]
        pk = self.prot_proj(prot_h).unsqueeze(1)  # [B, 1, ca_dim]

        d2p, _ = self.ca_drug(dq, pk, pk)         # drug ← prot context
        p2d, _ = self.ca_prot(pk, dq, dq)         # prot ← drug context

        drug_ca = self.ca_norm_d(self.drug_proj(drug_h) + d2p.squeeze(1))  # [B, ca_dim]
        prot_ca = self.ca_norm_p(self.prot_proj(prot_h) + p2d.squeeze(1))  # [B, ca_dim]

        out = self.decoder(torch.cat([drug_ca, prot_ca], dim=1))  # [B, 2]
        
        if return_attention:
            return out, attn_weights
        return out
