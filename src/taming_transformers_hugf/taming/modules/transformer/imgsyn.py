import torch
from torch import nn

class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, q_dim, n_heads, k_dim, v_dim, att_drop=0., batch_first=True, resid_drop=0.):
        super().__init__()
        assert k_dim == v_dim
        self.ln1 = nn.LayerNorm(q_dim)
        self.ln2 = nn.LayerNorm(k_dim)
        self.ln3 = nn.LayerNorm(q_dim)
        # self.proj_query = nn.Linear(q_dim, k_dim)
        self.attn = nn.MultiheadAttention(embed_dim=q_dim, num_heads=n_heads, dropout=att_drop, kdim=k_dim, vdim=v_dim, batch_first=batch_first)
        self.mlp = nn.Sequential(
            nn.Linear(q_dim, 4 * q_dim),
            nn.GELU(),  # nice
            nn.Linear(4 * q_dim, q_dim),
            nn.Dropout(resid_drop),
        )
        
    def forward(self, q, kv):
        kv = self.ln2(kv)
        q = self.ln1(q)
        attn, _ = self.attn(q, kv, kv, need_weights=False)

        x = q + attn
        x = x + self.mlp(self.ln3(x))
        
        return x

class ImgSynthesis(nn.Module):
    def __init__(self, *, 
                    q_dim,
                    n_head,
                    n_blocks,
                    drop_att,
                    drop_res):
        super(ImgSynthesis, self).__init__()
        self.q_dim = q_dim
        self.block_list = nn.ModuleList([Block(q_dim=q_dim, n_heads=n_head, k_dim=2*q_dim, v_dim=2*q_dim, att_drop=drop_att, resid_drop=drop_res)
                                    for _ in range(n_blocks)])
    
    def forward(self, img1_hidden:torch.Tensor, img2_hidden:torch.Tensor):
        b, c, h, w = img1_hidden.shape
        img1_hidden = img1_hidden.permute(0, 2, 3, 1).view(b, h*w, c)
        img2_hidden = img2_hidden.permute(0, 2, 3, 1).view(b, h*w, c)
        q = img1_hidden + img2_hidden
        kv = torch.cat([img1_hidden, img2_hidden], dim=-1)
        for blk in self.block_list:
            q = blk(q, kv)
        return q.view(b, h, w, c).permute(0, 3, 1, 2)

