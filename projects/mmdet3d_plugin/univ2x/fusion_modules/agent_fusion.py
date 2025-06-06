#-------------------------------------------------------------------------------------------#
# UniV2X: End-to-End Autonomous Driving through V2X Cooperation  #
# Source code: https://github.com/AIR-THU/UniV2X                                      #
# Copyright (c) DAIR-V2X. All rights reserved.                                                    #
#-------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..dense_heads.track_head_plugin import Instances


class AgentQueryFusion(nn.Module):

    def __init__(self, pc_range, embed_dims=256, fusion_mode='addition', 
                 num_attention_heads=8, attention_dropout=0.1, feedforward_dim=None):
        super(AgentQueryFusion, self).__init__()

        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.fusion_mode = fusion_mode
        
        assert fusion_mode in ['addition', 'attention'], \
            f"fusion_mode must be 'addition' or 'attention', got {fusion_mode}"

        # reference_points ---> pos_embed
        self.get_pos_embedding = nn.Linear(3, self.embed_dims)
        # cross-agent feature alignment
        self.cross_agent_align = nn.Linear(self.embed_dims+9, self.embed_dims)
        self.cross_agent_align_pos = nn.Linear(self.embed_dims+9, self.embed_dims)
        
        if self.fusion_mode == 'addition':
            # Original simple fusion
            self.cross_agent_fusion = nn.Linear(self.embed_dims, self.embed_dims)
        else:
            # Attention-based fusion
            if feedforward_dim is None:
                feedforward_dim = 4 * self.embed_dims
            
            self.attention_layer = nn.TransformerDecoderLayer(
                d_model=self.embed_dims,
                nhead=num_attention_heads,
                dim_feedforward=feedforward_dim,
                dropout=attention_dropout,
                batch_first=True
            )
            
            # Position and type embeddings for different agents
            self.pos_embed = nn.Linear(3, self.embed_dims)
            
            self.inf_type_embed = nn.Parameter(torch.randn(1, 1, self.embed_dims))
            self.veh_type_embed = nn.Parameter(torch.randn(1, 1, self.embed_dims))

        # Contrastive loss components
        self.projection_head = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims // 2)
        )
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

        # parameter initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _loc_norm(self, locs, pc_range):
        """
        absolute (x,y,z) in global coordinate system ---> normalized (x,y,z)
        """
        from mmdet.models.utils.transformer import inverse_sigmoid

        locs[..., 0:1] = (locs[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
        locs[..., 1:2] = (locs[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
        locs[..., 2:3] = (locs[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

        locs = inverse_sigmoid(locs)

        return locs
    
    def _loc_denorm(self, ref_pts, pc_range):
        """
        normalized (x,y,z) ---> absolute (x,y,z) in global coordinate system
        """
        locs = ref_pts.sigmoid().clone()

        locs[:, 0:1] = (locs[:, 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
        locs[:, 1:2] = (locs[:, 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
        locs[:, 2:3] = (locs[:, 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2])

        return locs
    
    def _dis_filt(self, veh_pts, inf_pts, veh_dims):
        """
        filter according to distance
        """
        diff = torch.abs(veh_pts - inf_pts) / veh_dims
        return diff[0] <= 1 and diff[1] <= 1 and diff[2] <= 1
    
    def _calculate_contrastive_loss(self, veh_queries, inf_queries):
        # veh_queries: [N, D], inf_queries: [N, D]
        # N is the number of matched pairs.
        
        # Project features into the contrastive space
        proj_veh_queries = self.projection_head(veh_queries)
        proj_inf_queries = self.projection_head(inf_queries)

        # L2-normalize features
        proj_veh_queries = F.normalize(proj_veh_queries, p=2, dim=-1)
        proj_inf_queries = F.normalize(proj_inf_queries, p=2, dim=-1)

        # Calculate cosine similarity. The diagonal represents positive pairs.
        # Off-diagonal elements are in-batch negative pairs.
        logits = torch.matmul(proj_veh_queries, proj_inf_queries.T) / self.temperature.exp()
        
        # Symmetric loss
        batch_size = logits.shape[0]
        if batch_size == 0:
            return torch.tensor(0.0).to(logits.device)
            
        labels = torch.arange(batch_size, device=logits.device)
        loss_veh_to_inf = F.cross_entropy(logits, labels)
        loss_inf_to_veh = F.cross_entropy(logits.T, labels)
        
        return (loss_veh_to_inf + loss_inf_to_veh) / 2.0

    def _query_matching(self, inf_ref_pts, veh_ref_pts, veh_mask, veh_pred_dims):
        """
        inf_ref_pts: [..., 3] (xyz)
        veh_ref_pts: [..., 3] (xyz)
        veh_pred_dims: [..., 3] (dx, dy, dz)
        """
        inf_nums = inf_ref_pts.shape[0]
        veh_nums = veh_ref_pts.shape[0]
        cost_matrix = np.ones((veh_nums, inf_nums))
        cost_matrix.fill(1e6)

        for i in veh_mask:
            # for j in range(i,inf_nums):
            for j in range(inf_nums):
                cost_matrix[i][j] = torch.sum((veh_ref_pts[i] - inf_ref_pts[j])**2)**0.5
                if not self._dis_filt(veh_ref_pts[i], inf_ref_pts[j], veh_pred_dims[i]):
                    cost_matrix[i][j] = 1e6
        
        idx_veh, idx_inf = linear_sum_assignment(cost_matrix)

        return idx_veh, idx_inf, cost_matrix
    
    def attention_fusion(self, inf_features, veh_features, inf_pts, veh_pts, attention_mask):
        """
        Attention-based fusion of infrastructure and vehicle features.
        
        Args:
            inf_features (torch.Tensor): Infrastructure features [N, D]
            veh_features (torch.Tensor): Vehicle features [N, D]
        
        Returns:
            torch.Tensor: Fused features [N, D]
        """
        B, N, D = inf_features.shape
        
        # Add positional and type embeddings
        inf_feats_emb = inf_features + self.inf_type_embed + self.pos_embed(inf_pts) # [N, 1, D]
        veh_feats_emb = veh_features + self.veh_type_embed + self.pos_embed(veh_pts)  # [N, 1, D]
        
        
        # Apply transformer attention
        attended_feats = self.attention_layer(veh_feats_emb, inf_feats_emb, memory_mask=attention_mask.cuda())  # [N, 2, D]
        

        return attended_feats
    
    def _query_fusion(self, inf, veh, inf_idx, veh_idx, cost_matrix):
        """
        Query fusion: 
            replacement for scores, ref_pts and pos_embed according to confidence_score
            fusion for features via MLP or Attention
        
        inf: Instance from infrastructure
        veh: Instance from vehicle
        inf_idx: matched idxs for inf side
        veh_idx: matched idxs for veh side
        cost_matrix
        """

        veh_accept_idx = []
        inf_accept_idx = []

        # Collect matched queries before modifying them
        pre_fusion_veh_queries = []
        pre_fusion_inf_queries = []

        for i in range(len(veh_idx)):
            if cost_matrix[veh_idx[i]][inf_idx[i]] < 1e5:
                v_i = veh_idx[i]
                i_i = inf_idx[i]
                veh_accept_idx.append(v_i)
                inf_accept_idx.append(i_i)
                
                # Store pre-fusion queries for contrastive loss
                pre_fusion_veh_queries.append(veh.query[v_i, self.embed_dims:])
                pre_fusion_inf_queries.append(inf.query[i_i, self.embed_dims:])
                
                if self.fusion_mode == 'addition':
                    # Original simple addition-based fusion
                    veh.query[v_i, self.embed_dims:] = veh.query[v_i, self.embed_dims:] + self.cross_agent_fusion(inf.query[i_i, self.embed_dims:])
        
        contrastive_loss = torch.tensor(0.0, device=veh.query.device)
        if len(pre_fusion_veh_queries) > 0:
            contrastive_loss = self._calculate_contrastive_loss(
                torch.stack(pre_fusion_veh_queries),
                torch.stack(pre_fusion_inf_queries)
            )
        
        if self.fusion_mode == 'attention':
            # Attention-based fusion
            inf_feat = inf.query[:, self.embed_dims:].unsqueeze(0)  # [1, D]
            veh_feat = veh.query[:, self.embed_dims:].unsqueeze(0)  # [1, D]
            
            cost_matrix = torch.as_tensor(cost_matrix)
            attention_mask = cost_matrix < 1e5
            
            inf_pts = inf.ref_pts
            veh_pts = veh.ref_pts
            # Apply attention fusion
            fused_feat = self.attention_fusion(inf_feat, veh_feat, inf_pts, veh_pts, attention_mask)  # [1, D]
            
            # Update vehicle query with fused features
            veh.query = torch.cat([
                veh.query[:, :self.embed_dims], 
                veh.query[:, self.embed_dims:] + fused_feat.squeeze(0)
            ], dim=1)

        return veh, veh_accept_idx, inf_accept_idx, contrastive_loss
    

    def _query_complementation(self, inf, veh, inf_accept_idx):
        """
        Query complementation: replace low-confidence vehicle-side query with unmatched inf-side query

        inf: Instance from infrastructure
        veh: Instance from vehicle
        inf_accept_idx: idxs of matched instances
        """
        # supply_idx = -1
        for i in range(inf.ref_pts.shape[0]):
            if i not in inf_accept_idx:
                veh = Instances.cat([veh, inf[i]])

        return veh

    
    def forward(self, inf, veh, ego2other_rt, other_agent_pc_range, threshold=0.3):
        """
        Query-based cross-agent interaction: only update ref_pts and query.

        inf: Instance from infrastructure
        veh: Instance from vehicle
        ego2other_rt: calibration parameters from infrastructure to vehicle
        """
        inf_mask = torch.where(inf.obj_idxes>=0)
        inf = inf[inf_mask]
        if len(inf) == 0:
            return veh, torch.tensor(0.0).to(veh.query.device)
        inf_mask_new = torch.where(inf.obj_idxes>=0)
        
        #not care obj_idxes of inf
        inf.obj_idxes = torch.ones_like(inf.obj_idxes) * -1
                
        # ref_pts norm2absolute
        inf_ref_pts = self._loc_denorm(inf.ref_pts, other_agent_pc_range)
        veh_ref_pts = self._loc_denorm(veh.ref_pts, self.pc_range)
            
        # inf_ref_pts inf2veh
        calib_inf2veh = np.linalg.inv(ego2other_rt[0].cpu().numpy().T)
        calib_inf2veh = inf_ref_pts.new_tensor(calib_inf2veh)
        inf_ref_pts = torch.cat((inf_ref_pts, torch.ones_like(inf_ref_pts[..., :1])), -1).unsqueeze(-1)
        inf_ref_pts = torch.matmul(calib_inf2veh, inf_ref_pts).squeeze(-1)[..., :3]

        # ego_selection
        remove_ego_ins = True
        if remove_ego_ins:
            H_B, H_F = -2.04, 2.04 # H = 4.084
            W_L, W_R = -0.92, 0.92 # W = 1.85
            def del_tensor_ele(arr,index):
                arr1 = arr[0:index]
                arr2 = arr[index+1:]
                return torch.cat((arr1,arr2),dim=0)

            inf_mask_new = list(inf_mask_new)
            for ii in range(len(inf_ref_pts)):
                xx, yy = inf_ref_pts[ii][0], inf_ref_pts[ii][1]
                if xx >= H_B and xx <= H_F and yy >= W_L and yy <= W_R:
                    inf_mask_new[0] = del_tensor_ele(inf_mask_new[0], ii)
                    break
            inf_mask_new = tuple(inf_mask_new)
            inf = inf[inf_mask_new]
            inf_ref_pts = inf_ref_pts[inf_mask_new]

        # matching
        veh_mask = torch.where(veh.scores >= 0.05)[0]
        veh_idx, inf_idx, cost_matrix = self._query_matching(inf_ref_pts, veh_ref_pts, veh_mask, veh.pred_boxes[..., [2,3,5]]) # veh.pred_boxes x,y,dx,dy,z,dz

        # ref_pts normalization
        inf_ref_pts = self._loc_norm(inf_ref_pts, self.pc_range)
        veh_ref_pts = self._loc_norm(veh_ref_pts, self.pc_range)
        inf.ref_pts = inf_ref_pts
        veh.ref_pts = veh_ref_pts

        # cross-agent feature alignment
        inf2veh_r = calib_inf2veh[:3,:3].reshape(1,9).repeat(inf.query.shape[0], 1)
        inf.query[..., :self.embed_dims] = self.cross_agent_align_pos(torch.cat([inf.query[..., :self.embed_dims],inf2veh_r], -1))
        inf.query[..., self.embed_dims:] = self.cross_agent_align(torch.cat([inf.query[..., self.embed_dims:],inf2veh_r], -1))

        # cross-agent query fusion (supports both addition and attention modes)
        veh, veh_accept_idx, inf_accept_idx, contrastive_loss = self._query_fusion(inf, veh, inf_idx, veh_idx, cost_matrix)

        # cross-agent query complementation
        veh = self._query_complementation(inf, veh, inf_accept_idx)

        return veh, contrastive_loss