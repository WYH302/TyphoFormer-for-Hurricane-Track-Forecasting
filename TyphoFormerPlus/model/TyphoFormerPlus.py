import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Continuous time embedding for t in [0, 1]."""
    if t.dim() > 1:
        t = t.reshape(t.shape[0])
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=t.dtype)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    args = t[:, None] * freqs[None, :] * 1000.0
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_len: int = 128,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steps = x.shape[1]
        z = self.input_proj(x) + self.pos[:, :steps]
        return self.norm(self.encoder(z))


class AnalogEncoder(nn.Module):
    def __init__(
        self,
        pred_len: int,
        d_model: int,
        num_heads: int,
        dropout: float,
        max_k: int = 16,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.point_proj = nn.Linear(2, d_model)
        self.time_pos = nn.Parameter(torch.zeros(1, pred_len, d_model))
        self.path_attn = nn.Linear(d_model, 1)
        self.k_pos = nn.Parameter(torch.zeros(1, max_k, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.k_attn = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, analog: torch.Tensor) -> Dict[str, torch.Tensor]:
        # analog: [B, K, P, 2]
        bsz, k_count, pred_len, _ = analog.shape
        flat = analog.reshape(bsz * k_count, pred_len, 2)
        z = self.point_proj(flat) + self.time_pos[:, :pred_len]
        z = self.temporal_encoder(z)
        path_weight = torch.softmax(self.path_attn(z), dim=1)
        path_emb = torch.sum(path_weight * z, dim=1).reshape(bsz, k_count, -1)
        path_emb = path_emb + self.k_pos[:, :k_count]
        k_weight = torch.softmax(self.k_attn(path_emb), dim=1)
        pooled = torch.sum(k_weight * path_emb, dim=1)
        return {"paths": self.norm(path_emb), "pooled": self.norm(pooled)}


class GatedFusion(nn.Module):
    def __init__(self, d_model: int, num_modalities: int, dropout: float, modality_dropout: float):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_modalities)])
        self.gates = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(num_modalities)])
        self.dropout = nn.Dropout(dropout)
        self.modality_dropout = modality_dropout
        self.norm = nn.LayerNorm(d_model)

    def forward(self, base: torch.Tensor, modalities):
        fused = base
        gate_values = []
        for mod, proj, gate_layer in zip(modalities, self.proj, self.gates):
            if self.training and self.modality_dropout > 0:
                keep = torch.rand(mod.shape[0], 1, 1, device=mod.device) > self.modality_dropout
                mod = mod * keep.to(mod.dtype)
            gate = torch.sigmoid(gate_layer(torch.cat([base, mod], dim=-1)))
            fused = fused + gate * self.dropout(proj(mod))
            gate_values.append(gate.mean())
        if gate_values:
            gates = torch.stack(gate_values)
        else:
            gates = fused.new_zeros(1)
        return self.norm(fused), gates


class FlowDecoder(nn.Module):
    def __init__(
        self,
        pred_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        time_dim: int = 128,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.time_dim = time_dim
        self.traj_proj = nn.Linear(2, d_model)
        self.time_proj = nn.Linear(time_dim, d_model)
        self.cond_proj = nn.Linear(d_model, d_model)
        self.pos = nn.Parameter(torch.zeros(1, pred_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))

    def forward(self, y_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        time_emb = _sinusoidal_time_embedding(t, self.time_dim)
        token = (
            self.traj_proj(y_t)
            + self.time_proj(time_emb).unsqueeze(1)
            + self.cond_proj(context).unsqueeze(1)
            + self.pos[:, : y_t.shape[1]]
        )
        return self.out(self.net(token))


class TrajectoryScorer(nn.Module):
    def __init__(self, pred_len: int, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.encoder = AnalogEncoder(pred_len, d_model, num_heads, dropout, max_k=1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, context: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(traj.unsqueeze(1))["pooled"]
        return self.mlp(torch.cat([context, emb], dim=-1)).squeeze(-1)


class TyphoFormerPlus(nn.Module):
    """TyphoFormer++ model for trajectory/text/analog experiments."""

    def __init__(
        self,
        input_dim: int,
        text_dim: int,
        pred_len: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        modality_dropout: float = 0.2,
        decoder_type: str = "deterministic",
        use_text: bool = True,
        use_analog: bool = True,
        use_positive_analog: bool = True,
        use_negative_context: bool = True,
        max_input_len: int = 32,
    ):
        super().__init__()
        if decoder_type not in {"deterministic", "flow"}:
            raise ValueError("decoder_type must be 'deterministic' or 'flow'")
        self.pred_len = pred_len
        self.decoder_type = decoder_type
        self.use_text = use_text
        self.use_analog = use_analog and (use_positive_analog or use_negative_context)
        self.use_positive_analog = use_positive_analog and self.use_analog
        self.use_negative_context = use_negative_context and use_analog
        self.traj_encoder = SequenceEncoder(input_dim, d_model, 2, num_heads, dropout, max_input_len)
        if self.use_text:
            self.text_encoder = SequenceEncoder(text_dim, d_model, 1, num_heads, dropout, max_input_len)
        if self.use_analog:
            self.analog_encoder = AnalogEncoder(pred_len, d_model, num_heads, dropout)
        num_modalities = int(self.use_text) + int(self.use_positive_analog) + int(self.use_negative_context)
        self.fusion = GatedFusion(d_model, num_modalities=num_modalities, dropout=dropout, modality_dropout=modality_dropout)
        self.context_pos = nn.Parameter(torch.zeros(1, max_input_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.context_norm = nn.LayerNorm(d_model)
        self.det_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_len * 2),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_len * 2),
        )
        self.residual_gate_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * 2),
        )
        self.dual_gate_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )
        self.register_buffer(
            "dual_gate_prior",
            torch.tensor([1.5, 1.5, 1.2, 1.0, 0.5, 0.2, 0.0, -0.3, -0.8, -1.0, -1.2, -1.5]).view(1, -1),
        )
        self.flow_decoder = FlowDecoder(pred_len, d_model, num_heads, num_layers=2, dropout=dropout)
        self.scorer = TrajectoryScorer(pred_len, d_model, num_heads, dropout)

    def encode(self, x_num, x_text, analog_pos, analog_neg) -> Dict[str, torch.Tensor]:
        traj_seq = self.traj_encoder(x_num)
        modalities = []
        features = {"traj_seq": traj_seq}
        if self.use_text:
            text_seq = self.text_encoder(x_text)
            modalities.append(text_seq)
            features["text_seq"] = text_seq
        if self.use_analog:
            pos = self.analog_encoder(analog_pos)
            features["analog_pos"] = pos["pooled"]
            if self.use_positive_analog:
                pos_seq = pos["pooled"].unsqueeze(1).expand(-1, traj_seq.shape[1], -1)
                modalities.append(pos_seq)
            if self.use_negative_context:
                neg = self.analog_encoder(analog_neg)
                neg_seq = neg["pooled"].unsqueeze(1).expand(-1, traj_seq.shape[1], -1)
                modalities.append(neg_seq)
                features["analog_neg"] = neg["pooled"]
        fused, gates = self.fusion(traj_seq, modalities)
        fused = fused + self.context_pos[:, : fused.shape[1]]
        h = self.backbone(fused)
        context = self.context_norm(0.5 * h[:, -1] + 0.5 * h.mean(dim=1))
        features["context"] = context
        features["gates"] = gates
        return features

    def predict_from_context(self, context: torch.Tensor) -> torch.Tensor:
        return self.det_head(context).reshape(context.shape[0], self.pred_len, 2)

    def residual_from_context(self, context: torch.Tensor) -> torch.Tensor:
        return self.residual_head(context).reshape(context.shape[0], self.pred_len, 2)

    def residual_gate_from_context(self, context: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.residual_gate_head(context)).reshape(context.shape[0], self.pred_len, 2)

    def dual_gate_from_context(self, context: torch.Tensor) -> torch.Tensor:
        bias = self.dual_gate_prior.to(device=context.device, dtype=context.dtype)
        lead_steps = getattr(self, "lead_steps", None)
        if lead_steps is not None:
            indices = torch.tensor(
                [min(max(int(step), 1), bias.shape[1]) - 1 for step in lead_steps],
                device=context.device,
                dtype=torch.long,
            )
            bias = bias.index_select(1, indices)
        else:
            if bias.shape[1] < self.pred_len:
                pad = bias[:, -1:].expand(-1, self.pred_len - bias.shape[1])
                bias = torch.cat([bias, pad], dim=1)
            bias = bias[:, : self.pred_len]
        learned = self.dual_gate_head(context)
        return torch.sigmoid(learned + bias).unsqueeze(-1)

    def forward(self, x_num, x_text, analog_pos, analog_neg) -> Dict[str, torch.Tensor]:
        features = self.encode(x_num, x_text, analog_pos, analog_neg)
        features["direct_pred"] = self.predict_from_context(features["context"])
        features["residual_pred"] = self.residual_from_context(features["context"])
        features["pred"] = features["direct_pred"]
        features["residual_gate"] = self.residual_gate_from_context(features["context"])
        features["dual_gate"] = self.dual_gate_from_context(features["context"])
        return features

    def flow_velocity(self, y_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.flow_decoder(y_t, t, context)

    @torch.no_grad()
    def sample_flow(
        self,
        context: torch.Tensor,
        num_samples: int = 20,
        ode_steps: int = 8,
        noise_std: float = 1.0,
    ) -> torch.Tensor:
        bsz = context.shape[0]
        context_rep = context.repeat_interleave(num_samples, dim=0)
        y = torch.randn(
            bsz * num_samples,
            self.pred_len,
            2,
            device=context.device,
            dtype=context.dtype,
        ) * noise_std
        dt = 1.0 / ode_steps
        for step in range(ode_steps):
            t = torch.full((bsz * num_samples,), (step + 0.5) * dt, device=context.device, dtype=context.dtype)
            y = y + dt * self.flow_velocity(y, t, context_rep)
        return y.reshape(bsz, num_samples, self.pred_len, 2)

    def score(self, context: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
        return self.scorer(context, traj)

    @staticmethod
    def alignment_loss(features: Dict[str, torch.Tensor], temperature: float = 0.1) -> torch.Tensor:
        def pooled(seq):
            if seq.dim() == 3:
                seq = seq.mean(dim=1)
            return F.normalize(seq, dim=-1)

        def infonce(a, b):
            if a.shape[0] < 2:
                return a.sum() * 0.0
            logits = pooled(a) @ pooled(b).T / temperature
            labels = torch.arange(logits.shape[0], device=logits.device)
            return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

        loss = features["traj_seq"].sum() * 0.0
        if "text_seq" in features:
            loss = loss + infonce(features["traj_seq"], features["text_seq"])
        if "analog_pos" in features:
            loss = loss + infonce(features["traj_seq"], features["analog_pos"])
        return loss
