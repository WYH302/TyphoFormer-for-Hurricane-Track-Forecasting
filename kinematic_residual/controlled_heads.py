"""Controlled numeric-only heads. Frozen local encoder; no label input at inference."""
import torch
from vendor.model import TyphoFormerPlus
from vendor.common import latlon_to_disp, normalize_disp


def cv_displacement(history, lead_steps, convention="legacy_previous_origin"):
    previous, origin = history[:, -2], history[:, -1]
    if convention == "legacy_previous_origin":
        step = latlon_to_disp(origin, previous)
    elif convention == "forecast_origin":
        step = -latlon_to_disp(previous, origin)
    else:
        raise ValueError(convention)
    leads = torch.as_tensor(lead_steps, dtype=history.dtype, device=history.device)
    return step[:, None] * leads[None, :, None]


class ControlledNumericModel(TyphoFormerPlus):
    def __init__(self, input_dim, input_len, lead_steps, target_mean, target_std,
                 head="direct", d_model=256, num_heads=4, num_layers=3,
                 dropout=0.1, convention="legacy_previous_origin"):
        if head not in {"direct", "residual", "fixed", "adaptive"}:
            raise ValueError(head)
        # Construct identically before pruning so shared tensors initialize identically.
        super().__init__(input_dim, 1, len(lead_steps), d_model, num_heads,
                         num_layers, dropout, 0.0, use_text=False,
                         use_analog=False, max_input_len=input_len)
        self.head, self.convention = head, convention
        for name in ("residual_gate_head", "flow_decoder", "scorer", "dual_gate_prior"):
            delattr(self, name)
        if head == "direct":
            del self.residual_head
        if head == "residual":
            del self.det_head
        if head != "adaptive":
            del self.dual_gate_head
        self.register_buffer("lead_steps", torch.as_tensor(lead_steps, dtype=torch.float32))
        self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.as_tensor(target_std, dtype=torch.float32))
        if not torch.all(self.target_std > 0):
            raise ValueError("Nonpositive target scale")

    def forward(self, x_num, history_latlon, force_cv=False):
        context = self.encode(x_num, None, None, None)["context"]
        cv_raw = cv_displacement(history_latlon, self.lead_steps, self.convention)
        cv_norm = normalize_disp(cv_raw, self.target_mean, self.target_std)
        if force_cv:
            return {"pred": cv_norm, "cv": cv_norm}
        out = {"cv": cv_norm}
        if self.head != "residual":
            out["direct"] = self.det_head(context).reshape_as(cv_norm)
        if self.head != "direct":
            # Residual is in std-scaled km; do not subtract target_mean twice.
            out["residual_delta"] = self.residual_head(context).reshape_as(cv_norm)
            out["cv_residual"] = cv_norm + out["residual_delta"]
        if self.head == "direct":
            out["pred"] = out["direct"]
        elif self.head == "residual":
            out["pred"] = out["cv_residual"]
        else:
            out["gate"] = (torch.sigmoid(self.dual_gate_head(context)).unsqueeze(-1)
                           if self.head == "adaptive" else torch.full_like(cv_norm[..., :1], 0.5))
            out["pred"] = out["gate"] * out["cv_residual"] + (1 - out["gate"]) * out["direct"]
        return out
