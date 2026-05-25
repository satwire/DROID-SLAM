from pathlib import Path

import roma
import torch
from lietorch import SE3


class IAAIAdapter:
    def __init__(
        self,
        iaai_disambiguation_dir: str,
        r_full_pct=0.50,
        r_zero_pct=0.95,
        device="cuda",
    ) -> None:
        path = Path(iaai_disambiguation_dir)
        self.trans: torch.Tensor = (
            torch.load(path / "translations.pt").to(device).float()
        )
        self.rots: torch.Tensor = torch.load(path / "rotations.pt").to(device).float()
        self.residuals: torch.Tensor = (
            torch.load(path / "residuals.pt").to(device).float()
        )
        assert self.residuals.shape[0] == self.trans.shape[0], (
            f"IAAI residuals length ({self.residuals.shape[0]}) must match "
            f"deltas length ({self.trans.shape[0]}). Did upstream trimming change?"
        )
        self.num_deltas = self.trans.shape[0]
        self.device = device

        self.r_full = float(self.residuals.quantile(r_full_pct))
        self.r_zero = float(self.residuals.quantile(r_zero_pct))

        n_full = int((self.residuals <= self.r_full).sum())
        n_zero = int((self.residuals >= self.r_zero).sum())
        print(
            f"IAAI residuals (N={self.num_deltas}) — "
            f"median={self.residuals.median().item():.0f}, "
            f"max={self.residuals.max().item():.0f}; "
            f"thresholds r_full={self.r_full:.0f} (p{int(r_full_pct * 100)}), "
            f"r_zero={self.r_zero:.0f} (p{int(r_zero_pct * 100)}); "
            f"{n_full} frames @ conf=1.0, {n_zero} @ conf=0.0"
        )

    def _confidence(self, residual) -> float:
        if residual <= self.r_full:
            return 1.0
        if residual >= self.r_zero:
            return 0.0
        return float((self.r_zero - residual) / (self.r_zero - self.r_full))

    def get_delta(self, frame_idx):
        i = frame_idx - 1
        if i < 0 or i >= self.num_deltas:
            return None, 0.0

        R = self.rots[i]
        t = self.trans[i]
        q_xyzw = roma.rotmat_to_unitquat(R)
        delta_pose = SE3(torch.cat([t, q_xyzw]))

        return delta_pose, self._confidence(self.residuals[i])
