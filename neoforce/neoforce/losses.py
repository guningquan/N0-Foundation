from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    def __init__(self, out_dim: int, student_temp: float = 0.1, center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def teacher_distribution(self, teacher_output: torch.Tensor, teacher_temp: float) -> torch.Tensor:
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    def forward(self, student_output: torch.Tensor, teacher_distribution: torch.Tensor) -> torch.Tensor:
        student = F.log_softmax(student_output / self.student_temp, dim=-1)
        return -(teacher_distribution * student).sum(dim=-1).mean()

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center * (1.0 - self.center_momentum))




@dataclass
class LossWeights:
    # Deep supervision on the decoder's intermediate scales. Applied to the mean over
    # scales, so the weight means the same thing regardless of how many stages there are.
    aux: float = 0.0
    force: float = 2.0
    force_global: float = 1.0
    force_delta: float = 0.0
    force_no_contact: float = 0.0
    mask: float = 1.0
    direction: float = 1.0
    contrast: float = 0.5
    dino_v: float = 1.0
    dino_t: float = 1.0
    dino_cross: float = 0.5
    jepa_v: float = 1.0
    jepa_t: float = 1.0


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    dims = tuple(range(2, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (union + eps)).mean()


def force_direction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    b, k, c, h, w = pred.shape
    if c % 3 != 0:
        return pred.new_tensor(0.0)
    pred = pred.reshape(b, k, c // 3, 3, h, w)
    target = target.reshape(b, k, c // 3, 3, h, w)
    sensor_mask = mask.reshape(b, k, c // 3, 3, h, w).amax(dim=3)
    pred = F.normalize(pred, dim=3, eps=eps)
    target = F.normalize(target, dim=3, eps=eps)
    cosine = (pred * target).sum(dim=3)
    return ((1.0 - cosine) * sensor_mask).sum() / sensor_mask.sum().clamp_min(1.0)


def channel_balanced_contact_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none") * mask
    denom = mask.sum(dim=(0, 1, 3, 4)).clamp_min(1.0)
    return (loss.sum(dim=(0, 1, 3, 4)) / denom).mean()


def no_contact_force_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    no_contact = (mask < 0.5).float()
    loss = F.smooth_l1_loss(pred, target, reduction="none") * no_contact
    return loss.sum() / no_contact.sum().clamp_min(1.0)


def force_delta_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    pred_delta = pred[:, 1:] - pred[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    pair_mask = torch.maximum(mask[:, 1:], mask[:, :-1])
    loss = F.smooth_l1_loss(pred_delta, target_delta, reduction="none") * pair_mask
    denom = pair_mask.sum(dim=(0, 1, 3, 4)).clamp_min(1.0)
    return (loss.sum(dim=(0, 1, 3, 4)) / denom).mean()


def multiscale_dense_loss(aux: list, target_force: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Force, mask and direction losses at each intermediate decoder scale.

    Targets are resampled to each scale the same way the main term resamples them. The
    global and temporal terms are deliberately left out: they add cost without adding
    shape information, which is the only thing the coarse stages are there for.
    """
    if not aux:
        return target_force.new_tensor(0.0)
    total = target_force.new_tensor(0.0)
    for entry in aux:
        size = entry["force"].shape[-2:]
        b, k = target_force.shape[:2]
        scaled_force = F.interpolate(
            target_force.flatten(0, 1), size=size, mode="bilinear", align_corners=False
        ).reshape(b, k, -1, *size)
        scaled_mask = F.interpolate(target_mask.flatten(0, 1), size=size, mode="nearest").reshape(b, k, -1, *size)
        total = total + (
            channel_balanced_contact_loss(entry["force"], scaled_force, scaled_mask)
            + F.binary_cross_entropy_with_logits(entry["mask_logits"], scaled_mask)
            + dice_loss(entry["mask_logits"], scaled_mask)
            + force_direction_loss(entry["force"], scaled_force, scaled_mask)
        )
    return total / float(len(aux))


def contrastive_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    z_a = F.normalize(z_a.flatten(0, 1), dim=-1)
    z_b = F.normalize(z_b.flatten(0, 1), dim=-1)
    logits = z_a @ z_b.t() / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class NeoForceLoss(nn.Module):
    def __init__(self, weights: LossWeights = LossWeights()):
        super().__init__()
        self.weights = weights

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_force: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pred_force = outputs["force"]
        mask_logits = outputs["mask_logits"]
        target_mask = target_mask.float()

        # Channel-wise mean of the full-resolution map, computed before any downsampling.
        target_global = target_force.mean(dim=(-2, -1))
        global_l = F.smooth_l1_loss(outputs["force_global"], target_global)

        out_size = pred_force.shape[-2:]
        if target_force.shape[-2:] != out_size:
            b, k = target_force.shape[:2]
            target_force = F.interpolate(
                target_force.flatten(0, 1), size=out_size, mode="bilinear", align_corners=False
            ).reshape(b, k, -1, *out_size)
            target_mask = F.interpolate(target_mask.flatten(0, 1), size=out_size, mode="nearest").reshape(b, k, -1, *out_size)

        force_l = channel_balanced_contact_loss(pred_force, target_force, target_mask)
        force_delta_l = force_delta_loss(pred_force, target_force, target_mask)
        force_no_contact_l = no_contact_force_loss(pred_force, target_force, target_mask)
        mask_bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
        mask_l = mask_bce + dice_loss(mask_logits, target_mask)
        dir_l = force_direction_loss(pred_force, target_force, target_mask)

        aux_l = (
            multiscale_dense_loss(outputs.get("aux", []), target_force, target_mask)
            if self.weights.aux != 0.0
            else pred_force.new_tensor(0.0)
        )
        contrast_l = contrastive_loss(outputs["z_v"], outputs["z_t"])

        total = pred_force.new_tensor(0.0)
        weighted_terms = (
            (self.weights.aux, aux_l),
            (self.weights.force, force_l),
            (self.weights.force_global, global_l),
            (self.weights.force_delta, force_delta_l),
            (self.weights.force_no_contact, force_no_contact_l),
            (self.weights.mask, mask_l),
            (self.weights.direction, dir_l),
            (self.weights.contrast, contrast_l),
        )
        for weight, value in weighted_terms:
            if weight != 0.0:
                total = total + weight * value
        return {
            "loss": total,
            "aux": aux_l.detach(),
            "force": force_l.detach(),
            "force_global": global_l.detach(),
            "force_delta": force_delta_l.detach(),
            "force_no_contact": force_no_contact_l.detach(),
            "mask": mask_l.detach(),
            "direction": dir_l.detach(),
            "contrast": contrast_l.detach(),
        }


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)


def jepa_latent_loss(
    pred: torch.Tensor,
    teacher_latents: torch.Tensor,
    patch_mask: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 between predicted and LayerNorm'd teacher latents, masked positions only.

    pred / teacher_latents: (B, K, N, D); patch_mask: (B, K, N) True = masked.
    """

    target = F.layer_norm(teacher_latents.detach().float(), (teacher_latents.shape[-1],))
    loss = F.smooth_l1_loss(pred.float(), target, reduction="none").mean(dim=-1)
    mask = patch_mask.float()
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def add_jepa_distillation_losses(
    losses: Dict[str, torch.Tensor],
    student: Dict[str, torch.Tensor],
    teacher: Dict[str, torch.Tensor],
    pred_v: torch.Tensor,
    pred_t: torch.Tensor,
    dino_loss: DINOLoss,
    weights: LossWeights,
    teacher_temp: float,
    vision_patch_mask: torch.Tensor,
    touch_patch_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """DINO CLS + cross-modal CLS distillation (kept from baseline) + JEPA latent terms."""

    sv = student["dino_v"].flatten(0, 1)
    st = student["dino_t"].flatten(0, 1)
    sxv = student["dino_cross_v"].flatten(0, 1)
    sxt = student["dino_cross_t"].flatten(0, 1)
    tv_logits = teacher["dino_v"].flatten(0, 1).detach()
    tt_logits = teacher["dino_t"].flatten(0, 1).detach()
    txv_logits = teacher["dino_cross_v"].flatten(0, 1).detach()
    txt_logits = teacher["dino_cross_t"].flatten(0, 1).detach()

    tv = dino_loss.teacher_distribution(tv_logits, teacher_temp)
    tt = dino_loss.teacher_distribution(tt_logits, teacher_temp)
    txv = dino_loss.teacher_distribution(txv_logits, teacher_temp)
    txt = dino_loss.teacher_distribution(txt_logits, teacher_temp)

    dino_v = dino_loss(sv, tv)
    dino_t = dino_loss(st, tt)
    dino_x = 0.5 * (dino_loss(sxv, txt) + dino_loss(sxt, txv))

    jepa_v = jepa_latent_loss(pred_v, teacher["patch_v"], vision_patch_mask)
    jepa_t = jepa_latent_loss(pred_t, teacher["patch_t"], touch_patch_mask)

    distill = (
        weights.dino_v * dino_v
        + weights.dino_t * dino_t
        + weights.dino_cross * dino_x
        + weights.jepa_v * jepa_v
        + weights.jepa_t * jepa_t
    )
    losses["loss"] = losses["loss"] + distill
    losses.update(
        {
            "dino_v": dino_v.detach(),
            "dino_t": dino_t.detach(),
            "dino_cross": dino_x.detach(),
            "jepa_v": jepa_v.detach(),
            "jepa_t": jepa_t.detach(),
        }
    )
    with torch.no_grad():
        dino_loss.update_center(torch.cat([tv_logits, tt_logits, txv_logits, txt_logits], dim=0))
    return losses
