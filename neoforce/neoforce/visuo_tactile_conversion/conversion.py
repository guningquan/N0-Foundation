"""Tactile conversion model: gel images -> dense three-axis force fields.

Converts raw tactile camera frames into the force-field representation NeoForce
trains on:

    640x360 frame -> upscale 2560x1440 -> homography warp 2154x1523 -> 640x360
      -> DIS optical flow against the episode reference frame
      -> tactile conversion network -> seg/pressure/shear masking -> scaled force map

The homography and threshold constants are inlined rather than imported, so this
module stays free of the capture/GUI dependencies of the original tool and is safe
to import inside DataLoader workers.

Two things this returns that matter downstream:

* ``force`` in physical units (N). ``fz = pressure * PRESSURE_MAX`` with a pressure
  threshold, so in-contact values are small in absolute terms, which is why the
  training target is rescaled per channel (see ``data/force_norm.py``).
* ``mask``, the pipeline's own contact mask, rather than a mask recovered by
  thresholding the force map afterwards. The scale-dependent ``mask_thresholds``
  guesswork disappears: this is the mask the force was actually zeroed with.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Conversion constants. Changing any of them silently changes the ground truth the
# encoder is trained against.
HOMOGRAPHY = np.array(
    [
        [1.9405102947673505, -0.029510258469070856, -689.17422205930859],
        [0.35748905956047777, 1.9090768045081032, -661.8776133515795],
        [0.00035771785981935869, 2.0370458982121464e-05, 1.0],
    ],
    dtype=np.float64,
)
POST_INPUT_SIZE = (2560, 1440)
WARP_INTERMEDIATE_SIZE = (2154, 1523)
FLOW_SIZE = (640, 360)  # (width, height)
BORDER_VALUE = (255, 255, 255)

SEG_THRESHOLD = 0.1
PRESSURE_THRESH = 0.01
PRESSURE_MAX = 0.001
SHEAR_MAX = 0.0005

DIS_KW = dict(
    finest_scale=2,
    patch_size=4,
    patch_stride=2,
    gradient_descent_iterations=25,
    variational_refinement_iterations=5,
    variational_refinement_alpha=20.0,
    variational_refinement_delta=5.0,
    variational_refinement_gamma=10.0,
)

DEFAULT_CODE_DIR = str(Path(__file__).resolve().parent)
MODEL_SUBDIR = "model"
WEIGHTS_NAME = "visuo_tactile_conversion.pt"
NETWORK_MODULE = "network.py"


def correct_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """640x360 -> 2560x1440 -> homography -> 640x360, matching BirdViewCorrector."""
    import cv2

    up = cv2.resize(frame_bgr, POST_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    warped = cv2.warpPerspective(
        up,
        HOMOGRAPHY,
        WARP_INTERMEDIATE_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BORDER_VALUE,
    )
    return cv2.resize(warped, FLOW_SIZE, interpolation=cv2.INTER_AREA)


class TactileConversionModel:
    """Frozen force-field conversion, batched over frames.

    Construct lazily per worker process: the CUDA context and the DIS handle are not
    fork-safe, and the object is intentionally not picklable.
    """

    def __init__(
        self,
        code_dir: str = DEFAULT_CODE_DIR,
        device: str = "cpu",
        gpu_batch: int = 16,
        seg_threshold: float = SEG_THRESHOLD,
        pressure_thresh: float = PRESSURE_THRESH,
        pressure_max: float = PRESSURE_MAX,
        shear_max: float = SHEAR_MAX,
        autocast_fp16: bool = True,
        single_threaded: bool = True,
    ) -> None:
        import cv2
        import torch

        if single_threaded:
            # DIS optical flow and BLAS each spawn many threads; with N DataLoader
            # workers that oversubscribes the CPU and collapses throughput.
            cv2.setNumThreads(1)
            torch.set_num_threads(1)

        self.code_dir = Path(code_dir)
        self.device = torch.device(device)
        self.gpu_batch = int(gpu_batch)
        self.seg_threshold = float(seg_threshold)
        self.pressure_thresh = float(pressure_thresh)
        self.pressure_max = float(pressure_max)
        self.shear_max = float(shear_max)
        self.autocast_fp16 = bool(autocast_fp16) and self.device.type == "cuda"

        self.net = self._load_net()
        self.dis, self._flow_fn, self._gray_fn = self._make_dis()
        self.base_p, self.base_s = self._zero_baseline()

    def _load_net(self):
        import torch

        model_dir = self.code_dir / MODEL_SUBDIR
        module_path = model_dir / NETWORK_MODULE
        weights_path = model_dir / WEIGHTS_NAME
        if not module_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"Tactile conversion model not found under {model_dir}. It must contain "
                f"{NETWORK_MODULE} and {WEIGHTS_NAME}; see docs/Installation.md."
            )
        spec = importlib.util.spec_from_file_location("tactile_conversion_network", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        net = module.MultiTaskPSPNet(in_channels=2, num_classes=2).to(self.device)
        weights = torch.load(weights_path, map_location=self.device, weights_only=False)
        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
        net.load_state_dict(weights)
        net.eval()
        for param in net.parameters():
            param.requires_grad_(False)
        return net

    def _make_dis(self):
        flow_dir = self.code_dir / "batch_flow"
        if not (flow_dir / "optical_flow_dis.py").is_file():
            raise FileNotFoundError(f"batch_flow/optical_flow_dis.py not found under {flow_dir}.")
        if str(flow_dir) not in sys.path:
            sys.path.insert(0, str(flow_dir))
        from optical_flow_dis import (  # type: ignore[import-not-found]
            create_dis_flow,
            optical_flow_dis_original,
            to_gray,
        )

        return create_dis_flow(style="original", **DIS_KW), optical_flow_dis_original, to_gray

    def _zero_baseline(self) -> Tuple[np.ndarray, np.ndarray]:
        """Network response to zero flow, subtracted from every prediction."""
        import torch

        with torch.inference_mode():
            zeros = torch.zeros(1, 2, FLOW_SIZE[1], FLOW_SIZE[0], device=self.device)
            _, pressure, shear = self.net(zeros)
        return (
            pressure.float().cpu().numpy()[0, 0],
            shear.float().cpu().numpy()[0, 0],
        )

    def reference_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Corrected grayscale reference for optical flow (the episode's first frame)."""
        return self._gray_fn(correct_frame(frame_bgr), 0)

    def force_maps(
        self,
        frames_bgr: Sequence[np.ndarray],
        reference_gray: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Infer force maps for a clip.

        Returns ``(force, mask)``, both ``(T, 3, 360, 640)`` float32, channels
        ``[fx, fy, fz]``. ``mask`` is the pipeline's contact mask: the shear mask for
        fx/fy and the pressure mask for fz.
        """
        import torch

        if not len(frames_bgr):
            empty = np.zeros((0, 3, FLOW_SIZE[1], FLOW_SIZE[0]), dtype=np.float32)
            return empty, empty

        flows = []
        for frame in frames_bgr:
            current = self._gray_fn(correct_frame(frame), 0)
            flow, _ = self._flow_fn(reference_gray, current, self.dis)
            flows.append(flow)

        forces: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        for start in range(0, len(flows), self.gpu_batch):
            chunk = np.stack(flows[start : start + self.gpu_batch])  # (B, H, W, 2)
            tensor = torch.from_numpy(chunk).permute(0, 3, 1, 2).float().to(self.device)
            with torch.inference_mode():
                if self.autocast_fp16:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        seg, pressure_out, shear_out = self.net(tensor)
                else:
                    seg, pressure_out, shear_out = self.net(tensor)
            seg = seg.float().cpu().numpy()
            pressure = np.clip(pressure_out.float().cpu().numpy()[:, 0] - self.base_p, 0, None)
            shear = np.clip(shear_out.float().cpu().numpy()[:, 0] - self.base_s, 0, None)

            for b in range(pressure.shape[0]):
                pressure_mask = (seg[b, 0] >= self.seg_threshold) & (pressure[b] >= self.pressure_thresh)
                shear_mask = (seg[b, 1] >= self.seg_threshold) & pressure_mask
                pressure_map = np.where(pressure_mask, pressure[b], 0.0)
                shear_map = np.where(shear_mask, shear[b], 0.0)

                u, v = chunk[b, ..., 0], chunk[b, ..., 1]
                magnitude = np.sqrt(u * u + v * v) + 1e-9
                fx = (u / magnitude) * shear_map * self.shear_max
                fy = (v / magnitude) * shear_map * self.shear_max
                fz = pressure_map * self.pressure_max

                forces.append(np.stack([fx, fy, fz], axis=0).astype(np.float32))
                masks.append(
                    np.stack([shear_mask, shear_mask, pressure_mask], axis=0).astype(np.float32)
                )

        return np.stack(forces, axis=0), np.stack(masks, axis=0)
