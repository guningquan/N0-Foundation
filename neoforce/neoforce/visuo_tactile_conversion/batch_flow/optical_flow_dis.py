"""
DIS 稠密光流（基于 OpenCV Dense Inverse Search）
参考: 灰度帧间 calc，幅值/角度与可视化
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import cv2
import numpy as np

ReferenceMode = Literal["consecutive", "first"]


def create_dis_flow_inference(
    finest_scale: int = 2,
    patch_size: int = 4,
    patch_stride: int = 2,
    gradient_descent_iterations: int = 25,
    variational_refinement_iterations: int = 5,
    variational_refinement_alpha: float = 20.0,
    variational_refinement_delta: float = 5.0,
    variational_refinement_gamma: float = 10.0,
) -> cv2.DISOpticalFlow:
    """与 inference/main.py 中 init_dis_flow() 相同。"""
    dis = cv2.DISOpticalFlow_create()
    dis.setFinestScale(finest_scale)
    dis.setPatchSize(patch_size)
    dis.setPatchStride(patch_stride)
    dis.setGradientDescentIterations(gradient_descent_iterations)
    dis.setVariationalRefinementIterations(variational_refinement_iterations)
    dis.setVariationalRefinementAlpha(variational_refinement_alpha)
    dis.setVariationalRefinementDelta(variational_refinement_delta)
    dis.setVariationalRefinementGamma(variational_refinement_gamma)
    return dis


def create_dis_flow_original() -> cv2.DISOpticalFlow:
    """与 OpticalFlow_DIS.py 相同：cv2.DISOpticalFlow_create() 默认参数。"""
    return cv2.DISOpticalFlow_create()


def create_dis_flow(
    preset: Optional[str] = None,
    *,
    style: str = "original",
    finest_scale: int = 2,
    patch_size: int = 4,
    patch_stride: int = 2,
    gradient_descent_iterations: int = 25,
    variational_refinement_iterations: int = 5,
    variational_refinement_alpha: float = 20.0,
    variational_refinement_delta: float = 5.0,
    variational_refinement_gamma: float = 10.0,
) -> cv2.DISOpticalFlow:
    """
    style:
      original  : 默认 create，无 preset
      inference : init_dis_flow 全套参数
      medium/fast/ultrafast : OpenCV preset（与上面两份不同）
    """
    key = (style or preset or "original").lower().strip()
    if key in ("original", "default", "dis"):
        return create_dis_flow_original()
    if key == "inference":
        return create_dis_flow_inference(
            finest_scale=finest_scale,
            patch_size=patch_size,
            patch_stride=patch_stride,
            gradient_descent_iterations=gradient_descent_iterations,
            variational_refinement_iterations=variational_refinement_iterations,
            variational_refinement_alpha=variational_refinement_alpha,
            variational_refinement_delta=variational_refinement_delta,
            variational_refinement_gamma=variational_refinement_gamma,
        )
    preset_map = {
        "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }
    if key not in preset_map:
        raise ValueError(
            f"未知 style/preset: {style!r}，可选 original | inference | ultrafast | fast | medium"
        )
    if hasattr(cv2, "DISOpticalFlow_create"):
        return cv2.DISOpticalFlow_create(preset_map[key])
    return cv2.optflow.DISOpticalFlow_create(preset_map[key])


def to_gray(image_bgr: np.ndarray, blur_ksize: int = 0) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    if blur_ksize and blur_ksize >= 3:
        k = blur_ksize | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray


def compute_dis_flow(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    dis: Optional[cv2.DISOpticalFlow] = None,
) -> np.ndarray:
    """
    计算稠密光流，返回 (H, W, 2) float32，channel0=u(x), channel1=v(y)。
    与 OpticalFlow_DIS.OpticalFlow_DIS 相同: flow.calc(pre_gary, gary, None)
    """
    if dis is None:
        dis = create_dis_flow_original()
    flow = dis.calc(prev_gray, gray, None)
    return flow.astype(np.float32)


def optical_flow_dis_original(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    dis: Optional[cv2.DISOpticalFlow] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    与 OpticalFlow_DIS.py 完全一致:
      flow_DIS = flow.calc(pre_gary, gary, None)
      cartToPolar -> int16 幅值/角度 -> new_array shape (2, H, W)
    """
    flow_dis = compute_dis_flow(prev_gray, gray, dis)
    polar = flow_to_polar_int16(flow_dis)
    return flow_dis, polar


def flow_to_polar_int16(flow: np.ndarray) -> np.ndarray:
    """对应原脚本 amp_new, angle_new -> new_array。"""
    amp, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    amp_new = amp.astype(np.int16)
    angle_new = angle.astype(np.int16)
    return np.array([amp_new, angle_new])


def flow_to_polar(flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """幅值 amp、角度 angle（弧度）。"""
    u = flow[..., 0]
    v = flow[..., 1]
    amp, angle = cv2.cartToPolar(u, v, angleInDegrees=False)
    return amp.astype(np.float32), angle.astype(np.float32)


@dataclass
class FlowResult:
    flow: np.ndarray
    amplitude: np.ndarray
    angle: np.ndarray


def analyze_flow(flow: np.ndarray) -> FlowResult:
    amp, angle = flow_to_polar(flow)
    return FlowResult(flow=flow, amplitude=amp, angle=angle)


def flow_to_hsv_image(flow: np.ndarray, clip_percentile: float = 99.0) -> np.ndarray:
    """HSV 色轮可视化光流（OpenCV 惯例：色调=方向，亮度=幅值）。"""
    amp, angle = flow_to_polar(flow)
    h = (angle * 180.0 / np.pi / 2.0).astype(np.uint8)
    if clip_percentile > 0:
        vmax = float(np.percentile(amp, clip_percentile)) or 1.0
    else:
        vmax = float(amp.max()) or 1.0
    v = np.clip(amp / vmax * 255.0, 0, 255).astype(np.uint8)
    s = np.full_like(v, 255)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def amplitude_to_gray(amp: np.ndarray, clip_percentile: float = 99.0) -> np.ndarray:
    if clip_percentile > 0:
        vmax = float(np.percentile(amp, clip_percentile)) or 1.0
    else:
        vmax = float(amp.max()) or 1.0
    gray = np.clip(amp / vmax * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def draw_flow_arrows(
    flow: np.ndarray,
    step: int = 16,
    scale: float = 1.0,
    background: Literal["black", "white"] = "black",
    min_magnitude: float = 0.5,
) -> np.ndarray:
    """
    在纯色背景上绘制稀疏箭头（类似原 Final_Flow）。
    step: 网格采样步长；scale: 箭头长度倍率。
    """
    h, w = flow.shape[:2]
    if background == "white":
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    else:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    y_coords, x_coords = np.mgrid[step // 2 : h : step, step // 2 : w : step]
    for y, x in zip(y_coords.flat, x_coords.flat):
        fx, fy = flow[y, x]
        mag = float(np.hypot(fx, fy))
        if mag < min_magnitude:
            continue
        x2 = int(round(x + fx * scale))
        y2 = int(round(y + fy * scale))
        color = (
            int(np.clip(50 + mag * 8, 0, 255)),
            int(np.clip(255 - mag * 4, 0, 255)),
            int(np.clip(100 + mag * 2, 0, 255)),
        )
        cv2.arrowedLine(canvas, (int(x), int(y)), (x2, y2), color, 1, tipLength=0.3)
        cv2.circle(canvas, (int(x), int(y)), 1, color, -1)
    return canvas


def draw_flow_on_image(
    image_bgr: np.ndarray,
    flow: np.ndarray,
    step: int = 16,
    arrow_scale: float = 10.0,
) -> np.ndarray:
    """
    对应 OpticalFlow_DIS.Draw_Flow: x+10*u, y+10*v，绿色箭头，线宽 1。
    """
    out = image_bgr.copy()
    h, w = flow.shape[:2]
    y_coords, x_coords = np.mgrid[step // 2 : h : step, step // 2 : w : step]
    green = (0, 255, 0)
    for y, x in zip(y_coords.flat, x_coords.flat):
        fx, fy = flow[y, x]
        x2 = int(round(x + arrow_scale * fx))
        y2 = int(round(y + arrow_scale * fy))
        cv2.arrowedLine(out, (int(x), int(y)), (x2, y2), green, 1, tipLength=0.3)
        cv2.circle(out, (int(x), int(y)), 1, green, -1)
    return out


def draw_final_flow_black(
    flow: np.ndarray,
    step: int = 16,
    arrow_scale: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    对应 OpticalFlow_DIS.Final_Flow: 黑底，x+u, y+v，随机彩色箭头。
    """
    h, w = flow.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y_coords, x_coords = np.mgrid[step // 2 : h : step, step // 2 : w : step]
    rng = np.random.default_rng(seed)
    for y, x in zip(y_coords.flat, x_coords.flat):
        fx, fy = flow[y, x]
        x2 = int(round(x + arrow_scale * fx))
        y2 = int(round(y + arrow_scale * fy))
        c = tuple(int(rng.integers(0, 256)) for _ in range(3))
        cv2.arrowedLine(canvas, (int(x), int(y)), (x2, y2), c, 1, tipLength=0.3)
        cv2.circle(canvas, (int(x), int(y)), 1, c, -1)
    return canvas


def overlay_flow_on_image(
    image_bgr: np.ndarray,
    flow: np.ndarray,
    step: int = 16,
    scale: float = 10.0,
    min_magnitude: float = 0.0,
) -> np.ndarray:
    """兼容旧接口；scale 为 Draw_Flow 的 10*u 倍率。"""
    del min_magnitude
    return draw_flow_on_image(image_bgr, flow, step=step, arrow_scale=scale)
