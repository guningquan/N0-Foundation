"""ZMQ + msgpack client for the N0-VTLA inference server (``scripts/serve_zmq.py`` in N0-VTLA).

Protocol helpers for ``eval/eval_n0vtla_dual_joint_cl.py`` (NeoSim dual-arm, config
``sim_dual_arm_tactile``). Needs only numpy / cv2 / msgpack / pyzmq, so it imports (and can be
tested) without Isaac Sim.

Wire protocol (msgpack over a ZMQ REQ socket; the server is one REP socket)
---------------------------------------------------------------------------
request  {"cmd": "reset"}                       -> {"status": "ok"}
request  {"cmd": "predict",                      -> {"status": "ok",
          "state": [float]*16,                       "actions": [[float]*32]*16,
          "prompt": str,                             "infer_time_ms": float}
          <3 camera keys + 4 tactile keys>: PNG bytes}
On failure the server replies {"status": "error", "message": str}; the client raises.

Key mapping (NeoSim observation -> wire key -> model view)
----------------------------------------------------------
  observation["observation"]["head"]["rgb"]       -> observation/image                     -> cam_high
  observation["observation"]["wrist"]["rgb"]      -> observation/wrist_image               -> cam_left_wrist
  observation["observation"]["wrist_b"]["rgb"]    -> observation/secondary_image           -> cam_right_wrist
  observation["tactile"]["left_tactile"]["rgb"]    -> observation/left_wrist_left_tactile   -> tactile_a
  observation["tactile"]["right_tactile"]["rgb"]   -> observation/left_wrist_right_tactile  -> tactile_b
  observation["tactile"]["left_tactile_b"]["rgb"]  -> observation/right_wrist_left_tactile  -> tactile_c
  observation["tactile"]["right_tactile_b"]["rgb"] -> observation/right_wrist_right_tactile -> tactile_d
The server only takes tactile keys of the form ``observation/*_tactile``, sorts them by its
``TAC_WIRE_ORDER`` and matches them POSITIONALLY onto the config's ``tactile_image_keys``; a view
it does not receive is silently dropped. NeoSim's arm-B pads end in ``_tactile_b``, so they MUST
be renamed, otherwise the model only sees 2 of its 4 tactile views.

Images are HxWx3 RGB uint8 at their NATIVE resolution (cameras 270x480, tactile 240x320): the
model transform letterboxes to 224x224, as in training, so do not resize. They are flipped
RGB->BGR and PNG-encoded (lossless): the server cv2-decodes the bytes and flips BGR->RGB, which
restores the original RGB exactly.

Actions: the server zero-pads every step to 32 dims; ``predict`` returns the leading 16
(arm A 7 joints + gripper, arm B 7 joints + gripper) for every step of the chunk, i.e. (16, 16)
float32 for ``sim_dual_arm_tactile``. Execute the WHOLE chunk (serve_zmq.py DEPLOYMENT note).
serve_zmq.py cannot report which config or checkpoint it serves, and its zero padding makes any
reply wide enough, so ``predict`` raises on a reply that is not this config's: another horizon,
non-zero columns past 16 (another action space) or an all-zero arm B (an 8-dim single-arm server,
which would drive arm B to q=0 with the gripper closed).

ONE CLIENT PER SERVER PROCESS
-----------------------------
serve_zmq.py keeps a SINGLE per-episode tactile baseline (tac_0): ``reset`` clears it and the
first ``predict`` after a reset captures it. A REP socket answers every connected REQ client in
turn, so two clients sharing one server interleave their resets and predicts, and each one's
contact signal (tac_t - tac_0) is computed against a baseline from the other's episode, silently.
Start one server process (own port) per eval client, and call ``reset`` at the start of every
episode.

Timeouts
--------
A REQ socket that did not get its reply stays stuck waiting for it and refuses further sends.
On timeout the client closes and recreates the socket (a late reply to the abandoned request
goes to the closed socket and is discarded), then raises ``TimeoutError``.
"""
from __future__ import annotations

import time

import cv2
import msgpack
import numpy as np
import zmq

# NeoSim camera name (observation["observation"][name]["rgb"]) -> serve_zmq.py wire key (CAM_MAP).
CAM_WIRE_KEYS = {
    "head": "observation/image",                # -> cam_high
    "wrist": "observation/wrist_image",         # -> cam_left_wrist  (arm A)
    "wrist_b": "observation/secondary_image",   # -> cam_right_wrist (arm B)
}

# NeoSim tactile pad (observation["tactile"][name]["rgb"]) -> serve_zmq.py wire key. The wire keys
# follow the server's TAC_WIRE_ORDER, so they land on tactile_a..tactile_d in this order.
TAC_WIRE_KEYS = {
    "left_tactile": "observation/left_wrist_left_tactile",        # -> tactile_a (arm A, left finger)
    "right_tactile": "observation/left_wrist_right_tactile",      # -> tactile_b (arm A, right finger)
    "left_tactile_b": "observation/right_wrist_left_tactile",     # -> tactile_c (arm B, left finger)
    "right_tactile_b": "observation/right_wrist_right_tactile",   # -> tactile_d (arm B, right finger)
}

ARM_DIM = 8          # 7 joints + gripper
STATE_DIM = 16       # 2 x (7 joints + gripper)
ACTION_DIM = 16      # same layout; the server pads it to 32
ACTION_HORIZON = 16  # action_horizon of sim_dual_arm_tactile


def encode_image(img) -> bytes:
    """HxWx3 RGB uint8 -> PNG bytes of the BGR image, the format serve_zmq.py::_decode_image expects.

    The server cv2-decodes the bytes (BGR) and flips them to RGB, so flipping here makes the
    round trip exact. No resize: the frame keeps its native resolution.
    """
    arr = np.asarray(img)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected an HxWx3 uint8 RGB image, got {arr.dtype} {arr.shape}")
    ok, buf = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f"PNG encode failed for an image of shape {arr.shape}")
    return buf.tobytes()


def _encode_views(views: dict, wire_keys: dict, kind: str) -> dict:
    """{NeoSim name: RGB image} -> {wire key: PNG bytes}; every view must be present, none extra."""
    missing = [k for k in wire_keys if k not in views]
    unknown = [k for k in views if k not in wire_keys]
    if missing or unknown:
        # A missing view would be dropped by the server without an error, so refuse it here.
        raise ValueError(f"{kind} views must be exactly {list(wire_keys)}; missing {missing}, unknown {unknown}")
    return {wire_keys[k]: encode_image(views[k]) for k in wire_keys}


def build_predict_request(state, prompt: str | None, cams: dict, tacs: dict, state_dim: int = STATE_DIM) -> dict:
    """Build the msgpack-able ``predict`` request.

    state: ``state_dim`` floats (arm A 7 joints + gripper, arm B 7 joints + gripper).
    prompt: the training prompt, verbatim; empty/None lets the server use its --default-prompt.
    cams: {"head", "wrist", "wrist_b"} -> HxWx3 RGB uint8.
    tacs: {"left_tactile", "right_tactile", "left_tactile_b", "right_tactile_b"} -> HxWx3 RGB uint8.
    """
    s = np.asarray(state, dtype=np.float32).reshape(-1)
    if s.size != state_dim:
        # The server zero-pads or truncates the state to its width without complaint.
        raise ValueError(f"state must have {state_dim} values, got {s.size}")
    if not np.isfinite(s).all():
        raise ValueError(f"state is not finite: {s.tolist()}")
    msg = {"cmd": "predict", "state": s.tolist()}
    if prompt:
        msg["prompt"] = str(prompt)
    msg.update(_encode_views(cams, CAM_WIRE_KEYS, "camera"))
    msg.update(_encode_views(tacs, TAC_WIRE_KEYS, "tactile"))
    return msg


class N0VTLAZmqClient:
    """REQ client for serve_zmq.py: ``reset()`` once per episode, then ``predict()`` per chunk.

    ONE client per server process: the server holds a single per-episode tactile baseline (see
    the module docstring).
    """

    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 5557,
                 timeout_s: float = 60.0,
                 state_dim: int = STATE_DIM,
                 action_dim: int = ACTION_DIM,
                 action_horizon: int = ACTION_HORIZON) -> None:
        self._addr = f"tcp://{host}:{port}"
        self._timeout_ms = max(1, int(timeout_s * 1000))
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._action_horizon = action_horizon
        self._ctx = zmq.Context()
        self._sock = None
        self._open_socket()
        self.last_infer_time_ms = None   # server-side inference time of the last predict

    @property
    def addr(self) -> str:
        return self._addr

    def _open_socket(self) -> None:
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)   # closing never blocks on an unanswered request
        sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        sock.connect(self._addr)
        self._sock = sock

    def _close_socket(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None

    def _call(self, msg: dict) -> dict:
        if self._sock is None:
            self._open_socket()
        payload = msgpack.packb(msg, use_bin_type=True)
        try:
            self._sock.send(payload)
            raw = self._sock.recv()
        except zmq.Again as e:
            # The REQ socket is now stuck waiting for this reply: replace it before raising.
            self._close_socket()
            self._open_socket()
            raise TimeoutError(f"no reply from N0-VTLA server {self._addr} within {self._timeout_ms / 1000:.1f}s "
                               f"(cmd={msg.get('cmd')}); REQ socket recreated") from e
        except zmq.ZMQError:
            self._close_socket()
            self._open_socket()
            raise
        reply = msgpack.unpackb(raw, raw=False)
        if not isinstance(reply, dict) or reply.get("status") != "ok":
            message = reply.get("message") if isinstance(reply, dict) else reply
            raise RuntimeError(f"N0-VTLA server error (cmd={msg.get('cmd')}): {message}")
        return reply

    def reset(self) -> None:
        """Start an episode: the server drops its tactile baseline and re-captures it on the next predict."""
        self._call({"cmd": "reset"})

    def predict(self, state, prompt: str | None, cams: dict, tacs: dict) -> np.ndarray:
        """One inference round trip -> (action_horizon, action_dim) float32 joint targets, (16, 16) here.

        Raises on a reply that is not this config's (see the module docstring): a server for another
        config or checkpoint would otherwise be executed silently, e.g. a 50-step chunk cut to its
        first 16 rows, which drops the gripper-close tail.
        """
        reply = self._call(build_predict_request(state, prompt, cams, tacs, self._state_dim))
        actions = np.asarray(reply.get("actions"), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] != self._action_horizon or actions.shape[1] < self._action_dim:
            raise RuntimeError(f"bad actions from server: shape {actions.shape}, expected ({self._action_horizon}, "
                               f">={self._action_dim}); is this port serving another config or checkpoint?")
        if not np.isfinite(actions).all():
            raise RuntimeError("server returned non-finite actions")
        if np.any(actions[:, self._action_dim:] != 0):
            raise RuntimeError(f"server returned non-zero columns past {self._action_dim}: not a "
                               f"{self._action_dim}-dim joint policy (another config on this port?)")
        if self._action_dim > ARM_DIM and not np.any(actions[:, ARM_DIM:self._action_dim]):
            raise RuntimeError(f"arm B columns {ARM_DIM}..{self._action_dim - 1} are all exactly 0: a single-arm "
                               f"({ARM_DIM}-dim) server? Executing them would drive arm B to q=0, gripper closed")
        self.last_infer_time_ms = reply.get("infer_time_ms")
        return actions[:, :self._action_dim]

    def wait_for_server(self, max_wait_s: float = 600.0) -> None:
        """Block until the server answers a reset (it may still be loading the checkpoint)."""
        deadline = time.monotonic() + max_wait_s
        while True:
            try:
                self.reset()
                return
            except TimeoutError:
                if time.monotonic() >= deadline:
                    raise
                print(f"[n0vtla-zmq] still waiting for server at {self._addr} ...", flush=True)

    def close(self) -> None:
        self._close_socket()
        if not self._ctx.closed:
            self._ctx.term()
