"""Visuo-tactile conversion: gel images -> dense three-axis force fields.

Turns raw tactile camera frames into the force-field representation NeoForce trains on.
"""

from .conversion import TactileConversionModel, correct_frame

__all__ = ["TactileConversionModel", "correct_frame"]
