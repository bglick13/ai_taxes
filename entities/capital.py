import numpy as np
from ai_economist.foundation.entities.landmarks import landmark_registry, Landmark


@landmark_registry.add
class Capital(Landmark):
    """Capital Landmark"""
    # TODO: This
    name = "Capital"
    color = np.array([220, 20, 220]) / 255.0
    ownable = False
    solid = False
