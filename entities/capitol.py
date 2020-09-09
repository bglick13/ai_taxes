import numpy as np
from ai_economist.foundation.entities.landmarks import landmark_registry, Landmark


@landmark_registry.add
class Capitol(Landmark):
    """Capitol Landmark"""
    # TODO: This
    name = "Capitol"
    color = np.array([220, 20, 220]) / 255.0
    ownable = False
    solid = False
