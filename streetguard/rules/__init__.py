"""
Surveillance analytic rules module for StreetGuard AI.
Contains rule implementations for intrusion and unattended object detection.
"""

from streetguard.rules.intrusion import IntrusionAlert, RestrictedAreaIntrusionDetector
from streetguard.rules.unattended import UnattendedAlert, UnattendedObjectDetector

__all__ = [
    "IntrusionAlert",
    "RestrictedAreaIntrusionDetector",
    "UnattendedAlert",
    "UnattendedObjectDetector",
]
