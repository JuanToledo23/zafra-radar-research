"""Harvest and burn detectors: rule-based baseline and intensity-ratio test (ex. 4), trained
classifier (ex. 4b), cane / non-cane gate (ex. 4c, cane_gate)."""

from zafra.detect.ratio_test import GcovPixelProvider, detect_ratio
from zafra.detect.rule import detect_rule

__all__ = ["GcovPixelProvider", "detect_ratio", "detect_rule"]
