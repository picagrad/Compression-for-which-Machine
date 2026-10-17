"""Compression for which Machine: the feature codec and the training / testing code that reproduce the paper.

The split task models come from the CFM-Model-Zoo library (``cfm_model_zoo``).
"""
# The library is imported FIRST. Its __init__ patches transformers for GLIP/GroundingDINO, which must happen
# before any mmdet/mmseg module is imported (codecs/codecs.py imports mmdet's PatchEmbed).
import cfm_model_zoo  # noqa: F401

__version__ = '0.1.0'   # equal to pyproject.toml's version; the tag every MLflow run of this repo carries
