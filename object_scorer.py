import os
import sys
import json
import numpy as np

from typing import List, Dict
from pathlib import Path

# Append paths to sys.path for importing modules from other directories
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Import from GroundingDINO
from letsdoit.utils.object_instance import ObjectInstance
from letsdoit.scoring.primitive import SpatialPrimitive, SpatialPrimitivePair, get_primitive
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Import from segment_anything
from segment_anything import (
    sam_hq_model_registry,
    SamPredictor
)

class ObjectScorer:
    def __init__(self, scene_objects: List[ObjectInstance], instructions_path: str | Path):
        pass

    def read_instructions(path: str | Path) -> List[Dict]:
        with open(path, 'r') as file:
            instructions = json.load(file)
        return instructions