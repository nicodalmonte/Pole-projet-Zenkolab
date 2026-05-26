"""Dataset classes for glaucoma classification."""

from __future__ import annotations

from src.datasets.ACRIMA import ACRIMADataset
from src.datasets.Fundus_Train_Val_Data import FundusTrainValDataset
from src.datasets.Harvard import HarvardGlaucomaDataset
from src.datasets.LAG import LAGDataset
from src.datasets.ORIGA import ORIGADataset
from src.datasets.REFUGE2 import REFUGE2Dataset
from src.datasets.JRAIGS import JRAIGSDataset
from src.datasets.G1020 import G1020Dataset
from src.datasets.RIMONE import RIMONEDataset
from src.datasets.AIROGSLight import AIROGSLightDataset

__all__ = [
    "ACRIMADataset",
    "FundusTrainValDataset",
    "HarvardGlaucomaDataset",
    "LAGDataset",
    "ORIGADataset",
    "REFUGE2Dataset",
    "JRAIGSDataset",
    "G1020Dataset",
    "RIMONEDataset",
    "AIROGSLightDataset",
]
