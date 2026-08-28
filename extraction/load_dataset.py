import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.annotation import Annotation
from src.datasets.filevideodataset import FileVideoDataset
from src.datasets.dataset import Dataset
from src.datasets.instance import FileVideoInstance

VIDEO_EXTENSIONS = ('.mp4', '.avi')


class GroundTruthDataset(Dataset):
    """Dataset of ground-truth experiment recordings organised by participant sub-folder."""

    def __init__(self, path: str, name: str):
        super().__init__(name=name)
        self.path = path

    def __iter__(self):
        for participant in os.listdir(self.path):
            participant_path = os.path.join(self.path, participant)
            if os.path.isdir(participant_path):
                for filename in os.listdir(participant_path):
                    if filename.lower().endswith(VIDEO_EXTENSIONS):
                        yield FileVideoInstance(
                            path=os.path.join(participant_path, filename),
                            annotation=Annotation({"authenticity": participant}),
                        )


class UBFCDataset(Dataset):
    """UBFC-rPPG benchmark dataset (DATASET_1 or DATASET_2)."""

    def __init__(self, path: str, name: str):
        super().__init__(dataset_name=name)
        self.path = path

    def __iter__(self):
        for subject in os.listdir(self.path):
            yield FileVideoInstance(
                path=os.path.join(self.path, subject, 'vid.avi'),
                annotation=Annotation({"subject": subject}),
            )


def load_dataset(dataset_name: str, path: str) -> Dataset:
    """Load a dataset by name from the given root path.

    Args:
        dataset_name: One of the supported dataset identifiers listed below.
        path: Root directory of the dataset on disk.

    Supported dataset names
    -----------------------
    ubfc1
        UBFC-rPPG DATASET_1.
    ubfc2
        UBFC-rPPG DATASET_2.
    ground-truth
        In-house ground-truth recording dataset.
    Anything else
        Generic file-based dataset with folder structure <authenticity>/<source>/<video>.
    """
    name_lower = dataset_name.lower()

    if name_lower == 'ubfc1':
        return UBFCDataset(name='UBFC1', path=path)

    if name_lower == 'ubfc2':
        return UBFCDataset(name='UBFC2', path=path)

    if name_lower == 'ground-truth':
        return GroundTruthDataset(name=dataset_name, path=path)

    # Generic file-based dataset (folder structure: <authenticity>/<source>/<video>)
    return FileVideoDataset(dataset_name=dataset_name, path=path)