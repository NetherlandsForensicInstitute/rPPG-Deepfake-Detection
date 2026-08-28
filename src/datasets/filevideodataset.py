import os
from pathlib import Path
from typing import List

from src.datasets.annotation import Annotation
from src.datasets.dataset import Dataset, MapStyleDatasetMixin
from src.datasets.instance import FileVideoInstance, Instance


class FileVideoDataset(MapStyleDatasetMixin, Dataset):
    """
    This dataset loads a dataset of videos from a filesystem.
    The dataset should be stored on the filesystem as follows:

    <root dataset dir>
        real
            <label 1>
                - video1
        fake
            <label 2>
                - video2
        ...

    Non-video files are ignored.

    :param path: The path to the root folder of the dataset.
    :param dataset_name: The name of the dataset.
    :param split_file: The path to a file containing the filenames of the videos that should be returned.
    """

    def __init__(self, path: str, dataset_name: str = None, split_file: str = None, label_mapping: dict = None):
        super().__init__(dataset_name)
        self.path = path

        # If split file provided store the filenames of included instances in a list
        if split_file:
            with open(split_file, "r") as f:
                self.included_instances = f.read().splitlines()
        else:
            self.included_instances = None

        # Store instance paths
        self.instance_paths = self._index()

        self.label_mapping = label_mapping

    def _index(self) -> List[Path]:
        """
        Indexes all files in the dataset and returns a list of filepaths.
        """
        # Loop over folders (labels) in dataset
        paths = []
        # Loop over folders (authenticity class) in dataset
        for folder in os.listdir(self.path):
            # If directory
            if os.path.isdir(os.path.join(self.path, folder)):
                # Loop over folders (models) in dataset
                for subfolder in os.listdir(os.path.join(self.path, folder)):
                    # If directory
                    if os.path.isdir(os.path.join(self.path, folder, subfolder)):
                        # Loop over videos
                        for video in os.listdir(
                            os.path.join(self.path, folder, subfolder)
                        ):
                            if video.split(".")[-1].lower() in ["mp4", "mov"]:
                                if (
                                    self.included_instances is None
                                    or video in self.included_instances
                                ):
                                    paths.append(
                                        Path(
                                            os.path.join(
                                                self.path, folder, subfolder, video
                                            )
                                        )
                                    )
                            else:
                                pass
                                print(
                                    "Found file that is not a mp4 or mov file: {}".format(
                                        video
                                    )
                                )
        return paths

    def __getitem__(self, idx: int) -> Instance:
        # Get instance path
        path = self.instance_paths[idx]

        # Get labels from path
        authenticity_label, source_label, img_name = path.parts[-3:]

        # Return instance
        return FileVideoInstance(
            str(path),
            Annotation({"authenticity": authenticity_label, "source": source_label}),
        )

    def __len__(self):
        return len(self.instance_paths)
