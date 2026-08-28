import hashlib
import os
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Optional

from PIL import Image
import cv2
import copy

from src.datasets.annotation import Annotation

def hash_image_to_int(image: Image.Image) -> int:
    return int(hashlib.md5(image.tobytes()).hexdigest(), 16)


class Instance(ABC):
    """
    A single dataset instance. Each instance can have one or more optional labels.

    :param annotation: An annotation for the instance.
    :param meta: an optional mapping containing extra metadata about the instance.
    """

    def __init__(
        self, annotation: Optional[Annotation] = None, meta: Optional[dict] = None
    ):
        self.annotation = annotation
        self.meta = meta

    def __eq__(self, other):
        if not isinstance(other, Instance):
            return ValueError("Other object is not of type Instance.")
        return self.__hash__() == other.__hash__()

    def deepcopy(self):
        return copy.deepcopy(self)

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> Path:
        """
        Saves an instance to disk.

        :param path: The path to save the instance to.
        :return: The path to the saved instance.
        """
        raise NotImplementedError


class ImageInstance(Instance):
    """
    A default image instance is initialized with only image data and an optional label.

    :param data: The image data in the form of a PIL Image.
    :param annotation: An annotation for the instance.
    :param meta: an optional mapping containing extra metadata about the instance.
    """

    def __init__(
        self,
        data: Image,
        annotation: Optional[Annotation] = None,
        meta: Optional[dict] = None,
    ):
        super().__init__(annotation, meta)
        self.data = data

    def __hash__(self):
        return hash_image_to_int(self.data)

    def save(self, path: Path) -> Path:
        if not isinstance(path, Path):
            raise ValueError("Path must be of type Path.")
        if "." not in str(path):
            path = path.with_suffix(".png")
        self.data.save(path)
        return path


class FileImageInstance(Instance):
    """
    An image instance that is created from an image file stored on disk.

    :param path: The image path.
    :param annotation: An annotation for the instance.
    :param meta: an optional mapping containing extra metadata about the instance.
    """

    def __init__(
        self,
        path: str,
        annotation: Optional[Annotation] = None,
        meta: Optional[dict] = None,
    ):
        super().__init__(annotation, meta)
        self.path = Path(path)

    @cached_property
    def data(self):
        return Image.open(self.path).convert("RGB")

    def __hash__(self):
        return hash(self.path)

    def save(self, path: Path) -> Path:
        if not isinstance(path, Path):
            raise ValueError("Path must be of type Path.")
        # Save as png by default when no suffix provided
        if "." not in str(path):
            path = path.with_suffix(".png")
        self.data.save(path)
        return path


class FileImageSequenceInstance(Instance):
    """
    An instance that is created from a sequence of images stored on disk.

    :param path: The path to the image sequence directory.
    :param annotation: An annotation for the instance.
    :param meta: an optional mapping containing extra metadata about the instance.
    """

    def __init__(
        self,
        path: str,
        annotation: Optional[Annotation] = None,
        meta: Optional[dict] = None,
    ):
        super().__init__(annotation, meta)
        self.path = Path(path)

    @cached_property
    def data(self) -> list[FileImageInstance]:
        return [
            FileImageInstance(os.path.join(self.path, img), self.annotation)
            for img in os.listdir(self.path)
        ]

    def __len__(self):
        return len(self.data)

    def __hash__(self):
        return hash(self.path)

    def save(self, path: Path) -> Path:
        if not isinstance(path, Path):
            raise ValueError("Path must be of type Path.")
        path.mkdir(parents=True, exist_ok=True)
        for img in self.data:
            img.save(Path(os.path.join(path, img.path.name)))
        return path


class FileVideoInstance(Instance):
    """
    An instance that is created from a video stored on disk.

    :param path: The path to the video file.
    :param annotation: An annotation for the instance.
    :param meta: an optional mapping containing extra metadata about the instance.
    """

    def __init__(
        self,
        path: str,
        annotation: Optional[Annotation] = None,
        meta: Optional[dict] = None,
    ):
        super().__init__(annotation, meta)
        self.path = Path(path)

    @cached_property
    def data(self):
        return cv2.VideoCapture(str(self.path))

    def __hash__(self):
        return hash(self.path)

    def save(self, path: Path) -> Path:
        # TODO: implement video saving in the future
        raise NotImplementedError
