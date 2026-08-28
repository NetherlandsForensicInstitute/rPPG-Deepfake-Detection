from abc import ABC, abstractmethod
from itertools import islice
from typing import Iterable, Optional
from collections import Counter

from src.datasets.instance import Instance


class Dataset(ABC, Iterable[Instance]):
    """
    An abstract dataset class that other datasets should inherit from.
    """

    def __init__(self, dataset_name: Optional[str] = "unspecified_dataset"):
        """
        :param dataset_name: The name of the dataset (optional).
        """
        self.dataset_name = dataset_name

    @abstractmethod
    def __iter__(self):
        """Iterates over the dataset, yielding instances. Should be implemented by subclasses."""
        raise NotImplementedError

    def __eq__(self, other):
        return set(self.__iter__()) == set(other.__iter__())

    def iter(self, batch_size: int):
        iterator = iter(self)
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            yield batch

    def count_labels(self, label: Optional[str] = None) -> Counter:
        """
        Returns a Counter containing counts for every value of a given label.

        :param label: The label to count. Must be a key in 'Annotation'.
                      If not provided, total number of samples will be returned.
        """
        counts = Counter(
            instance.annotation.get_label(label)
            if (instance.annotation and label is not None)
            else "<no label>"
            for instance in self
        )
        return counts

    def info(self, label: Optional[str] = None) -> str:
        """
        Returns a string representing the number of samples (per label) in the dataset.

        :param label: The attribute to count. Must be a key in 'Annotation'.
                      If not provided, total number of samples will be returned.
        """
        # Calculate counts
        counts = self.count_labels(label)

        # Format the header
        header = f"Dataset: {self.dataset_name}\n"
        separator = "-" * (len(header) - 1) + "\n"

        # Create lines for each label, sorted by count (descending)
        stats = (
            "\n".join([f"{label}: {count}" for label, count in counts.most_common()])
            + "\n"
        )

        return f"{header}{separator}{stats}{separator}"


class MapStyleDatasetMixin(ABC):
    """
    Extension of the regular dataset class that adds the __getitem__ and __len__ methods.
    In addition, it automatically provides an implementation for the __iter__ method.
    """

    @abstractmethod
    def __getitem__(self, idx: int) -> Instance:
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
