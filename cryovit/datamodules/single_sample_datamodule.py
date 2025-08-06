"""Implementation of the single sample data module."""

from typing import List, Optional, Union

import pandas as pd

from cryovit.config import Sample
from cryovit.datamodules.base_datamodule import BaseDataModule


class SingleSampleDataModule(BaseDataModule):
    """Data module for CryoVIT experiments involving a single sample."""

    def __init__(self, sample: Union[str, List[str]], split_id: Optional[int], split_key: str, test_sample: Optional[Union[str, List[str]]] = None, **kwargs) -> None:
        """Create a datamodule for training and testing on a single sample.

        Args:
            sample (Union[str, List[str]]): The sample to train on.
            split_id (Optional[int]): An optional split_id to validate with.
            split_key (str): The key used to select splits using split_id.
            test_sample (Optional[Union[str, List[str]]]): The sample to test on. Should be equal to sample or None.
        """
        super(SingleSampleDataModule, self).__init__(**kwargs)
        self.sample = sample
        self.split_id = split_id
        self.split_key = split_key
        self.test_sample = test_sample
        
        # Validity checks
        assert isinstance(self.sample, str), f"Single sample 'sample' should be a single string. Got {self.sample} instead."
        assert self.test_sample is None or isinstance(self.test_sample, str), f"Single sample 'test_sample' should be a single string or None. Got {self.test_sample} instead."

    def train_df(self) -> pd.DataFrame:
        """Train tomograms: exclude those from the sample with the specified split_id.

        Returns:
            pd.DataFrame: A dataframe specifying the train tomograms.
        """
        if self.split_id is not None:
            df = self.record_df[(self.record_df[self.split_key] != self.split_id) & (self.record_df["sample"] == self.sample)]
        else:
            df = self.record_df[self.record_df["sample"] == self.sample][["sample", "tomo_name"]]
        return df

    def val_df(self) -> pd.DataFrame:
        """Validation tomograms: optionally validate on tomograms with the specified split_id.

        Returns:
            pd.DataFrame: A dataframe specifying the validation tomograms.
        """
        if self.split_id is None:  # validate on train set
            return self.train_df()

        return self.record_df[
            (self.record_df[self.split_key] == self.split_id)
            & (self.record_df["sample"] == self.sample)
        ]

    def test_df(self) -> pd.DataFrame:
        """Test tomograms: test on tomograms with the specified split_id.

        Returns:
            pd.DataFrame: A dataframe specifying the test tomograms.
        """
        if self.test_sample is None:
            return self.val_df()

        # If testing on another sample, use the whole sample
        return self.record_df[self.record_df["sample"] == self.test_sample][["sample", "tomo_name"]]

    def predict_df(self) -> pd.DataFrame:
        return self.record_df[self.record_df["sample"] == self.sample][["sample", "tomo_name"]]