"""Implementation of the leave one out data module."""

from typing import List, Union
from typing import Optional

import pandas as pd

from cryovit.config import samples
from cryovit.datamodules.base_datamodule import BaseDataModule


class LOOSampleDataModule(BaseDataModule):
    """Data module for CryoVIT experiments leaving out one sample."""

    def __init__(self, sample: Union[str, List[str]], split_id: Optional[int], split_key: Optional[str], test_sample: Optional[Union[str, List[str]]] = None, **kwargs) -> None:
        """Train on a fraction of tomograms and leave out one sample for evaluation.

        Args:
            sample (Union[str, List[str]]): The sample to excluded from training and used for testing.
            split_id (Optional[int]): An optional split ID for validation.
            split_key (str): The key used to select splits using split_id.
            test_sample (Optional[Union[str, List[str]]]): The sample to test on. Should be None.
        """
        super(LOOSampleDataModule, self).__init__(**kwargs)
        self.sample = sample
        self.split_id = split_id
        self.split_key = split_key
        self.test_sample = test_sample
        
        # Validity checks
        assert isinstance(self.sample, str), f"LOO sample 'sample' should be a single string. Got {self.sample} instead."
        assert test_sample is None, f"LOO sample 'test_sample' should be None. Got {self.test_sample} instead."

    def train_df(self) -> pd.DataFrame:
        """Train tomograms: exclude those with the specified split_id and sample.

        Returns:
            pd.DataFrame: A dataframe specifying the train tomograms.
        """
        if self.split_id is not None:
            df = self.record_df[(self.record_df[self.split_key] != self.split_id) & (self.record_df["sample"] != self.sample) & (self.record_df["sample"].isin(samples))]
        else:
            df = self.record_df[(self.record_df["sample"] != self.sample) & (self.record_df["sample"].isin(samples))][["sample", "tomo_name"]]
        
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
            & (self.record_df["sample"] != self.sample)
            & (self.record_df["sample"].isin(samples))
        ]

    def test_df(self) -> pd.DataFrame:
        """Test tomograms: test on tomograms from the held out sample.

        Returns:
            pd.DataFrame: A dataframe specifying the test tomograms.
        """
        return self.record_df[self.record_df["sample"] == self.sample][["sample", "tomo_name"]]

    def predict_df(self) -> pd.DataFrame:
        raise NotImplementedError