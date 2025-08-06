"""Implementation of the fractional leave one out data module."""

from typing import List, Optional, Union

import pandas as pd

from cryovit.config import samples
from cryovit.datamodules.base_datamodule import BaseDataModule


class FractionalSampleDataModule(BaseDataModule):
    """Data module for fractional leave-one-out CryoVIT experiments."""

    def __init__(self, sample: Union[str, List[str]], split_id: Optional[int], split_key: Optional[str], test_sample: Optional[Union[str, List[str]]] = None, **kwargs) -> None:
        """Train on a fraction of tomograms and leave out one sample for evaluation.

        Args:
            sample (Union[str, List[str]]): The sample to excluded from training and used for testing.
            split_id (Optional[int]): The number of splits used for training. If None, defaults to all splits.
            split_key (str): The key used to select splits using split_id.
            test_sample (Optional[Union[str, List[str]]]): The sample to test on. Should be None.
        """
        super(FractionalSampleDataModule, self).__init__(**kwargs)
        self.sample = sample
        self.split_id = split_id
        self.split_key = split_key
        self.test_sample = test_sample
        
        # Validity checks
        assert isinstance(self.sample, str), f"Fractional sample 'sample' should be a single string. Got {self.sample} instead."
        assert test_sample is None, f"Fractional sample 'test_sample' should be None. Got {self.test_sample} instead."

    def train_df(self) -> pd.DataFrame:
        """Train tomograms: include a subset of all splits, leaving out one sample.

        Returns:
            pd.DataFrame: A dataframe specifying the train tomograms.
        """
        if self.split_id is not None:
            training_splits = list(range(self.split_id))
        else:
            training_splits = list(range(self.record_df[self.split_key].max()))
        
        return self.record_df[
            (self.record_df[self.split_key].isin(training_splits))
            & (self.record_df["sample"] != self.sample)
            & (self.record_df["sample"].isin(samples))
        ][["sample", "tomo_name"]]

    def val_df(self) -> pd.DataFrame:
        """Validation tomograms: validate on the train tomograms. Not really useful.

        Returns:
            pd.DataFrame: A dataframe specifying the validation tomograms.
        """
        return self.train_df()  # validate on train set

    def test_df(self) -> pd.DataFrame:
        """Test tomograms: test on tomograms from the held out sample.

        Returns:
            pd.DataFrame: A dataframe specifying the test tomograms.
        """
        return self.record_df[self.record_df["sample"] == self.sample][["sample", "tomo_name"]]
    
    def predict_df(self) -> pd.DataFrame:
        raise NotImplementedError
