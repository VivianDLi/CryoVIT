"""Implementation of the fractional leave one out data module."""

import pandas as pd

from cryovit.datamodules.base_datamodule import BaseDataModule


class FractionalSampleDataModule(BaseDataModule):
    """Data module for fractional leave-one-out CryoVIT experiments."""

    def __init__(
        self,
        sample: list[str],
        split_id: int | None,
        split_key: str | None,
        test_sample: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Train on a fraction of tomograms and leave out one sample for evaluation.

        Args:
            sample (list[str]): The samples to train and test on.
            split_id (Optional[int]): The number of splits used for training. If None, defaults to all splits.
            split_key (str): The key used to select splits using split_id.
            test_sample (Optional[list[str]]): The sample to exclude from training and use for testing.
        """
        super().__init__(**kwargs)
        # Validity checks
        assert (
            test_sample is not None
        ), "Fractional sample `test_sample` cannot be None."
        assert (
            len(test_sample) == 1
        ), f"Fractional sample 'test_sample' should be a single string list. Got {test_sample} instead."

        self.sample = sample
        self.split_id = split_id
        self.split_key = split_key
        self.test_sample = test_sample

    def train_df(self) -> pd.DataFrame:
        """Train tomograms: include a subset of all splits, leaving out one sample.

        Returns:
            pd.DataFrame: A dataframe specifying the train tomograms.
        """
        assert self.record_df is not None
        if self.split_id is not None:
            training_splits = list(range(self.split_id))
        else:
            training_splits = list(range(self.record_df[self.split_key].max()))

        return self.record_df[
            (self.record_df[self.split_key].isin(training_splits))
            & (self.record_df["sample"].isin(self.sample))
            & ~self.record_df["sample"].isin(self.test_sample)
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
        assert self.record_df is not None
        return self.record_df[self.record_df["sample"].isin(self.test_sample)][
            ["sample", "tomo_name"]
        ]

    def predict_df(self) -> pd.DataFrame:
        assert self.record_df is not None
        return self.record_df[self.record_df["sample"].isin(self.sample)][
            ["sample", "tomo_name"]
        ]
