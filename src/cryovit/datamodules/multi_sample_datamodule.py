"""Implementation of the multi sample data module."""

import pandas as pd

from cryovit.datamodules.base_datamodule import BaseDataModule


class MultiSampleDataModule(BaseDataModule):
    """Data module for CryoVIT experiments involving multiple samples."""

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
            sample (list[str]): list of samples used for training.
            split_id (Optional[int]): An optional split ID for validation.
            split_key (str): The key used to select splits using split_id.
            test_sample (Optional[list[str]]): list of samples used for testing. If None, test on the validation set.
        """

        super().__init__(**kwargs)
        # Validity checks
        assert isinstance(
            sample, list
        ), f"Multi sample 'sample' should be a list. Got {sample} instead."
        assert test_sample is None or isinstance(
            test_sample, list
        ), f"Multi sample 'test_sample' should be None or a list. Got {test_sample} instead."

        self.sample = sample
        self.split_id = split_id
        self.split_key = split_key
        self.test_sample = test_sample

    def train_df(self) -> pd.DataFrame:
        """Train tomograms: exclude those with the specified split_id.

        Returns:
            pd.DataFrame: A dataframe specifying the train tomograms.
        """

        assert self.record_df is not None
        if self.split_id is not None:
            df = self.record_df[
                (self.record_df[self.split_key] != self.split_id)
                & (self.record_df["sample"].isin(self.sample))
            ]
        else:
            df = self.record_df[self.record_df["sample"].isin(self.sample)][
                ["sample", "tomo_name"]
            ]

        return df

    def val_df(self) -> pd.DataFrame:
        """Validation tomograms: optionally validate on tomograms with the specified split_id.

        Returns:
            pd.DataFrame: A dataframe specifying the validation tomograms.
        """

        assert self.record_df is not None
        if self.split_id is None:  # validate on train set
            return self.train_df()

        return self.record_df[
            (self.record_df[self.split_key] == self.split_id)
            & (self.record_df["sample"].isin(self.sample))
        ]

    def test_df(self) -> pd.DataFrame:
        """Test tomograms: test on tomograms from the test samples.

        Returns:
            pd.DataFrame: A dataframe specifying the test tomograms.
        """

        assert self.record_df is not None
        if self.test_sample is None:
            return self.val_df()

        return self.record_df[
            (self.record_df["sample"].isin(self.test_sample))
        ][["sample", "tomo_name"]]

    def predict_df(self) -> pd.DataFrame:
        """Predict tomograms: predict on the specified samples.

        Returns:
            pd.DataFrame: A dataframe specifying the predict tomograms.
        """

        assert self.record_df is not None
        return self.record_df[self.record_df["sample"].isin(self.sample)][
            ["sample", "tomo_name"]
        ]
