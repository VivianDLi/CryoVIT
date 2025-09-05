"""Implementation of the single sample data module."""

import pandas as pd

from cryovit.datamodules.base_datamodule import BaseDataModule


class SingleSampleDataModule(BaseDataModule):
    """Data module for CryoVIT experiments involving a single sample."""

    def __init__(
        self,
        sample: list[str],
        split_id: int | None,
        split_key: str,
        test_sample: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Create a datamodule for training and testing on a single sample.

        Args:
            sample (list[str]): The sample to train on.
            split_id (Optional[int]): An optional split_id to validate with.
            split_key (str): The key used to select splits using split_id.
            test_sample (Optional[list[str]]): The sample to test on. Should be equal to sample or None.
        """
        super().__init__(**kwargs)
        # Validity checks
        assert (
            len(sample) == 1
        ), f"Single sample 'sample' should be a single string list. Got {sample} instead."
        assert (
            test_sample is None or len(test_sample) == 1
        ), f"Single sample 'test_sample' should be a single string list or None. Got {test_sample} instead."

        self.sample = sample[0]
        self.split_id = split_id
        self.split_key = split_key
        self.test_sample = (
            test_sample[0] if test_sample is not None else test_sample
        )

    def train_df(self) -> pd.DataFrame:
        """Train tomograms: exclude those from the sample with the specified split_id.

        Returns:
            pd.DataFrame: A dataframe specifying the train tomograms.
        """
        assert self.record_df is not None
        if self.split_id is not None:
            df = self.record_df[
                (self.record_df[self.split_key] != self.split_id)
                & (self.record_df["sample"] == self.sample)
            ]
        else:
            df = self.record_df[self.record_df["sample"] == self.sample][
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
            & (self.record_df["sample"] == self.sample)
        ]

    def test_df(self) -> pd.DataFrame:
        """Test tomograms: test on tomograms with the specified split_id.

        Returns:
            pd.DataFrame: A dataframe specifying the test tomograms.
        """
        assert self.record_df is not None
        if self.test_sample is None:
            return self.val_df()

        # If testing on another sample, use the whole sample
        return self.record_df[self.record_df["sample"] == self.test_sample][
            ["sample", "tomo_name"]
        ]

    def predict_df(self) -> pd.DataFrame:
        assert self.record_df is not None
        return self.record_df[self.record_df["sample"] == self.sample][
            ["sample", "tomo_name"]
        ]
