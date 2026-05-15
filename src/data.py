"""Load and process data sessions for training and testing."""

from pathlib import Path
import pandas as pd


def ensure_dataframe_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure anomaly label columns exist in dataframe.
    """

    df_ensured = df.copy()

    if "anomaly_level" not in df_ensured.columns:
        df_ensured["anomaly_level"] = 0

    if "is_injected_anomaly" not in df_ensured.columns:
        df_ensured["is_injected_anomaly"] = 0

    if "anomaly_type" not in df_ensured.columns:
        df_ensured["anomaly_type"] = "normal"

    return df_ensured


def extract_numeric_session_id(filepath: str | Path) -> int:
    """
    Extract numeric session id from filename.

    Example:
    data_session_16_2026...
    -> 16
    """

    return int(Path(filepath).stem.split("_")[2])


def extract_augmented_session_id(filepath: str | Path) -> str:
    """
    Extract augmented session id from filename.

    Example:
    data_session_16_F1_12-0.3...
    -> 16_F1
    """

    parts = Path(filepath).stem.split("_")

    return "_".join(parts[2:4])


def load_dataset_dict(
    folder_path: str | Path,
    id_extractor,
    exclude_last_n: int = 0,
) -> dict:
    """
    Load all CSV files from a folder into a dictionary.

    Parameters
    ----------
    folder_path : str | Path
        Folder containing csv files.

    id_extractor : callable
        Function used to extract dictionary key from filename.

    exclude_last_n : int
        Number of files to exclude from the end.

    Returns
    -------
    dict
        Dictionary containing loaded dataframes.
    """

    folder_path = Path(folder_path)

    files = sorted(
        folder_path.glob("*.csv"),
        key=id_extractor
    )

    if exclude_last_n > 0:
        files = files[:-exclude_last_n]

    dataset_dict = {}

    for file in files:

        dict_id = id_extractor(file)

        df = pd.read_csv(
            file,
            index_col="timestamp",
            parse_dates=True
        )

        dataset_dict[dict_id] = ensure_dataframe_label_columns(df)

    return dataset_dict


def load_all_datasets(data_path: str | Path):
    """
    Load train, severe and small anomaly datasets.
    """

    data_path = Path(data_path)

    train_dict = load_dataset_dict(
        data_path / "clean",
        id_extractor=extract_numeric_session_id,
        exclude_last_n=2,
    )

    test_severe_dict = load_dataset_dict(
        data_path / "augmented" / "severe",
        id_extractor=extract_augmented_session_id,
    )

    test_small_dict = load_dataset_dict(
        data_path / "augmented" / "small",
        id_extractor=extract_augmented_session_id,
    )

    return train_dict, test_severe_dict, test_small_dict