import os
from abc import ABC, abstractmethod
from typing import Any, Optional
from pathlib import Path

import numpy as np
from dask.distributed import Client
from osl_ephys.preprocessing import run_proc_batch


class Preprocessing(ABC):
    """
    Base class for preprocessing ephys data.

    Attributes:
        data_path: Path to the data directory.
        osl_config: OSL config string.
        preproc_config: Dictionary containing the preprocessing configuration.
        n_workers: Number of workers for parallel processing.
        chunk_seconds: Length of each chunk in seconds.
        save_folder: Path to the save directory.
        outdir_path: Path to the output directory.
        batch_args: Dictionary containing the batch arguments.
        extra_funcs: List of extra functions to apply to the data.
        delete_fif: Whether to delete the .fif file after processing.
    """

    def __init__(
        self,
        data_path: str,
        log_dir: Optional[str] = None,
        save_path: Optional[str] = None,
        osl_config: Optional[str] = None,
        preproc_config: Optional[dict[str, Any]] = None,
        n_workers: int = 1,
        delete_fif: bool = True,
        chunk_seconds: int = 60,
        skip_done: bool = False,
    ) -> None:
        """
        Base class for preprocessing ephys data.

        Args:
            data_path: Path to the data directory.
            config: OSL config string.
            n_workers: Number of workers for parallel processing.
            chunk_seconds: Length of each chunk in seconds.
        """
        self.data_path = data_path
        self.preproc_config = preproc_config
        self.delete_fif = delete_fif
        self.n_workers = n_workers
        self.batch_args = {}
        self.chunk_seconds = chunk_seconds
        self.skip_done = skip_done

        # load osl config from file if provided
        self.osl_config = osl_config

        # Set save folder path
        default_save_path = os.path.join(os.path.dirname(data_path), "preprocessed")
        self.save_folder = save_path if save_path is not None else default_save_path
        self.log_dir = log_dir if log_dir is not None else self.save_folder

        # Set extra_funcs for this class based on preproc_config
        self.extra_funcs = []
        if self.preproc_config is not None:
            self.extra_funcs = [
                getattr(self, name) for name in self.preproc_config.keys()
            ]

        self.load()

    @abstractmethod
    def extract_raw(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Extract raw data from the data dictionary.
        """
        return NotImplementedError

    @abstractmethod
    def load(self) -> dict[str, np.ndarray]:
        """
        Should populate self.batch_args with the following keys:
        - files (str or list or mne.Raw) - Can be a list of Raw objects or a list of
        filenames (or .ds dir names if CTF data) or a path to a textfile list of
        filenames (or .ds dir names if CTF data).
        - subjects (list of str) - Subject directory names. These are sub-directories
        in outdir.
        """
        return NotImplementedError

    def preprocess_stage_1(self) -> None:
        """
        Stage 1: Run OSL preprocessing pipeline.
        """
        client = Client(threads_per_worker=1, n_workers=self.n_workers)

        print(f"INFO: Preprocessing {self.data_path}")
        # paths for logs and reports
        print(f"INFO: Will save preprocessed data to {self.save_folder}")
        logsdir = os.path.join(self.log_dir, "logs")
        reportdir = os.path.join(self.save_folder, "reports")

        # make these dirs
        os.makedirs(self.save_folder, exist_ok=True)
        os.makedirs(logsdir, exist_ok=True)
        os.makedirs(reportdir, exist_ok=True)

        # if skip_done is true, filter files and only keep those that
        # don't already exist in self.save_folder
        # apply same filter to subjects
        if self.skip_done:
            total_files = len(self.batch_args["subjects"])
            save_folder = Path(self.save_folder)
            files, subjects = [], []

            num_kept = 0
            for f, s in zip(self.batch_args["files"], self.batch_args["subjects"]):
                if not (save_folder / Path(s).name).exists():
                    files.append(f)
                    subjects.append(s)
                    num_kept += 1

            self.batch_args["files"] = files
            self.batch_args["subjects"] = subjects

        print(f"INFO: Kept {num_kept} files out of {total_files}")

        # Stage 1: Run OSL preprocessing pipeline
        print("Stage 1: Running OSL preprocessing pipeline...")
        run_proc_batch(
            self.osl_config,
            files=self.batch_args["files"],
            subjects=self.batch_args["subjects"],
            outdir=self.save_folder,
            logsdir=logsdir,
            reportdir=reportdir,
            gen_report=True,
            dask_client=True,
        )

        client.close()

    def preprocess_stage_2(self) -> None:
        """
        Stage 2: Load and apply custom processing.
        """
        print("\nStage 2: Applying custom processing pipeline...")
        for subject in self.batch_args["subjects"]:
            subject_dir = os.path.join(self.save_folder, subject)
            fif_file = os.path.join(subject_dir, f"{subject}_preproc-raw.fif")

            if os.path.exists(fif_file):
                try:
                    # Load the .fif file
                    print(f"Processing {subject}...")

                    # Apply our custom processing chain
                    data = self.process_custom(fif_file, subject)

                    # Save full data and chunked examples
                    self.chunk_and_save(data, subject)

                    print(f"INFO: Successfully processed {subject}")
                except Exception as e:
                    print(f"ERROR: Failed to process {subject}: {str(e)}")
            else:
                print(f"WARNING: No .fif file found for {subject}")

        # Clean up intermediate files
        for subject in self.batch_args["subjects"]:
            fif_file = os.path.join(
                self.save_folder, subject, f"{subject}_preproc-raw.fif"
            )
            if os.path.exists(fif_file) and self.delete_fif:
                try:
                    os.remove(fif_file)
                except Exception as e:
                    print(f"Warning: Could not remove {fif_file}: {str(e)}")

    def process_custom(self, fif_file: str, subject: str) -> dict[str, Any]:
        """Process a single raw file with custom pipeline steps.

        Args:
            raw: MNE Raw object after OSL preprocessing

        Returns:
            dict: Processed data and metadata
        """
        data = self.extract_raw(fif_file, subject)
        for func, name in zip(self.extra_funcs, self.preproc_config):
            print(f"INFO: Applying {name}")
            data = func(data, **self.preproc_config[name])
        return data

    def chunk_and_save(self, data: dict[str, Any], session: str) -> None:
        """Chunk session data into fixed length segments and save to disk.

        Args:
            data: Processed data and metadata.
            session: Session name.
        """
        extra_data = {
            "ch_names": data["ch_names"],
            "ch_types": data["ch_types"],
            "pos_2d": data["pos_2d"],
            "session": data["session"],
            "sfreq": data["sfreq"],
        }

        chunk_len = int(data["sfreq"] * self.chunk_seconds)
        array = data["raw_array"]
        n_samples = array.shape[1]
        n_chunks = n_samples // chunk_len
        session_dir = os.path.join(self.save_folder, session)
        os.makedirs(session_dir, exist_ok=True)
        for i in range(n_chunks):
            start = i * chunk_len
            chunk = {
                "data": array[:, start : start + chunk_len],
                **extra_data,
            }
            np.save(os.path.join(session_dir, f"{i}.npy"), chunk)

        # Save the last chunk if there are leftover samples
        remainder = n_samples % chunk_len
        if remainder != 0:
            start = n_chunks * chunk_len
            chunk = {
                "data": array[:, start:],
                **extra_data,
            }
            np.save(os.path.join(session_dir, f"{n_chunks}.npy"), chunk)
