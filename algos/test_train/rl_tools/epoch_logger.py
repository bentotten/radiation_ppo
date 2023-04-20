# Taken largely from https://github.com/peproctor/REU_rl/blob/350797ef99bf8405a95f8b8b7e1be3292ce41d11/rl_tools/rl_tools/logx.py
"""
Simple logging functionality, inspired by rllab's logging. Logs to a tab-separated-values file (path/to/output_directory/progress.txt)
"""
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, NamedTuple, Optional, TypedDict, List, Dict, Union
import numpy as np
import numpy.typing as npt
import pickle
import torch
import atexit
import warnings
import time
import torch.nn as nn


class Stats(NamedTuple):
    n: int
    sum: float
    min: float
    max: float
    mean: float
    std: float


class EpochLoggerKwargs(TypedDict):
    output_dir: Path
    exp_name: str


def get_stats(xs: npt.NDArray[np.float32]) -> Stats:
    sum, n = xs.sum(), len(xs)
    mean: float = sum / n
    std: float = np.sqrt(((xs - mean) ** 2 / n).sum())
    min: float = xs.min() if len(xs) > 0 else np.inf
    max: float = xs.max() if len(xs) > 0 else -np.inf
    return Stats(n=n, sum=sum, min=min, max=max, mean=mean, std=std)


def convert_json(obj: Any) -> Any:
    """Convert obj to a version which can be serialized with JSON."""
    try:
        json.dumps(obj)
        return obj
    except:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return tuple(map(convert_json, obj))

        elif isinstance(obj, list):
            return list(map(convert_json, obj))

        elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v) for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)


def setup_logger_kwargs(
    exp_name: str,
    seed: Optional[int] = None,
    data_dir: str = "../exp",
    env_name: Optional[str] = None,
) -> EpochLoggerKwargs:
    """
    Sets up the output_dir for a logger and returns a dict for logger kwargs.

    If no seed is given,

    ::

        output_dir = data_dir/exp_name

    If a seed is given,

    ::

        output_dir = data_dir/exp_name/exp_name_s[seed]

    Args:

        exp_name (string): Name for experiment.

        seed (Optional[int]): Seed for random number generators used by experiment.

        data_dir (string): Path to folder where results should be saved.
            Default is the ``DEFAULT_DATA_DIR`` in ``spinup/user_config.py``.

    Returns:

        logger_kwargs, a dict containing output_dir and exp_name.
    """
    relpath = Path(data_dir) / (env_name if env_name else exp_name)
    if seed is not None:
        relpath = relpath / f"{exp_name}_s{seed}"

    return EpochLoggerKwargs(output_dir=relpath, exp_name=exp_name)


class Logger:
    """
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the
    state of a training run, and the trained model.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        output_fname: str = "progress.txt",
        exp_name: Optional[str] = None,
    ):
        """
        Initialize a Logger.

        Args:
            output_dir (string): A directory for saving results to. If
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.

            output_fname (string): Name for the tab-separated-value file
                containing metrics logged throughout a training run.
                Defaults to ``progress.txt``.

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """

        if output_dir is None:
            self.output_dir = None
            self.output_file = None
        else:
            self.output_dir = output_dir or Path(f"/tmp/experiments/{int(time.time())}")
            if self.output_dir.exists():
                print(
                    f"Warning: Log dir {self.output_dir} already exists! Storing info there anyway."
                )
            else:
                self.output_dir.mkdir(parents=True)
            self.output_file = open(self.output_dir / output_fname, "w+")
            atexit.register(self.output_file.close)
            print(f"Logging data to {self.output_file.name}")

        self.first_row: bool = True
        self.log_headers: List[str] = []
        # TODO: Values may not be primitives
        self.log_current_row: Dict[str, Union[str, int, float]] = {}
        self.exp_name: Optional[str] = exp_name

    def log(self, msg: str) -> None:
        """Print a message to stdout."""
        print(msg)

    def log_tabular(self, key: str, val) -> None:
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert (
                key in self.log_headers
            ), f"Trying to introduce a new key {key} that you didn't include in the first iteration"
        assert (
            key not in self.log_current_row
        ), f"You already set {key} this iteration. Maybe you forgot to call dump_tabular()"
        self.log_current_row[key] = val

    def save_config(self, config: Any) -> None:
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible).

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json: Dict[str, Any] = convert_json(config)
        if self.exp_name is not None:
            config_json["exp_name"] = self.exp_name

        output = json.dumps(
            config_json, separators=(",", ":\t"), indent=4, sort_keys=True
        )
        print("Saving config:\n")
        print(output)
        with open(self.output_dir / "config.json", "w+") as out: # type: ignore
            out.write(output)

    def save_state(self, state_dict: Dict[str, Any], itr: Optional[int] = None) -> None:
        """
        Saves the state of an experiment.

        To be clear: this is about saving *state*, not logging diagnostics.
        All diagnostic logging is separate from this function. This function
        will save whatever is in ``state_dict``---usually just a copy of the
        environment---and the most recent parameters for the model you
        previously set up saving for with ``setup_tf_saver``.

        Call with any frequency you prefer. If you only want to maintain a
        single state and overwrite it at each call with the most recent
        version, leave ``itr=None``. If you want to keep all of the states you
        save, provide unique (increasing) values for 'itr'.

        Args:
            state_dict (Dict): Dictionary containing essential elements to
                describe the current state of training.

            itr: An int, or None. Current iteration of training.
        """

        fname = "vars.pkl" if itr is None else f"vars{itr}.pkl"
        try:
            with open(self.output_dir / fname, "wb+") as f: # type: ignore
                pickle.dump(state_dict, f, pickle.HIGHEST_PROTOCOL)
        except:
            self.log("Warning: could not pickle state_dict.")
        if hasattr(self, "pytorch_saver_elements"):
            self._pytorch_simple_save(itr)

    def setup_pytorch_saver(self, what_to_save: nn.Module) -> None:
        """
        Set up easy model saving for a single PyTorch model.

        Because PyTorch saving and loading is especially painless, this is
        very minimal; we just need references to whatever we would like to
        pickle. This is integrated into the logger because the logger
        knows where the user would like to save information about this
        training run.

        Args:
            what_to_save: Any PyTorch model or serializable object containing
                PyTorch models.
        """
        self.pytorch_saver_elements = what_to_save.state_dict()

    def _pytorch_simple_save(self, itr: Optional[int] = None) -> None:
        """
        Saves the PyTorch model (or models).
        """
        assert hasattr(
            self, "pytorch_saver_elements"
        ), "First have to setup saving with self.setup_pytorch_saver"
        fpath: Path = self.output_dir / "pyt_save" # type: ignore
        fname = f"model{itr if itr is not None else ''}.pt"
        fname = fpath / fname # type: ignore 
        fpath.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # We are using a non-recommended way of saving PyTorch models,
            # by pickling whole objects (which are dependent on the exact
            # directory structure at the time of saving) as opposed to
            # just saving network weights. This works sufficiently well
            # for the purposes of Spinning Up, but you may want to do
            # something different for your personal PyTorch project.
            # We use a catch_warnings() context to avoid the warnings about
            # not being able to save the source code.
            torch.save(self.pytorch_saver_elements, fname)

    def dump_tabular(self) -> None:
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        keystr = "%" + "%d" % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))
            vals.append(val)
        print("-" * n_slashes, flush=True)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers) + "\n")
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False


class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use

    .. code-block:: python

        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you
    would use

    .. code-block:: python

        epoch_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.epoch_dict: Dict[str, List[Any]] = defaultdict(list)

    def store(self, **kwargs: Any) -> None:
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical
        values.
        """
        for k, v in kwargs.items():
            self.epoch_dict[k].append(v)

    def log_tabular(
        self,
        key: str,
        val: Any = None,
        with_min_and_max: bool = False,
        average_only: bool = False,
        sum_only: bool = False,
        rate_only: bool = False,
    ) -> None:
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key, val)
        else:
            v = self.epoch_dict[key]
            vals: npt.NDArray[np.float32] = (
                np.concatenate(v)
                if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0
                else np.array(v, dtype=np.float32)
            )
            stats = get_stats(vals)

            if not rate_only:
                super().log_tabular(
                    key if (average_only or sum_only) else "Mean" + key, stats.mean
                )
            if not (average_only or sum_only):
                super().log_tabular("Std" + key, stats.std)
            if with_min_and_max:
                super().log_tabular("Max" + key, stats.max)
                super().log_tabular("Min" + key, stats.min)
            if rate_only:
                rate = sum(vals) / sum(self.epoch_dict["EpLen"])
                super().log_tabular(key, rate)
        self.epoch_dict[key] = []
