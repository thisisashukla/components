import os
import time
import shutil
import typing
import binascii
import warnings
import itertools
import collections
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from itertools import product

from common import *
from models.segregator import Segregator


class MLHandler:
    def __init__(
        self,
        data: dict,
        model_classes: list,
        data_segregation_keys: list,
        features: list[str],
        target: str,
        grain_columns: list[str],
        base_path: str,
        model_path: str = None,
        preprocessor: callable = None,
        postprocessor: callable = None,
        preprocess_on: list = None,
        postprocess_on: list = None,
        seg_fit_on: str = "train",
        log_function: callable = None,
        init_msg: str = None,
        model_kwargs: dict = None,
        verbose=1,
    ) -> None:

        assert len(model_classes) == len(data_segregation_keys)

        self.data = data
        self.model_objects = model_classes
        self.data_segregation_keys = data_segregation_keys
        self.features = features
        self.target = target
        self.grain_columns = grain_columns
        self.seg_fit_on = seg_fit_on
        self.base_path = base_path
        self.model_path = model_path
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.preprocess_on = preprocess_on
        self.postprocess_on = postprocess_on
        self.ml_log = log_function
        self.init_msg = init_msg
        self.verbose = verbose

        # configuring required files based on mode and env
        if preprocess_on is None:
            self.preprocess_on = ["train", "validation", "test"]
        if postprocess_on is None:
            self.postprocess_on = ["train", "validation", "test"]

        if self.verbose == 0:
            self.ml_log = self.dummy_log

        self.ml_log(
            f"INIT Settings: {self.__dict__}",
        )

        self.files = {}
        self.models = []

        if not os.path.exists(f"{self.base_path}/models"):
            os.mkdir(f"{self.base_path}/models")

        if model_path is None:
            self.model_path = base_path

        if init_msg is not None:
            self.ml_log(f"INIT NOTE: {init_msg}")

    def dummy_log(self, msg):
        pass

    def get_model_attribute(self, attr: str) -> dict:

        attributes = {}
        for m in self.models:
            if attr in dict(m).keys():
                attributes[m.name] = getattr(m, attr)

        return attributes

    def read_data_sample(self, name: str, sample_size: int or float, nsamples: int = 1):

        df = self.read_data(name)

        sample_rows = 0
        if isinstance(sample_size, int):
            sample_rows = sample_size
        else:
            sample_rows = int(df.shape[0] * sample_size)

        samples = []
        for i in range(nsamples):
            samples.append(df.sample(sample_rows))

        return samples

    def read_data(self, name: str) -> pd.DataFrame:

        if "_sample_" in name:
            file_name, sample, size, identifier = name.split("_")
            size = eval(size)
            self.ml_log(f"Sampling {size} from {name} data")
            samples = self.read_data_sample(file_name, size)
            self.files[name] = samples[0]

            return samples[0]

        # calling the read method of the IO Handler for the given checkpoint
        self.ml_log("Reading Data using base path")
        file = self.read_data(name)
        file = file.drop_duplicates(subset=self.grain_columns)

        # passing the data through the preprocessing function
        if self.preprocessor and name in self.preprocess_on:
            self.ml_log(f"Passing {name} file through the preprocessor method")
            file = self.preprocessor(file)
            assert (
                file.shape[0] > 0
            ), f"Preprocessor function did not return a valid dataframe"

        # adding the file to the dictionary
        self.files[name] = file

        return file

    def initialize_models(
        self,
        criterion: typing.Union[typing.Callable, pd.DataFrame] = None,
        read_from_disk: bool = False,
        names: typing.List = None,
    ) -> None:

        if read_from_disk:
            assert (
                names is not None
            ), f"To read models from disk, passing the path to pickle file is required"
            # checking if number of names passed is equal to number of models
            assert len(names) == len(
                self.model_objects
            ), f"Number of names passed ({len(names)}) do not match number of models ({len(self.model_list)} in the config"

            # removing .pkl extension from names if passed by user
            names = [n if ".pkl" not in n else n.split(".")[0] for n in names]

        # instantiating segregator object is not already done
        if "segregator" not in self.__dict__:
            self.segregator = Segregator(
                {k: self.read_data(k) for k in self.files_required},
                criterion,
                self.grain_columns,
                self.seg_fit_on,
                logger=self.ml_log,
                verbose=self.verbose,
            )

        self.models = []
        for i, (model, seg) in enumerate(
            zip(self.model_objects, self.data_segregation_keys)
        ):

            # obtaining correct split of data for current model
            part = self.segregator.get_partial(s)

            for k, v in part.items():
                assert v.shape[0] > 0, f"{k} data is empty"

            # instantiating model class
            # model_class = getattr(
            #     importlib.import_module(f"pipeline.models.{m}"), "Model"
            # )
            model_class = m

            # model name is same as that passed by user or is generated in the format: <class_name>_<timestamp>_<hex_code>
            name = ""
            if read_from_disk:
                name = names[i]
            else:
                name = f"{model}_{utils.localize_ts(datetime.now())}_{binascii.b2a_hex(os.urandom(15)).decode('utf-8')[:5]}"
            model.name = name

            # saving model data to S3 for future debugging
            # self.ml_log(f"Saving data for model {name} to S3")
            # for k, v in part.items():
            #     # getattr(IO_HANDLER, f"write_data")(v, validate=True, suffix=name)
            #     self.write_data(v, k)

            # instantiating model object
            model.initialize()
            if read_from_disk:
                self.ml_log(f"Reading {model.name}")
                model.read_model(f"{self.model_path}/{names[i]}.pkl")

            self.models.append(model)

    def train(self, save_on_s3=True):

        # if self.models is empty then there are not models to be trained
        assert len(self.models) > 0, f"No models initialized or read from disk"

        for model in self.models:
            self.ml_log(
                f"Starting training for model {model.name} at {utils.localize_ts(datetime.now())}"
            )
            try:
                tic = time.time()
                model.train()
                print(
                    f"Time taken to train {model.name} = {round((time.time()-tic)/60, 2)} minutes"
                )
                model.save_model(f"{self.base_path}/models")
                self.ml_log(
                    f"Training complete and model saved to local for model {model.name} at {utils.localize_ts(datetime.now())}"
                )
            except Exception as e:
                self.ml_log(
                    f"Training for model {model} failed. Exception: {e}. Skipping."
                )

    def predict(
        self,
        model_names: typing.Optional[typing.List[tuple]] = None,
        file_name: str = None,
        save_on_s3: bool = True,
        criterion: typing.Union[typing.Callable, pd.DataFrame] = None,
    ) -> pd.DataFrame:

        # if model_names are passed then initialize models with read_from_disk = True else check if all models are trained or not
        if model_names is None:
            if not all(self.model_attribute("name")):
                warnings.warn(
                    f"models with the names {[model.name for model in self.models if model.trained != True]} are not trained",
                    category=Warning,
                )
        else:
            if "models" not in self.__dict__:
                self.initialize_models(
                    criterion, read_from_disk=True, names=model_names
                )

        # generate part predictions for different cases
        part_predictions = []
        if file_name is None:
            # if file_name is none then run default predictions for each model
            part_predictions = [m.predict() for m in self.models]
        elif file_name in self.files_required:
            # if file_name is passed and it is present in required files then predict using that file name
            part_predictions = [m.predict(key=file_name) for m in self.models]
        else:
            # if file_name if passed but not present in the required files then read data from S3
            # apply segregation on the data and then predict using those parts
            data = self.read_data(file_name)
            data = data.drop_duplicates(subset=self.grain_columns)
            segregated = self.segregator.apply_criterion(data)
            parts = [{file_name: segregated[s]} for s in self.data_segregation_keys]

            assert len(parts) == len(
                self.models
            ), f"{len(parts)} and {len(self.models)} do not match"

            part_predictions = [m.predict(data=p) for m, p in zip(self.models, parts)]

        assert len(part_predictions) == len(
            self.models
        ), f"Number of predictions ({len(part_predictions)}) do not match the number of effective models ({len(self.models)})"

        # checking whether all predictions have model name for the correspoding model
        assert all(
            [
                m.name == pred["model_name"].unique()[0]
                for m, part in zip(self.models, part_predictions)
                for k, pred in part.items()
            ]
        ), f"Prediction order mismatch"

        # stiching part predictions into one for every key
        final_predictions = self.stitch_predictions(part_predictions)

        if not (self.postprocessor is None):
            self.ml_log(f"Postprocessing predictions")
            for k, df in final_predictions.items():
                if k in self.postprocess_on:
                    final_predictions[k]["prediction"] = self.postprocessor(
                        df["prediction"]
                    )

        return final_predictions

    def stitch_predictions(self, part_predictions: list):

        # getting keys from each element in the part_prediction list and creating a lower diagonal triangle of pairs of these keys
        part_keys = [p.keys() for p in part_predictions]
        ind_pairs = [
            (i, j) for i, j in product(range(len(part_keys)), repeat=2) if i < j
        ]

        # checking if any pair has dissimlar keys
        key_bool = True
        for i, j in ind_pairs:
            if part_keys[i] != part_keys[j]:
                key_bool = False
                self.ml_log(f"Predictions have dissimilar keys. Not combining them.")
                break

        # checking if any dataframe in the list has dissimilar shape
        shape_bool = True
        if key_bool:
            keys = part_predictions[0].keys()
            for key in keys:
                shapes = set([p[key].shape[1] for p in part_predictions])

                def compare(x, y):
                    return collections.Counter(x) == collections.Counter(y)

                cols = all(
                    [
                        compare(
                            part_predictions[i][key].columns,
                            part_predictions[j][key].columns,
                        )
                        for i, j in product(range(len(part_predictions)), repeat=2)
                        if i < j
                    ]
                )
                if len(list(shapes)) > 1 and cols:
                    shape_bool = False
                    self.ml_log(
                        f"Shape or columns of predictions is dissimilar. Cannot combine data frames"
                    )
                    break

        # combining parts into final predictions by iterating on keys and then concating them
        if key_bool and shape_bool:
            keys = part_predictions[0].keys()
            final_predictions = {}
            for k in keys:
                for p in part_predictions:
                    p[k].columns = [str(i) for i in p[k].columns]
                final_predictions[k] = pd.concat(
                    [p[k] for p in part_predictions], axis=0
                ).reset_index(drop=True)

                # validating predictions before converting them to final format
                valid = self.validate_predictions(final_predictions[k], k)
                if valid:
                    formatted_prediction = self.prepare_predictions(
                        final_predictions[k]
                    )
                    final_predictions[k] = formatted_prediction

            return final_predictions

        self.ml_log(f"Stitching predictions failed. Returning list of parts")

        return part_predictions

    def prepare_predictions(self, prediction: pd.DataFrame) -> pd.DataFrame:

        model_columns = ["model_name"]
        if "model_type" in prediction.columns:
            model_columns.append("model_type")

        # metling predictions
        df = pd.melt(
            prediction,
            id_vars=self.grain_columns + model_columns,
            value_vars="prediction",
            var_name="quantile",
            value_name="prediction",
        )

        # filtering columns and sorting
        return df

    def save_models(self):

        self.ml_log(f"Saving models to S3")

        # raising warning if all models are not trained
        if not all(self.model_attribute("trained")):
            warnings.warn(f"Not all models are in trained state", category=Warning)

        for model in self.models:
            # saving model to temporary directory
            result = model.save_model(f"{self.model_path}/models")

        return self.model_attribute("name")
