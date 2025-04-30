import copy
import torch
from omegaconf import DictConfig
from appfl.algorithm.aggregator import BaseAggregator
from typing import Union, Dict, OrderedDict, Any, Optional

#TODO
import tenseal as ts
from filelock import FileLock
import pickle
import time

class FedAsyncAggregator(BaseAggregator):
    """
    FedAsync Aggregator class for Federated Learning.
    For more details, check paper: https://arxiv.org/pdf/1903.03934.pdf
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.logger = logger
        self.aggregator_configs = aggregator_configs
        self.client_weights_mode = aggregator_configs.get(
            "client_weights_mode", "equal"
        )

        if model is not None:
            self.named_parameters = set()
            for name, _ in self.model.named_parameters():
                self.named_parameters.add(name)
        else:
            self.named_parameters = None

        self.global_state = None  # Models parameters that are used for aggregation, this is unknown at the beginning

        self.staleness_fn = self.__staleness_fn_factory(
            staleness_fn_name=self.aggregator_configs.get("staleness_fn", "constant"),
            **self.aggregator_configs.get("staleness_fn_kwargs", {}),
        )
        self.alpha = self.aggregator_configs.get("alpha", 0.9)
        self.global_step = 0
        self.client_step = {}
        self.step = {}

        self.encryptedModelWeights = copy.deepcopy(self.model.state_dict())
        self.global_step = 0
        self.ckks_context = None

    def get_parameters(self, **kwargs) -> Dict:
        if self.global_step == 0:
            return self.encryptedModelWeights
        else:
            for name in self.encryptedModelWeights:
                if isinstance(self.encryptedModelWeights[name], bytes):
                    continue
                else:
                    self.encryptedModelWeights[name] = self.encryptedModelWeights[name].serialize()
            return self.encryptedModelWeights

    def aggregate(
        self,
        client_id: Union[str, int],
        local_model: Union[Dict, OrderedDict],
        **kwargs,
    ) -> Dict:

        self.logger.info(f"server is starting aggregation on global step: {self.global_step}")

        if client_id not in self.client_step:
            self.client_step[client_id] = 0

        weight = self.client_sample_size[client_id] / sum(self.client_sample_size.values())

        alpha_t = (
                self.alpha
                * self.staleness_fn(self.global_step - self.client_step[client_id])
                * weight
        )

        if self.global_step == 0:
            self.ckks_context = ts.context_from(kwargs['ckks_context'])
            self.logger.info("server is flattening to list the model first time")
            for name in self.encryptedModelWeights:
                self.encryptedModelWeights[name] = self.encryptedModelWeights[name].flatten().tolist()

        if self.global_step != 0 and self.global_step % 4 == 0:
            model_file = 'latest_updated_global_model.pkl'
            lock_file = model_file + ".lock"
            lock = FileLock(lock_file)
            RETRY_DELAY = 5  # Delay (in seconds) between retries
            while True:
                try:
                    with lock:  # Acquire the lock (blocks indefinitely)
                        with open(model_file, 'rb') as f:
                            updated_global_model = pickle.load(f)

                    for name, encrypted_param_serialized in updated_global_model.items():
                        if isinstance(encrypted_param_serialized, bytes):
                            self.encryptedModelWeights[name] = ts.ckks_vector_from(self.ckks_context,
                                                                                   encrypted_param_serialized)
                    self.logger.info("server done loading from storage file")
                    break  # Exit loop if successful

                except (pickle.UnpicklingError, EOFError) as e:
                    self.logger.error(f"Error loading pickle file: {e}")
                    self.logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)

        for name in self.encryptedModelWeights:
            self.encryptedModelWeights[name] += (local_model[name] - self.encryptedModelWeights[name]) * alpha_t

        self.global_step += 1
        self.client_step[client_id] = self.global_step

        self.logger.info("server is finishing aggregation")
        return {
            name: param.serialize()
            for name, param in self.encryptedModelWeights.items()
        }

    def __staleness_fn_factory(self, staleness_fn_name, **kwargs):
        if staleness_fn_name == "constant":
            return lambda u: 1
        elif staleness_fn_name == "polynomial":
            a = kwargs["a"]
            return lambda u: (u + 1) ** (-a)
        elif staleness_fn_name == "hinge":
            a = kwargs["a"]
            b = kwargs["b"]
            return lambda u: 1 if u <= b else 1.0 / (a * (u - b) + 1.0)
        else:
            raise NotImplementedError
