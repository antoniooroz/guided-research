

from pgnn.data.model_input import ModelInput
from pgnn.base.network_mode import NetworkMode
from pgnn.result.model_output import ModelOutput

import pyro
from pyro.nn import PyroModule
import pyro.distributions as dist


class P_Wrapper(PyroModule):
    def __init__(self, model):
        super().__init__("wrapper")
        self.model = model
        
    def forward(self, model_input: ModelInput, y = None) -> dict[NetworkMode, ModelOutput]:
        model_outputs: dict[NetworkMode, ModelOutput] = self.model(model_input)

        with pyro.plate("data", model_input.indices.shape[0]):
            pyro.sample(
                name="obs", 
                fn=dist.Categorical(logits=model_outputs[NetworkMode.PROPAGATED].logits), 
                obs=y
            )
            
        for network_mode, model_output in model_outputs.items():
            model_output.pyro_deterministic(network_mode)

        return None