##############################################################
# This file is a modified version from the following source
# Author: Aleksa Gordić
# Last Visited: 14.06.2022
# Title: GAT - Graph Attention Network (PyTorch)
# URL: https://github.com/gordicaleksa/pytorch-GAT
##############################################################

import torch
import torch.nn as nn
from pgnn.base.network_mode import NetworkMode
from pgnn.configuration.configuration import Configuration
from pgnn.configuration.model_configuration import ModelType
from pgnn.data.model_input import ModelInput

from pgnn.logger import LogWeight, LogWeightValue
from pgnn.base import Base, Linear
from pgnn.result.model_output import ModelOutput
from pgnn.utils import get_device, get_edge_indices, get_self_edge_indices
from pyro.nn import PyroModule, PyroParam, PyroSample
import pyro.distributions as dist


class GAT(Base):
    def __init__(self, nfeatures: int, nclasses: int, configuration: Configuration, adj_matrix: torch.Tensor):
        super().__init__()
        self.configuration = configuration
        self.nclasses = nclasses
        self.nfeatures = nfeatures

        self.self_edge_indices = get_self_edge_indices(adj_matrix)
        self.edge_indices = torch.cat((get_edge_indices(adj_matrix)[0], self.self_edge_indices), dim=1).to(self.self_edge_indices.device)
        
        hiddenunits = self.configuration.model.hidden_layer_size
        heads_per_layer = self.configuration.model.gat_hidden_layers_heads

        GATLayer = GATLayerImp3  # fetch one of 3 available implementations
        heads_per_layer = [1] + heads_per_layer + [1]  # trick - so that I can nicely create GAT layers below
        featues_per_layer = [nfeatures] + hiddenunits + [nclasses]
        
        assert len(featues_per_layer) == len(heads_per_layer), f'Enter valid arch params.'

        layers = []  # collect GAT layers
        for i in range(len(featues_per_layer) - 1):
            layer = GATLayer(
                num_in_features=featues_per_layer[i] * heads_per_layer[i],  # consequence of concatenation
                num_out_features=featues_per_layer[i+1],
                num_of_heads=heads_per_layer[i+1],
                configuration=self.configuration,
                concat=True if i < len(featues_per_layer) - 2 else False,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < len(featues_per_layer) - 2 else None,  # last layer just outputs raw scores
                dropout_prob=self.configuration.training.drop_prob,
                add_skip_connection=self.configuration.model.gat_skip_connections,
                bias=self.configuration.model.bias,
                log_attention_weights=False,
                name="gat_layer_"+str(i)
            )
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    # data is just a (in_nodes_features, topology) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, model_input: ModelInput) -> dict[NetworkMode, ModelOutput]:
        isolated_input = (model_input.features, self.self_edge_indices)
        propagated_input = (model_input.features, self.edge_indices)
        
        isolated_logits = self.layers(isolated_input)[0].squeeze(0).index_select(
            dim=-2, index=model_input.indices)
        propagated_logits = self.layers(propagated_input)[0].squeeze(0).index_select(
            dim=-2, index=model_input.indices)

        return {
            NetworkMode.ISOLATED: ModelOutput(
                logits=isolated_logits,
                softmax_scores=isolated_logits.softmax(-1)
            ),
            NetworkMode.PROPAGATED: ModelOutput(
                logits=propagated_logits,
                softmax_scores=propagated_logits.softmax(-1)
            )
        }

class GATLayer(PyroModule):
    """
    Base class for all implementations as there is much code that would otherwise be copy/pasted.
    """

    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, configuration: Configuration, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False, name=""):

        super().__init__(name)
        self.configuration = configuration

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection

        # Scoring
        self.scoring_fn_target = None
        self.scoring_fn_target_mean = None
        self.scoring_fn_target_var = None

        self.scoring_fn_source = None
        self.scoring_fn_source_mean = None
        self.scoring_fn_source_var = None
        
        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = Linear(
            input_dim=num_in_features, 
            output_dim=num_of_heads * num_out_features,
            configuration=self.configuration,
            bias=False,
            activation=lambda x: x,
            dropout=0,
            name=self._pyro_name[4:]
            )

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = PyroParam(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = PyroParam(torch.Tensor(1, num_of_heads, num_out_features))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params()

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow
        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.
        """
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
        # only imp1 will enter this one
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, 1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(out_nodes_features.shape[0], out_nodes_features.shape[1], self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)

    def pyronize_weights(self):
        bayesian_linear_proj = self.configuration.model.type in ModelType.gat_bayesian_projection()
        bayesian_attention = self.configuration.model.type in ModelType.gat_bayesian_projection()

        # Linear Projection
        if bayesian_linear_proj:
            self.linear_proj.pyronize_weights()

        # Attention
        if bayesian_attention:
            # Mean
            self.scoring_fn_target_mean = torch.Tensor(1, self.num_of_heads, self.num_out_features).to(get_device()) + self.configuration.model.initial_mean
            self.scoring_fn_source_mean = torch.Tensor(1, self.num_of_heads, self.num_out_features).to(get_device()) + self.configuration.model.initial_mean
            if self.configuration.model.train_mean:
                self.scoring_fn_target_mean = PyroParam(self.scoring_fn_target_mean)
                self.scoring_fn_source_mean = PyroParam(self.scoring_fn_source_mean)

            nn.init.xavier_uniform_(self.scoring_fn_target_mean)
            nn.init.xavier_uniform_(self.scoring_fn_source_mean)
        
            # Variance
            self.scoring_fn_target_var = torch.ones([1, self.num_of_heads, self.num_out_features]).to(get_device()) * self.configuration.model.initial_var
            self.scoring_fn_source_var = torch.ones([1, self.num_of_heads, self.num_out_features]).to(get_device()) * self.configuration.model.initial_var
            if self.configuration.model.train_var:
                self.scoring_fn_target_var = PyroParam(self.scoring_fn_target_var, constraint=dist.constraints.positive)
                self.scoring_fn_source_var = PyroParam(self.scoring_fn_source_var, constraint=dist.constraints.positive)
            
            # Weight
            if self.configuration.model.weight_prior == "normal":
                self.scoring_fn_target = PyroSample(lambda self: dist.Normal(self.scoring_fn_target_mean, self.scoring_fn_target_var).to_event(2))
                self.scoring_fn_source = PyroSample(lambda self: dist.Normal(self.scoring_fn_source_mean, self.scoring_fn_source_var).to_event(2))
            elif self.configuration.model.weight_prior == "laplace":
                self.scoring_fn_target = PyroSample(lambda self: dist.Laplace(self.scoring_fn_target_mean, self.scoring_fn_target_var).to_event(2))
                self.scoring_fn_source = PyroSample(lambda self: dist.Laplace(self.scoring_fn_source_mean, self.scoring_fn_source_var).to_event(2))
            elif self.configuration.model.weight_prior == "none":
                self.scoring_fn_target = self.scoring_fn_target_mean
                self.scoring_fn_source = self.scoring_fn_source_mean
            else:
                raise NotImplementedError()
    
    def log_weights(self):
        log = self.linear_proj.log_weights()

        if self.scoring_fn_target_mean is not None:
            log.extend([
                LogWeight(
                    name=self._pyro_name + '_scoring_target',
                    type='att',
                    mu=LogWeightValue(self.scoring_fn_target_mean),
                    sigma=LogWeightValue(self.scoring_fn_target_var)
                ),
                LogWeight(
                    name=self._pyro_name + '_scoring_source',
                    type='att',
                    mu=LogWeightValue(self.scoring_fn_source_mean),
                    sigma=LogWeightValue(self.scoring_fn_source_var)
                ),
            ])
        else:
            log.extend([
                LogWeight(
                    name=self._pyro_name + '_scoring_target',
                    type='att',
                    mu=LogWeightValue(self.scoring_fn_target)
                ),
                LogWeight(
                    name=self._pyro_name + '_scoring_source',
                    type='att',
                    mu=LogWeightValue(self.scoring_fn_source)
                ),
            ])

        return log


class GATLayerImp3(GATLayer):
    """
    Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric
    But, it's hopefully much more readable! (and of similar performance)
    It's suitable for both transductive and inductive settings. In the inductive setting we just merge the graphs
    into a single graph with multiple components and this layer is agnostic to that fact! <3
    """

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 1      # node dimension/axis
    head_dim = 2      # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, configuration: Configuration, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False, name=""):

        # Delegate initialization to the base class
        super().__init__(num_in_features, num_out_features, num_of_heads, configuration, concat, activation, dropout_prob,
                      add_skip_connection, bias, log_attention_weights, name)

    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization
        #

        in_nodes_features, edge_index = data  # unpack data
        num_of_nodes = in_nodes_features.shape[-2]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        
        # Fix if shapes mismatch due to different behavior of pyro when training compared to prediction
        if (len(self.linear_proj.weight.shape) > len(in_nodes_features.shape)):
            in_nodes_features = in_nodes_features.unsqueeze(len(self.linear_proj.weight.shape)-3)

        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, num_of_nodes, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        self.edge_scores = scores_per_edge

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index)

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.
        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:
        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
        """
        if scores_per_edge.shape[0] > 0:
            # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
            scores_per_edge = scores_per_edge - scores_per_edge.amax(dim=[1,2], keepdim=True)
        
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index.unsqueeze(0), exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim].unsqueeze(0), nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).
        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)
