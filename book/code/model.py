"""Defines PyTorch models, aggregations, layers, etc for graph classification."""
from functools import partial

import torch
from torch import nn
from utils import LOG as logger

##############
# CONV BLOCK #
##############


class GraphConv(nn.Module):
    """Basic graph convolutional layer implementing the simple neighborhood aggregation."""

    def __init__(self, in_features, out_features, activation=None):
        """Initialize the graph convolutional layer.

        Args:
            in_features (int): number of input node features.
            out_features (int): number of output node features.
            activation (nn.Module or callable): activation function to apply. (optional)
        """
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Linear(in_features, out_features, bias=False)
        self.activation = activation if activation is not None else lambda x: x
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight.weight)
        nn.init.kaiming_uniform_(self.bias.weight)

        # nn.init.xavier_uniform_(self.weight.weight)
        # nn.init.xavier_uniform_(self.bias.weight)

    def forward(self, x, adj):
        """Perform graph convolution operation.

        Args:
            x (Tensor): Input node features of shape (num_nodes, in_features).
            adj (Tensor): Adjacency matrix of the graph, shape (num_nodes, num_nodes).

        Returns:
            Tensor: Output node features after graph convolution, shape (num_nodes, out_features).
        """
        logger.debug(f"adj: {adj.shape}")
        logger.debug(f"x: {x.shape}")

        degree = torch.sum(adj, dim=2, keepdim=True).clamp(min=1)

        logger.debug(f"degree: {degree.shape}")
        # compute A_norm where degree is not 0, but keep the shape of degree
        # diagonal = torch.diag_embed(degree)
        adj = adj / degree
        support = torch.bmm(adj, x)
        logger.debug(f"support: {support.shape}")
        out = self.weight(support) + self.bias(x)
        return self.activation(out)


class GraphSAGEConv(nn.Module):
    """GraphSAGE convolutional layer."""

    def __init__(
        self,
        in_features,
        out_features,
        aggregation,
        activation=None,
        use_bias=False,
    ):
        """Initialize the GraphSAGE convolutional layer.

        Args:
            in_features (int): number of input node features.
            out_features (int): number of output node features.
            aggregation (nn.Module or callable): aggregation function to apply, as x_agg = aggegration(x, adj).
            activation (nn.Module or callable): activation function to apply. (optional)
            use_bias (bool): whether to use a bias term. (default: False)
        """
        super().__init__()

        self.weight = nn.Linear(2 * in_features, out_features, bias=False)
        self.bias = (
            nn.Linear(2 * in_features, out_features, bias=False)
            if use_bias
            else None
        )
        self.activation = activation if activation is not None else lambda x: x
        self.aggregation = aggregation

        # self._reset_parameters()

    def _reset_parameters(self):
        # self.weight.data.normal_(std=1.0 / math.sqrt(self.weight.size(1)))
        # if self.bias is not None:
        # self.bias.data.normal_(std=1.0 / math.sqrt(self.bias.size(1)))

        nn.init.kaiming_uniform_(self.weight.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias.weight)

    def forward(self, x, adj):
        """Perform graph convolution operation.

        Args:
            x (Tensor): Input node features of shape (num_nodes, in_features).
            adj (Tensor): Adjacency matrix of the graph, typically sparse, shape (num_nodes, num_nodes).

        Returns:
            Tensor: Output node features after graph convolution, shape (num_nodes, out_features).
        """
        agg = self.aggregation(x, adj)
        logger.debug(f"agg: {agg.shape}")
        cat = torch.cat([x, agg], dim=2)
        logger.debug(f"cat: {cat.shape}")
        logger.debug(f"weight: {self.weight.weight.shape}")
        out = self.weight(cat)
        if self.bias is not None:
            out += self.bias(cat)

        return self.activation(out)


class AttentionConv(nn.Module):
    """Attention-based graph convolutional layer as described in Graph Attention Networks (https://arxiv.org/pdf/1710.10903.pdf)."""

    def __init__(
        self,
        in_features,
        out_features,
        num_nodes=28,
        activation=None,
        attention_activation=None,
        alpha=0.2,
    ):
        """Initialize the attention-based graph convolutional layer.

        Args:
            in_features (int): number of input node features.
            out_features (int): number of output node features.
            num_nodes (int): max number of nodes in the graph. (default: 28)
            activation (nn.Module or callable): activation function to apply. (optional)
            attention_activation (nn.Module or callable): activation function to apply to the attention scores. (default : LeakyReLU(0.2))
            alpha (float): alpha value for the LeakyReLU activation of attention score. (default: 0.2)
        """
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.S = nn.Linear(2 * out_features, num_nodes, bias=False)
        self.activation = activation if activation is not None else lambda x: x
        self.att_activation = (
            attention_activation
            if attention_activation is not None
            else nn.LeakyReLU(alpha)
        )
        self.softmax = nn.Softmax(dim=2)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight.weight)
        nn.init.kaiming_uniform_(self.S.weight)

    def forward(self, x, adj):
        """Perform graph convolution operation."""
        logger.debug(f"support: {self.weight(x).shape}")
        cat_features = torch.cat([self.weight(x), self.weight(x)], dim=2)
        att_scores = self.att_activation(self.S(cat_features))
        logger.debug(f"att_score: {att_scores.shape}")
        norm_att = self.softmax(att_scores)
        logger.debug(f"norm_att: {norm_att.shape}")
        norm_att = norm_att * adj
        out = torch.bmm(norm_att, self.weight(x))
        logger.debug(f"out: {out.shape}")
        return self.activation(out.squeeze())


class TorchAttentionConv(nn.Module):
    """Using torch MultiheadAttention for comparison with the above approach."""

    def __init__(
        self,
        in_features,
        out_features,
        activation=None,
        attention_activation=None,
        alpha=0.2,
    ):
        """Initialize the attention-based graph convolutional layer.

        Args:
            in_features (int): number of input node features.
            out_features (int): number of output node features.
            activation (nn.Module or callable): activation function to apply. (optional)
            attention_activation (nn.Module or callable): activation function to apply to the attention scores. (default : LeakyReLU(0.2))
            alpha (float): alpha value for the LeakyReLU activation of attention score. (default: 0.2)
        """
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)
        # self.S = nn.Linear(2 * out_features, 28, bias=False)
        self.attention = nn.MultiheadAttention(
            embed_dim=out_features,
            num_heads=8,
            batch_first=True,
            dropout=0.2,
        )
        self.activation = activation if activation is not None else lambda x: x
        self.att_activation = (
            attention_activation
            if attention_activation is not None
            else nn.LeakyReLU(alpha)
        )
        self.softmax = nn.Softmax(dim=2)

        # self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal(self.weight.weight)
        nn.init.xavier_normal_(self.S.weight)

    def forward(self, x, adj):
        """Perform graph convolution operation."""
        logger.debug(f"support: {self.weight(x).shape}")
        # cat_features = torch.cat([self.weight(x), self.weight(x)], dim=2)
        # att_scores = self.att_activation(self.S(cat_features))
        # logger.debug(f"att_score: {att_scores.shape}")
        # norm_att = self.softmax(att_scores)
        norm_att = self.attention(
            self.weight(x), self.weight(x), self.weight(x)
        )[0]
        logger.debug(f"norm_att: {norm_att.shape}")
        # out = torch.bmm(norm_att, self.weight(x))
        # logger.debug(f"out: {out.shape}")
        # return self.activation(out.squeeze())
        return self.activation(norm_att)


class EdgeConv(nn.Module):
    """Attention-based graph convolutional layer with edge features replacing attention scores.

    Reference : Exploiting Edge Features for Graph Neural Networks, Gong and Cheng, 2019

    Args:
        in_features (int): number of input node features.
        out_features (int): number of output node features.
        activation (nn.Module or callable): activation function to apply. (optional)
        attention_activation (nn.Module or callable): activation function to apply to the attention scores. (default : LeakyReLU(0.2))
    """

    def __init__(
        self,
        in_features,
        out_features,
        num_nodes=28,
        activation=None,
        attention_activation=None,
        alpha=0.1,
    ):
        """Initialize the attention-based graph convolutional layer.

        Args:
            in_features (int): number of input node features.
            out_features (int): number of output node features.
            num_nodes (int): max number of nodes in the graph. (default: 28)
            activation (nn.Module or callable): activation function to apply. (optional)
            attention_activation (nn.Module or callable): activation function to apply to the attention scores. (default : LeakyReLU(0.2))
            alpha (float): alpha value for the LeakyReLU activation of attention score. (default: 0.2)
        """
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.S = nn.Linear(2 * out_features, num_nodes, bias=False)
        self.edge_layer = nn.Linear(
            4 * num_nodes, num_nodes, bias=False
        )  # layer to go from concatenated features for each edge to out_features, like multi-head attention
        self.activation = activation if activation is not None else lambda x: x
        self.att_activation = (
            attention_activation
            if attention_activation is not None
            else nn.LeakyReLU(alpha)
        )
        self.softmax = nn.Softmax(dim=1)
        # self.ds_norm = DoublyStochasticNormalization()
        # self.instance_norm = nn.InstanceNorm2d(4)
        # self.instance_norm = nn.LayerNorm(4)
        self.instance_norm = nn.GroupNorm(4, 4)

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight.weight)
        nn.init.kaiming_uniform_(self.a.weight)

    def forward(self, x, adj, E):
        """Perform graph convolution operation with edge features.

        Uses edge features as "channels" for the attention scores.

        Args:
            x (Tensor): Input node features of shape (batch, channels, num_nodes, in_features).
            adj (Tensor): Adjacency matrix of the graph, shape (batch, num_nodes, num_nodes).
            E (Tensor): Edge features of shape (batch, num_nodes, num_nodes, num_features).

        Returns:
            Tensor: Output node features after graph convolution, shape (batch, channels, num_nodes, out_features).
        """
        logger.debug(f"weight: {self.weight.weight.shape}")
        support = self.weight(x)
        logger.debug(f"support: {support.shape}")

        cat_features = torch.cat([self.weight(x), self.weight(x)], dim=2)
        att_score = self.att_activation(self.S(cat_features))
        att_score = self.softmax(att_score)
        ############### TEST : set edge features to 1
        # E = torch.ones_like(E)
        ###############
        att_score = att_score.unsqueeze(-1).repeat(
            1, 1, 1, E.shape[-1]
        )  # duplicate att_score to be of shape BxNxNxE
        logger.debug(f"att_score: {att_score.shape}")
        logger.debug(f"E: {E.shape}")
        alpha_channels = att_score * E
        alpha_channels = alpha_channels.swapaxes(1, 3)
        logger.debug(f"alpha_channels: {alpha_channels.shape}")
        ############### TEST : remove the normalization
        alpha_channels = self.instance_norm(  # normalize across the nodesxnodes dimension (BxNxNxE)
            alpha_channels
        )
        ############### TEST : set alpha_channels to 1
        # alpha_channels = torch.ones_like(alpha_channels)

        # stack the channels back together
        stacked_edge_features = torch.cat(  # concatenate the edge features along the channels dimension
            [
                alpha_channels[:, i, :, :]
                for i in range(alpha_channels.shape[1])
            ],
            dim=2,
        )
        logger.debug(f"stacked_edge_features: {stacked_edge_features.shape}")
        # pass them to Linear layer to unify the channels back to out_features ? not in paper but cannot match shapes after cat otherwise
        out = self.edge_layer(stacked_edge_features)
        # logger.debug(f"out: {out.shape}")
        out = torch.bmm(out, support)
        # logger.debug(f"out: {out.shape}")
        return self.activation(out), alpha_channels.swapaxes(1, 3)


##############
# AGG FUNCS #
##############

# Note: all these aggregations take as input the node embedding x, adjacency matrix adj,
# and return the aggregated node embedding x_agg.


class MeanAggregation(nn.Module):
    """Aggregate node features by averaging over the neighborhood."""

    def __init__(self):
        """Initialize the mean aggregation layer."""
        super().__init__()

    def forward(self, x, adj):
        """Perform mean aggregation."""
        degree = torch.sum(adj, dim=2, keepdim=True).clamp(min=1)
        adj = adj / degree
        return torch.bmm(adj, x)


class SumAggregation(nn.Module):
    """Aggregate node features by summing over the neighborhood."""

    def __init__(self):
        """Initialize the sum aggregation layer."""
        super().__init__()

    def forward(self, x, adj):
        """Perform sum aggregation."""
        return torch.bmm(adj, x)


class SqrtDegAggregation(nn.Module):
    """Aggregate node features by summing over the neighborhood and normalizing by the degrees."""

    def __init__(self):
        """Initialize the sqrt degree normalization layer."""
        super().__init__()

    def forward(self, x, adj):
        """Perform mean with square root of degree normalization."""
        degree = torch.sum(adj, dim=2, keepdim=True).clamp(min=1).sqrt()
        adj = adj / degree
        return torch.bmm(adj, x)


class LSTMAggregation(nn.Module):
    """Aggregate node features by using an LSTM on the neighborhood."""

    def __init__(self, in_features=None, out_features=None):
        """Initialize the LSTM aggregation layer."""
        super().__init__()
        self.in_f = in_features
        self.out_f = out_features
        if in_features is None or out_features is None:
            self.lstm = None
        else:
            self.lstm = nn.LSTM(in_features, out_features, batch_first=True)

    @classmethod
    def make_lstm_agg(cls, in_features, out_features):
        """Create an LSTM aggregation layer with the given input and output features."""
        return cls(in_features, out_features)

    def __call__(self, x, adj):
        """Perform LSTM aggregation."""
        logger.debug(f"LSTM adj: {adj.shape}")
        logger.debug(f"LSTM x: {x.shape}")
        logger.debug(f"LSTM in_f: {self.in_f}")
        logger.debug(f"LSTM out_f: {self.out_f}")
        out, _ = self.lstm(torch.bmm(adj, x))
        logger.debug(f"LSTM out: {out.shape}")
        return out


####################
# POOLING AND NORM #
####################


class MaxPooling(nn.Module):
    """Max pooling layer."""

    def __init__(self):
        """Initialize the max pooling layer."""
        super().__init__()

    def __call__(self, x):
        """Perform max pooling over the node dimension."""
        return torch.max(x, dim=1)[0]


class GroupNormWrapper(nn.Module):
    """Custom wrapper for GroupNorm to allow for dynamic num_groups."""

    def __init__(self, num_channels, num_groups=4):
        """Initialize the GroupNorm wrapper.

        Args:
            num_channels (int): Number of channels in the input tensor. Should be set within model.
            num_groups (int): Number of groups to use for GroupNorm. (default: 8)
        """
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        """Perform GroupNorm on the input tensor."""
        return self.norm(x)


class DoublyStochasticNormalization(nn.Module):
    """Perform doubly stochastic normalization on edge features.

    Based on Exploiting Edge Features for Graph Neural Networks, Gong and Cheng, 2019
    """

    def __init__(self):
        """Initialize the doubly stochastic normalization layer."""
        super().__init__()

    def forward(self, alpha_p):
        """Perform doubly stochastic normalization on edge features.

        Args :
            alpha_p (Tensor) : alpha of edge feature p of shape (batch, num_edges, num_edges)

        Returns :
            Tensor : alpha_p normalized, shape (batch, num_edges, num_edges)
        """
        norm_alpha_p = alpha_p / torch.sum(alpha_p, dim=2, keepdim=True).clamp(
            min=1
        )
        num = norm_alpha_p * norm_alpha_p
        denom = torch.sum(norm_alpha_p, dim=1, keepdim=True).clamp(min=1)
        return num / denom


#########
# MODEL #
#########
class GCN(nn.Module):
    """Graph Convolutional Neural Network for binary classification."""

    def __init__(
        self,
        num_features,
        conv_dims,
        activation,
        fcn_layers=None,
        pooling="max",
        norm=nn.BatchNorm1d,
        dropout=0.0,
        out_channels=1,
    ):
        """Initialize the GCN model for binary classification.

        Args:
            num_features (int): Number of input node features.
            conv_dims (list of int): Number of hidden features in the convolution layers.
            activation (nn.Module or callable): Activation function to apply.
            fcn_layers (list of int): Number of hidden features in the fully connected layers. Last layer always outputs 2 features.
            pooling (str) : 'mean' or 'max' pooling. (default: 'max')
            norm (nn.Module or callable): Normalization function to apply. (default: nn.BatchNorm1d)
            dropout (float): Dropout probability. (default: 0.0)
            out_channels (int): Number of output channels. (default: 1)
        """
        super().__init__()
        self.num_features = num_features
        self.pool_type = pooling
        self.norm_type = norm

        if fcn_layers is None:
            fcn_layers = [conv_dims[-1]]

        self.convs_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs_layers.append(
            GraphConv(num_features, conv_dims[0], activation)
        )
        self.batch_norms.append(norm(conv_dims[0]))

        for i, conv_dim in enumerate(conv_dims):
            if i == len(conv_dims) - 2:
                self.convs_layers.append(
                    GraphConv(
                        conv_dim, conv_dims[i + 1], activation=nn.Identity()
                    )
                )
                self.batch_norms.append(nn.Identity())
                break

            self.convs_layers.append(
                GraphConv(conv_dim, conv_dims[i + 1], activation)
            )
            self.batch_norms.append(norm(conv_dims[i + 1]))

        logger.info(
            f"Initialized model with {len(self.convs_layers)} graph conv layers"
        )

        self.fcn_layers = nn.ModuleList()

        for i, fcn_dim in enumerate(fcn_layers):
            if i == len(fcn_layers) - 1:
                self.fcn_layers.append(nn.Linear(fcn_dim, out_channels))
                break
            self.fcn_layers.append(nn.Linear(fcn_dim, fcn_layers[i + 1]))

        logger.info(
            f"Initialized model with {len(self.fcn_layers)} fully connected layers"
        )

        self.dropout = nn.Dropout(dropout)
        if pooling == "mean":
            self.pooling = partial(torch.mean, dim=1)  # mean over all nodes
        elif pooling == "max":
            self.pooling = MaxPooling()  # max of each feature dim
        else:
            raise ValueError(f"Pooling {pooling} not supported")

    def forward(self, x, adj, _=None):
        """Perform forward pass for binary classification.

        Args:
            x (Tensor): Input node features of shape (num_nodes, num_features).
            adj (Tensor): Adjacency matrix of the graph, typically sparse, shape (num_nodes, num_nodes).
            _ (Tensor): Unused, here to allow general calls to forward when using edge features.

        Returns:
            Tensor: Predicted edge probabilities for each pair of nodes, shape (num_nodes, num_nodes).
        """
        for i, layer in enumerate(self.convs_layers):
            logger.debug(f"Conv layer: {layer}")
            x = layer(x, adj)
            x = self.dropout(x)
            logger.debug(f"Conv layer output: {x.shape}")
            x = self.batch_norms[i](x.swapaxes(1, 2)).swapaxes(1, 2)

        for fcn_layer in self.fcn_layers:
            x = self.pooling(x)
            logger.debug(f"FCN layer: {fcn_layer}")
            logger.debug(f"FCN layer input: {x.shape}")
            x = fcn_layer(x)
            # x = self.dropout(x) # we might want to control dropout per layer type

        return x.squeeze()


class GraphSAGE(nn.Module):
    """GraphSAGE Neural Network for binary classification."""

    def __init__(
        self,
        num_features,
        conv_dims,
        aggregation,
        activation,
        fcn_layers=None,
        pooling="max",
        norm=nn.BatchNorm1d,
        dropout=0.0,
        out_channels=1,
    ):
        """Initialize the GraphSAGE model for binary classification.

        Args:
            num_features (int): Number of input node features.
            conv_dims (list of int): Number of hidden features in the convolution layers.
            aggregation (nn.Module or callable): Aaggregation function to apply. Shape must be the same as conv_dims.
            activation (nn.Module or callable): Activation function to apply.
            fcn_layers (list of int): Number of hidden features in the fully connected layers. Last layer always outputs 2 features.
            pooling (str) : 'mean' or 'max' pooling. (default: 'max')
            norm (nn.Module or callable): Normalization function to apply. (default: nn.BatchNorm1d)
            dropout (float): Dropout probability. (default: 0.0)
            out_channels (int): Number of output channels. (default: 1)
        """
        super().__init__()
        self.num_features = num_features
        self.pool_type = pooling
        self.norm_type = norm

        if fcn_layers is None:
            fcn_layers = [conv_dims[-1]]

        self.convs_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # INPUT LAYER #
        if isinstance(aggregation, LSTMAggregation):
            logger.debug(
                f"Adding LSTM with dims {self.num_features}, {self.num_features}"
            )
            aggregation = LSTMAggregation.make_lstm_agg(
                num_features, num_features
            )

        self.convs_layers.append(
            GraphSAGEConv(
                num_features,
                conv_dims[0],
                activation=activation,
                aggregation=aggregation,
            )
        )
        self.batch_norms.append(norm(conv_dims[0]))

        for i, conv_dim in enumerate(conv_dims):
            if i == len(conv_dims) - 2:
                # OUTPUT LAYER # no activation
                if isinstance(aggregation, LSTMAggregation):
                    logger.debug(
                        f"Adding LSTM with dims {conv_dim}, {conv_dim}"
                    )
                    aggregation = LSTMAggregation.make_lstm_agg(
                        conv_dim, conv_dim
                    )

                self.convs_layers.append(
                    GraphSAGEConv(
                        conv_dim,
                        conv_dims[i + 1],
                        activation=nn.Identity(),
                        aggregation=aggregation,
                    )
                )
                self.batch_norms.append(nn.Identity())
                break

            # MID LAYERS #
            if isinstance(aggregation, LSTMAggregation):
                logger.debug(f"Adding LSTM with dims {conv_dim}, {conv_dim}")
                aggregation = LSTMAggregation.make_lstm_agg(conv_dim, conv_dim)

            self.convs_layers.append(
                GraphSAGEConv(
                    conv_dim,
                    conv_dims[i + 1],
                    activation=activation,
                    aggregation=aggregation,
                )
            )
            self.batch_norms.append(norm(conv_dims[i + 1]))

        logger.info(
            f"Initialized model with {len(self.convs_layers)} graph conv layers"
        )

        self.fcn_layers = nn.ModuleList()

        for i, fcn_dim in enumerate(fcn_layers):
            if i == len(fcn_layers) - 1:
                self.fcn_layers.append(nn.Linear(fcn_dim, out_channels))
                break
            self.fcn_layers.append(nn.Linear(fcn_dim, fcn_layers[i + 1]))

        logger.info(
            f"Initialized model with {len(self.fcn_layers)} fully connected layers"
        )

        self.dropout = nn.Dropout(dropout)

        if pooling == "mean":
            self.pooling = partial(torch.mean, dim=1)  # mean over all nodes
        elif pooling == "max":
            self.pooling = MaxPooling()  # max of each feature dim
        else:
            raise ValueError(f"Pooling {pooling} not supported")

    def forward(self, x, adj, _=None):
        """Perform forward pass for binary classification.

        Args:
            x (Tensor): Input node features of shape (num_nodes, num_features).
            adj (Tensor): Adjacency matrix of the graph, typically sparse, shape (num_nodes, num_nodes).
            _ (Tensor): Unused, here to allow general calls to forward when using edge features.

        Returns:
            Tensor: Predicted edge probabilities for each pair of nodes, shape (num_nodes, num_nodes).
        """
        for i, layer in enumerate(self.convs_layers):
            logger.debug(f"Running conv layer: {i}")
            logger.debug(f"x shape: {x.shape}")
            logger.debug(f"adj shape: {adj.shape}")
            x = layer(x, adj)
            x = self.dropout(x)
            x = self.batch_norms[i](x.swapaxes(1, 2)).swapaxes(1, 2)

        for fcn_layer in self.fcn_layers:
            x = self.pooling(x)
            x = fcn_layer(x)
            # x = self.dropout(x) # we might want to control dropout per layer type

        return x.squeeze()


class AttentionGCN(nn.Module):
    """Attention-based GCN Neural Network for binary classification."""

    def __init__(
        self,
        num_features,
        conv_dims,
        activation,
        fcn_layers=None,
        pooling="max",
        norm=nn.BatchNorm1d,
        dropout=0.0,
        out_channels=1,
    ):
        """Initialize the GraphSAGE model for binary classification.

        Args:
            num_features (int): Number of input node features.
            conv_dims (list of int): Number of hidden features in the convolution layers.
            activation (nn.Module or callable): Activation function to apply.
            fcn_layers (list of int): Number of hidden features in the fully connected layers. Last layer always outputs 2 features.
            pooling (str) : 'mean' or 'max' pooling. (default: 'max')
            norm (nn.Module or callable): Normalization function to apply. (default: nn.BatchNorm1d)
            dropout (float): Dropout probability. (default: 0.0)
            out_channels (int): Number of output channels. (default: 1)
        """
        super().__init__()
        self.num_features = num_features
        self.pool_type = pooling
        self.norm_type = norm

        if fcn_layers is None:
            fcn_layers = [conv_dims[-1]]

        self.convs_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # INPUT LAYER #
        self.convs_layers.append(
            AttentionConv(
                num_features,
                conv_dims[0],
                activation=activation,
            )
        )
        self.batch_norms.append(norm(conv_dims[0]))

        for i, conv_dim in enumerate(conv_dims):
            if i == len(conv_dims) - 2:
                # OUTPUT LAYER # no activation
                self.convs_layers.append(
                    AttentionConv(
                        # GraphConv(
                        conv_dim,
                        conv_dims[i + 1],
                        activation=nn.Identity(),
                    )
                )
                self.batch_norms.append(nn.Identity())
                break
            # MID LAYERS #
            self.convs_layers.append(
                AttentionConv(
                    # GraphConv(
                    conv_dim,
                    conv_dims[i + 1],
                    activation=activation,
                )
            )
            self.batch_norms.append(norm(conv_dims[i + 1]))

        logger.info(
            f"Initialized model with {len(self.convs_layers)} graph conv layers"
        )

        self.fcn_layers = nn.ModuleList()

        for i, fcn_dim in enumerate(fcn_layers):
            if i == len(fcn_layers) - 1:
                self.fcn_layers.append(nn.Linear(fcn_dim, out_channels))
                break
            self.fcn_layers.append(nn.Linear(fcn_dim, fcn_layers[i + 1]))

        logger.info(
            f"Initialized model with {len(self.fcn_layers)} fully connected layers"
        )

        self.dropout = nn.Dropout(dropout)

        if pooling == "mean":
            self.pooling = partial(torch.mean, dim=1)  # mean over all nodes
        elif pooling == "max":
            self.pooling = MaxPooling()  # max of each feature dim
        else:
            raise ValueError(f"Pooling {pooling} not supported")

    def forward(self, x, adj, _=None):
        """Perform forward pass for binary classification.

        Args:
            x (Tensor): Input node features of shape (num_nodes, num_features).
            adj (Tensor): Adjacency matrix of the graph, typically sparse, shape (num_nodes, num_nodes).
            _ (Tensor): Unused, here to allow general calls to forward when using edge features.

        Returns:
            Tensor: Predicted edge probabilities for each pair of nodes, shape (num_nodes, num_nodes).
        """
        for i, layer in enumerate(self.convs_layers):
            logger.debug(f"Running conv layer: {i}")
            logger.debug(f"x shape: {x.shape}")
            logger.debug(f"adj shape: {adj.shape}")
            x = layer(x, adj)
            x = self.dropout(x)
            logger.debug(f"Conv layer output: {x.shape}")
            x = self.batch_norms[i](x.swapaxes(1, 2)).swapaxes(1, 2)

        for fcn_layer in self.fcn_layers:
            x = self.pooling(x)
            x = fcn_layer(x)
            # x = self.dropout(x) # we might want to control dropout per layer type

        return x.squeeze()


class EdgeAttentionGCN(nn.Module):
    """Edge-enhanced attention-based GCN Neural Network for binary classification."""

    def __init__(
        self,
        num_features,
        conv_dims,
        activation,
        fcn_layers=None,
        pooling="max",
        norm=nn.BatchNorm1d,
        dropout=0.0,
        out_channels=1,
    ):
        """Initialize the Edge-Enchanced Attention GCN model for binary classification.

        Args:
            num_features (int): Number of input node features.
            conv_dims (list of int): Number of hidden features in the convolution layers.
            activation (nn.Module or callable): Activation function to apply.
            fcn_layers (list of int): Number of hidden features in the fully connected layers. Last layer always outputs 2 features.
            pooling (str) : 'mean' or 'max' pooling. (default: 'max')
            norm (nn.Module or callable): Normalization function to apply. (default: nn.BatchNorm1d)
            dropout (float): Dropout probability. (default: 0.0)
            out_channels (int): Number of output channels. (default: 1)
        """
        super().__init__()
        self.num_features = num_features
        self.pool_type = pooling
        self.norm_type = norm

        if fcn_layers is None:
            fcn_layers = [conv_dims[-1]]

        self.convs_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # INPUT LAYER #
        self.convs_layers.append(
            EdgeConv(
                num_features,
                conv_dims[0],
                activation=activation,
            )
        )
        self.batch_norms.append(norm(conv_dims[0]))

        for i, conv_dim in enumerate(conv_dims):
            if i == len(conv_dims) - 2:
                # OUTPUT LAYER # no activation
                self.convs_layers.append(
                    EdgeConv(
                        conv_dim,
                        conv_dims[i + 1],
                        activation=nn.Identity(),
                    )
                )
                self.batch_norms.append(nn.Identity())
                break
            # MID LAYERS #
            self.convs_layers.append(
                EdgeConv(
                    conv_dim,
                    conv_dims[i + 1],
                    activation=activation,
                )
            )
            self.batch_norms.append(norm(conv_dims[i + 1]))

        logger.info(
            f"Initialized model with {len(self.convs_layers)} graph conv layers"
        )

        self.fcn_layers = nn.ModuleList()

        for i, fcn_dim in enumerate(fcn_layers):
            if i == len(fcn_layers) - 1:
                self.fcn_layers.append(nn.Linear(fcn_dim, out_channels))
                break
            self.fcn_layers.append(nn.Linear(fcn_dim, fcn_layers[i + 1]))

        logger.info(
            f"Initialized model with {len(self.fcn_layers)} fully connected layers"
        )

        self.dropout = nn.Dropout(dropout)

        if pooling == "mean":
            self.pooling = partial(torch.mean, dim=1)  # mean over all nodes
        elif pooling == "max":
            self.pooling = MaxPooling()  # max of each feature dim
        else:
            raise ValueError(f"Pooling {pooling} not supported")

    def forward(self, x, adj, E):
        """Perform forward pass for binary classification.

        Args:
            x (Tensor): Input node features of shape (num_nodes, num_features).
            adj (Tensor): Adjacency matrix of the graph, typically sparse, shape (num_nodes, num_nodes).
            E (Tensor): Edge features of shape (num_edges, num_edges, num_features)

        Returns:
            Tensor: Predicted edge probabilities for each pair of nodes, shape (num_nodes, num_nodes).
        """
        for i, layer in enumerate(self.convs_layers):
            logger.debug(f"Running conv layer: {i}")
            logger.debug(f"x shape: {x.shape}")
            logger.debug(f"adj shape: {adj.shape}")
            logger.debug(f"E shape: {E.shape}")
            x, E = layer(
                x, adj, E.clone()
            )  # use attention scores as edge features in next layer (eq.12)
            x = self.dropout(x)
            x = self.batch_norms[i](x.swapaxes(1, 2)).swapaxes(1, 2)

        for fcn_layer in self.fcn_layers:
            x = self.pooling(x)
            x = fcn_layer(x)
            # x = self.dropout(x) # we might want to control dropout per layer type

        return x.squeeze()
