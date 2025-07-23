import numpy as np
import torch
import torch.nn as nn
# import torch.backends.cudnn as cudnn
# cudnn.deterministic = True
# cudnn.benchmark = True
import torch.nn.functional as F

from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from collections import OrderedDict
from tqdm import tqdm
from torch_geometric.nn.norm import BatchNorm



class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 heads: int = 1,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0,
                 add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads  # 1
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        # if isinstance(in_channels, int):
        #     self.lin_src = Linear(in_channels, heads * out_channels,
        #                           bias=False, weight_initializer='glorot')
        #     self.lin_dst = self.lin_src
        # else:
        #     self.lin_src = Linear(in_channels[0], heads * out_channels, False,
        #                           weight_initializer='glorot')
        #     self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
        #                           weight_initializer='glorot')

        self.lin_src = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)  # initialize  Xavier初始化:通过网络层时，输入和输出的方差相同
        self.lin_dst = self.lin_src


        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)  # initialize
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)  # initialize

        # if bias and concat:
        #     self.bias = Parameter(torch.Tensor(heads * out_channels))
        # elif bias and not concat:
        #     self.bias = Parameter(torch.Tensor(out_channels))
        # else:
        #     self.register_parameter('bias', None)

        self._alpha = None
        self.attentions = None

        # self.reset_parameters()

    # def reset_parameters(self):
    #     self.lin_src.reset_parameters()
    #     self.lin_dst.reset_parameters()
    #     glorot(self.att_src)
    #     glorot(self.att_dst)
    #     # zeros(self.bias)

    def forward(self,
                x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                size: Size = None,
                return_attention_weights=None,
                attention=True,
                tied_attention=None):
        '''
        type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        '''
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # 1.set the output dim and heads num
        H, C = self.heads, self.out_channels  # H=1, C=512

        # 2.nodes' feature put into the MLP (source node and target node)
        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"  # If False, print the string
            # x_src = x_dst = self.lin_src(x).view(-1, H, C)
            x_src = x_dst = torch.mm(x, self.lin_src).view(-1, H, C)  # 3590*512 -> 3590*1*512
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # 3.No attention, output the nodes' feature
        if not attention:  # attention == True  # Conv2 4's output
            return x[0].mean(dim=1)
            # return x[0].view(-1, self.heads * self.out_channels)

        # 4.calculate the attention
        if tied_attention == None:  # Conv3 used this IF
            # Next, we compute node-level attention coefficients, both for source
            # and target nodes (if present):
            alpha_src = (x_src * self.att_src).sum(dim=-1)  # 3590*1*512 -> 3590*1
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)  # 3590*1*512 -> 3590*1
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            alpha = tied_attention

        # break: A=A+I
        if self.add_self_loops:  # self.add_self_loops==True
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # 5.use the attention, adj, x to calculate the output
        # Conv1 3's output
        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        # multi head results need to cat
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # if self.bias is not None:
        #     out += self.bias

        # estimate the edge_index is Sparse or not
        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    '''calclulate the attention weighted edge_index '''
    def message(self, x_j: Tensor,
                alpha_j: Tensor,
                alpha_i: OptTensor,
                index: Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # Given egel-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i  # shape:(edge_num,1)

        # alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = torch.sigmoid(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)  # 45310*1*512  *  45310  = 45310*1*512

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GATE_Encoder_3Layers(nn.Module):
    def __init__(self, hidden_dims):
        super(GATE_Encoder_3Layers, self).__init__()
        assert len(hidden_dims) == 4, "hidden_dims should have 4 dimensions"
        # encoder
        self.conv1 = GATConv(hidden_dims[0], hidden_dims[1], heads=1, concat=False,
                             dropout=0.1, add_self_loops=False, bias=False)
        self.bn1 = BatchNorm(hidden_dims[1])
        self.conv2 = GATConv(hidden_dims[1], hidden_dims[2], heads=1, concat=False,
                             dropout=0.1, add_self_loops=False, bias=False)
        self.bn2 = BatchNorm(hidden_dims[2])
        self.conv3 = GATConv(hidden_dims[2], hidden_dims[3], heads=1, concat=False,
                             dropout=0.1, add_self_loops=False, bias=False)
        self.bn3 = BatchNorm(hidden_dims[3])


    def forward(self, features, edge_index):
        h1 = F.elu(self.bn1(self.conv1(features, edge_index))) 
        h2 = F.elu(self.bn2(self.conv2(h1, edge_index)))
        h3 = self.bn3(self.conv3(h2, edge_index))
        return h1, h2, h3

    

class GATE_Decoder_3Layers(nn.Module):
    def __init__(self, hidden_dims):
        super(GATE_Decoder_3Layers, self).__init__()
        # decoder
        self.conv3_ = GATConv(hidden_dims[3], hidden_dims[2], heads=1, concat=False,
                              dropout=0.1, add_self_loops=False, bias=False)
        self.bn3_ = BatchNorm(hidden_dims[2])
        self.conv2_ = GATConv(hidden_dims[2], hidden_dims[1], heads=1, concat=False,
                              dropout=0.1, add_self_loops=False, bias=False)
        self.bn2_ = BatchNorm(hidden_dims[1])
        self.conv1_ = GATConv(hidden_dims[1], hidden_dims[0], heads=1, concat=False,
                              dropout=0.1, add_self_loops=False, bias=False)
        self.bn1_ = BatchNorm(hidden_dims[0])

    def forward(self, h1, h2, h3, edge_index):
        h3_ = F.elu(self.bn3_(self.conv3_(h3, edge_index)))

        h2_fused = h3_ + h2
        h2_ = F.elu(self.bn2_(self.conv2_(h2_fused, edge_index)))

        h1_fused = h2_ + h1
        h1_ = self.bn1_(self.conv1_(h1_fused, edge_index))
        return h1_
    


class SpaCon(torch.nn.Module):
    def __init__(self, hidden_dims, fusion_method='concat'):
        super(SpaCon, self).__init__()

        assert fusion_method in ['concat', 'add'], f"Unsupported fusion_method: '{fusion_method}'"
        assert len(hidden_dims) == 4, "hidden_dims should have 4 dimensions"
        
        # fusion_method = 'concat' or 'add'
        self.fusion_method = fusion_method
        # Encoder
        self.encoder_spa = GATE_Encoder_3Layers(hidden_dims)

        self.encoder_con = GATE_Encoder_3Layers(hidden_dims)
        # Decoder
        if fusion_method == 'concat':
            self.decoder = GATE_Decoder_3Layers(hidden_dims[:-1] + [2*hidden_dims[-1]])
        elif self.fusion_method == 'add':
            self.decoder = GATE_Decoder_3Layers(hidden_dims)


    def forward(self, features, edge_index_con, edge_index_spa): 
        # Connect Encoder
        h1_con, h2_con, h3_con = self.encoder_con(features, edge_index_con)
        # Spatial Encoder
        h1_spa, h2_spa, h3_spa = self.encoder_spa(features, edge_index_spa)

        h1_fused = h1_con + h1_spa
        h2_fused = h2_con + h2_spa

        # Features Fusion
        if self.fusion_method == 'concat':
            encode_features = torch.cat((h3_con, h3_spa), dim=1)
        elif self.fusion_method == 'add':
            h3_con_norm = torch.nn.functional.normalize(h3_con, p=2, dim=1)
            h3_spa_norm = torch.nn.functional.normalize(h3_spa, p=2, dim=1)
            encode_features = h3_con_norm + h3_spa_norm

        # Decoder
        features_hat = self.decoder(h1_fused, h2_fused, encode_features, edge_index_con)
        return h3_con, h3_spa, features_hat

    @torch.no_grad()
    def spa_inference(self, x_all, subgraph_loader,
                  device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        pbar = tqdm(total=len(subgraph_loader.dataset) * 3)   # layers' number
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i in range(3):  # layers' number
            xs = []
            for batch in subgraph_loader:
                # slice the x_all use the n_id
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                # calculate the x used by graph
                if i == 0:
                    x = F.elu(self.encoder_spa.bn1(self.encoder_spa.conv1(x, batch.edge_index.to(device))))
                elif i == 1:
                    x = F.elu(self.encoder_spa.bn2(self.encoder_spa.conv2(x, batch.edge_index.to(device))))
                elif i == 2:
                    x = self.encoder_spa.bn3(self.encoder_spa.conv3(x, batch.edge_index.to(device)))
                # add the calculated node features to the list
                # slice the feature use the batch.batch_size
                xs.append(x[:batch.batch_size].cpu())
                # print('x[:batch.batch_size].cpu().shape:',x[:batch.batch_size].cpu().shape)  # torch.Size([512, 30])
                # update the bar
                pbar.update(batch.batch_size)
            # select the last layer output
            # transform the [tensor,tensor,..] to a entire tensor
            x_all = torch.cat(xs, dim=0)
            # print('torch.tensor(xs).shape:', len(xs))
            # print('x_all.shape:', x_all.shape)
        pbar.close()
        return x_all
    
    @torch.no_grad()    
    def con_inference(self, x_all, subgraph_loader,
                  device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        pbar = tqdm(total=len(subgraph_loader.dataset) * 3)   # layers' number
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i in range(3):  # layers' number
            xs = []
            for batch in subgraph_loader:
                # slice the x_all use the n_id
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                # calculate the x used by graph
                if i == 0:
                    x = F.elu(self.encoder_con.bn1(self.encoder_con.conv1(x, batch.edge_index.to(device))))
                elif i == 1:
                    x = F.elu(self.encoder_con.bn2(self.encoder_con.conv2(x, batch.edge_index.to(device))))
                elif i == 2:
                    x = self.encoder_con.bn3(self.encoder_con.conv3(x, batch.edge_index.to(device)))
                # add the calculated node features to the list
                # slice the feature use the batch.batch_size
                xs.append(x[:batch.batch_size].cpu())
                # print('x[:batch.batch_size].cpu().shape:',x[:batch.batch_size].cpu().shape)  # torch.Size([512, 30])
                # update the bar
                pbar.update(batch.batch_size)
            # select the last layer output
            # transform the [tensor,tensor,..] to a entire tensor
            x_all = torch.cat(xs, dim=0)
            # print('torch.tensor(xs).shape:', len(xs))
            # print('x_all.shape:', x_all.shape)
        pbar.close()
        return x_all
    


