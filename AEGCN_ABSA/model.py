import yaml
import math
import mindspore as ms
from mindspore import ops, nn, Tensor, Parameter
from mindspore.common.initializer import initializer, HeUniform, Uniform, Normal, _calculate_fan_in_and_fan_out
from AEGCN_ABSA.utils.init import XavierNormal



class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True, activation=None):
        super().__init__(in_channels, out_channels, weight_init='normal', bias_init='zeros', has_bias=has_bias,
                         activation=activation)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape))
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))


class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal', dtype=ms.float32,
                 padding_idx=None):
        if embedding_table == 'normal':
            embedding_table = Normal(1.0)
        super().__init__(vocab_size, embedding_size, use_one_hot, embedding_table, dtype, padding_idx)

    @classmethod
    def from_pretrained_embedding(cls, embeddings: Tensor, freeze=True, padding_idx=None):
        rows, cols = embeddings.shape
        embedding = cls(rows, cols, embedding_table=embeddings, padding_idx=padding_idx)
        embedding.embedding_table.requires_grad = not freeze

        return embedding



class Attention(nn.Cell):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = Dense(embed_dim, n_head * hidden_dim)
        self.w_q = Dense(embed_dim, n_head * hidden_dim)
        self.proj = Dense(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(1 - dropout)
        if score_function == 'mlp':
            self.weight = Parameter(ops.ones((2 * self.hidden_dim), ms.float32))
        elif self.score_function == 'bi_linear':
            self.weight = Parameter(ops.ones((self.hidden_dim, self.hidden_dim), ms.float32))
        else:  # dot_product / scaled_dot_product
            self.weight = None

        self.expandims = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.bmm = ops.BatchMatMul()
        self.div = ops.Div()
        self.softmax = ops.Softmax(-1)
        self.cat = ops.Concat(-1)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.set_data(initializer(Uniform(stdv), self.weight.shape))

    def construct(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = self.expandims(q, 1)
        if len(k.shape) == 2:  # k_len missing
            k = self.expandims(k, 1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = self.transpose(kx, (2, 0, 1, 3)).view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = self.transpose(qx, (2, 0, 1, 3)).view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = self.transpose(kx, (0, 2, 1))
            score = self.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = self.transpose(kx, (0, 2, 1))
            qkt = self.bmm(qx, kt)
            score = self.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = ops.BroadcastTo((-1, q_len, -1, -1))(self.expandims(kx, 1))
            qxx = ops.BroadcastTo((-1, -1, k_len, -1))(self.expandims(qx, 2))
            kq = ops.Concat(-1)((kxx, qxx))  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = ops.tanh(ops.matmul(kq.astype(ms.float16), self.weight.astype(ms.float16)).astype(ms.float32))
        elif self.score_function == 'bi_linear':
            qw = ops.matmul(qx.astype(ms.float16), self.weight.astype(ms.float16)).astype(ms.float32)
            kt = self.transpose(kx, (0, 2, 1))
            score = self.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = self.softmax(score)
        output = self.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = ops.Concat(-1)(ops.split(output, axis=0, output_num=3))  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


class PositionwiseFeedForward_GCN(nn.Cell):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_hid, d_inner_hid=None, dropout=0):
        super(PositionwiseFeedForward_GCN, self).__init__()
        if d_inner_hid is None:
            d_inner_hid = d_hid
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.dropout = nn.Dropout(1 - dropout)
        self.relu = nn.ReLU()

    def construct(self, x):
        output = self.relu(self.w_1(ops.transpose(x, (0, 2, 1))))
        output = ops.transpose(output, (0, 2, 1))

        output = self.dropout(output)
        return output


class GraphConvolution(nn.Cell):
    """
    Similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, dropout, bias=True):
        super(GraphConvolution, self).__init__()
        #self.opt = opt
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(ops.ones((in_features, out_features), ms.float32))

        self.ffn_gcn = PositionwiseFeedForward_GCN(in_features, dropout=dropout)    #
        if bias:
            self.bias = Parameter(ops.ones((out_features), ms.float32))
        else:
            self.bias = None

        self.sum = ops.ReduceSum(keep_dims=True)

    def construct(self, text, adj):
        hidden = ops.matmul(text.astype(ms.float16), self.weight.astype(ms.float16)).astype(ms.float32)
        denom = self.sum(adj, 2) + 1
        output = ops.matmul(adj.astype(ms.float16), hidden.astype(ms.float16)).astype(ms.float32) / denom
        # output = self.attn_g(output,output)
        output = self.ffn_gcn(output)   #

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Model(nn.Cell):
    def __init__(self, embedding_matrix):
        super(Model, self).__init__()
        #self.opt = opt
        with open('./AEGCN_ABSA/config.yaml', 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.embed = Embedding.from_pretrained_embedding(Tensor(embedding_matrix, ms.float32))
        #self.squeeze_embedding = SqueezeEmbedding()  #
        self.text_lstm = nn.LSTM(self.cfg['embed_dim'], self.cfg['hidden_dim'], num_layers=1, batch_first=True, bidirectional=True)
        self.aspect_lstm = nn.LSTM(self.cfg['embed_dim'], self.cfg['hidden_dim'], num_layers=1, batch_first=True, bidirectional=True)
        self.attn_k = Attention(self.cfg['embed_dim'] * 2, out_dim=self.cfg['hidden_dim'], n_head=self.cfg['head'], score_function='mlp',
                                dropout=self.cfg['dropout'])  #
        self.attn_a = Attention(self.cfg['embed_dim'] * 2, out_dim=self.cfg['hidden_dim'], n_head=self.cfg['head'], score_function='mlp',
                                dropout=self.cfg['dropout'])  #
        # self.attn_s1 = Attention(opt.embed_dim*2, out_dim=opt.hidden_dim, n_head=3, score_function='mlp', dropout=0.5)

        self.attn_q = Attention(self.cfg['embed_dim']*2, out_dim=self.cfg['hidden_dim'], n_head=self.cfg['head'], score_function='mlp',
                                dropout=self.cfg['dropout'])  #

        self.gc1 = GraphConvolution(2*self.cfg['hidden_dim'], 2*self.cfg['hidden_dim'], self.cfg['dropout'])
        self.gc2 = GraphConvolution(2*self.cfg['hidden_dim'], 2*self.cfg['hidden_dim'], self.cfg['dropout'])
        self.attn_k_q = Attention(self.cfg['hidden_dim'], n_head=self.cfg['head'], score_function='mlp', dropout=self.cfg['dropout']) #
        self.attn_k_a = Attention(self.cfg['hidden_dim'], n_head=self.cfg['head'], score_function='mlp', dropout=self.cfg['dropout'])
        #self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)

        self.text_embed_dropout = nn.Dropout(1 - self.cfg['dropout'])
        self.aspect_embed_dropout = nn.Dropout(1 - self.cfg['dropout'])

        self.dense = Dense(self.cfg['hidden_dim'] * 3, self.cfg['polarities_dim']).to_float(ms.float16)

        self.relu = ops.ReLU()
        self.cast = ops.Cast()
        self.div = ops.Div()
        self.sum = ops.ReduceSum()
        self.cat = ops.Concat(-1)
        self.expand = ops.ExpandDims()

        self.reset_parameters()

    def reset_parameters(self):
        for param in self.get_parameters():
            if param.requires_grad:
                if len(param.shape) > 1:
                    param.set_data(initializer(XavierNormal(), param.shape))
                else:
                    stdv = 1. / math.sqrt(param.shape[0])
                    param.set_data(initializer(Uniform(stdv), param.shape))

    def construct(self, inputs):
        text_indices = inputs[self.cfg['input_columns'][0]]
        text_len = inputs[self.cfg['input_columns'][1]]
        aspect_indices = inputs[self.cfg['input_columns'][2]]
        aspect_len = inputs[self.cfg['input_columns'][3]]
        adj = inputs[self.cfg['input_columns'][4]]
        weight = inputs[self.cfg['input_columns'][5]]
        
        text = self.embed(text_indices)
        # text = self.squeeze_embedding(text, text_len)
        text = self.text_embed_dropout(text)
        aspect = self.embed(aspect_indices)    #
        # aspect = self.aspect_embed_dropout(aspect)    #
        #aspect = self.squeeze_embedding(aspect, aspect_len)
        text_out, (_, _) = self.text_lstm(text, seq_length=text_len)
        aspect_out, (_, _) = self.aspect_lstm(aspect, seq_length=aspect_len)  #add aspect
        #hid_context = self.squeeze_embedding(text_out, text_len)   #
        #hid_aspect = self.squeeze_embedding(aspect_out, aspect_len)  #




        hc, _ = self.attn_k(text_out, text_out)   #


        ha, _ = self.attn_a(aspect_out, aspect_out)



        weight = self.expand(weight, 2)
        x = self.relu(self.gc1(weight * text_out, adj))
        x = self.relu(self.gc2(weight * x, adj))

        #x = self.squeeze_embedding(x, text_len)
        hg, _ = self.attn_q(x, x)


        hc_hg, _ = self.attn_k_q(hc, hg)

        hg_ha, _ = self.attn_k_a(hg, ha)

        text_len = self.cast(text_len, ms.float32).copy()
        aspect_len = self.cast(aspect_len, ms.float32).copy()
        #text_len = text_len.float().clone().detach().to(self.opt.device)
        #aspect_len = aspect_len.float().clone().detach().to(self.opt.device)
        #text_len = torch.tensor(text_len, dtype=torch.float).to(self.opt.device)
        #aspect_len = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)

        hc_mean = self.div(self.sum(hc, 1), text_len.view(text_len.shape[0], 1))

        hc_hg_mean = self.div(self.sum(hc_hg, 1), text_len.view(text_len.shape[0], 1))

        hg_ha_mean = self.div(self.sum(hg_ha, 1), aspect_len.view(text_len.shape[0], 1))

        final_x = self.cat((hc_hg_mean,hc_mean, hg_ha_mean))


        output = self.dense(final_x)
        return output.astype(ms.float32)