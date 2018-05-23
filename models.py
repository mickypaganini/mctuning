import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

# modified from t-ae's gist
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims): #, mean=False):
        super(MinibatchDiscrimination, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        # self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        nn.init.normal(self.T, 0, 1)

    def forward(self, x, w):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (w.unsqueeze(2).expand(list(w.shape) + [self.out_features]) * expnorm).sum(0)  # NxB

        o_b = o_b - w.transpose(1,0).expand(list(w.shape) + [self.out_features])# subtract self-distance
        # if self.mean:
        #     o_b /= x.size(0) - 1

        return o_b.squeeze(0)


class NTrackModel(nn.Module):

    def __init__(self, input_size):
        super(NTrackModel, self).__init__()

        self.dense = nn.Linear(input_size, 32)
        self.dropout = nn.Dropout(p=0.2)
#         self.dense0 = nn.Linear(32, 64)
#         self.dropout0 = nn.Dropout(p=0.7)
#         self.dense1 = nn.Linear(64, 64)
        self.dense1 = nn.Linear(32, 2)

#        self.mbd = MinibatchDiscrimination(in_features=2, out_features=6, kernel_dims=8)
        
#         self.dense2 = nn.Linear(64 * 4, 32) # 4 is from the concat below
        # self.dropout2 = nn.Dropout(p=0.5)
#         self.dense3 = nn.Linear(32, 1)

        self.dense3 = nn.Linear(3 * 2, 1)
        #self.dense3 = nn.Linear(2 + 6, 1)


    def forward(self, inputs, batch_weights, batch_size):

        inputs = (inputs - 50) / 100. 

        hidden = F.relu(
            self.dense1(
#                 self.dropout0(
#                     F.relu(
#                         self.dense0(
                            self.dropout(
                                F.relu(
                                    self.dense(
                                        inputs)))))#)))

        # batch_features = batch_weights.mm(hidden).expand(batch_size, hidden.shape[-1])
        # weighted_mult = batch_weights.transpose(0, 1).expand(-1, hidden.shape[-1]) * hidden
        # std = torch.std(weighted_mult, 0).expand(batch_size, -1)
        ###
        batch_mean = batch_weights.mm(hidden).expand(batch_size, hidden.shape[-1])
        batch_second_moment = batch_weights.mm(torch.pow(hidden, 2)).expand(batch_size, hidden.shape[-1])
        batch_std = batch_second_moment - torch.pow(batch_mean, 2)
        all_features = torch.cat([batch_mean, batch_std, hidden], 1)
        ###
        #mbd_features = self.mbd(hidden, batch_weights)
        #all_features = torch.cat([hidden, mbd_features], 1)


        # self.batch_features = all_features       
        outputs = self.dense3(
            # self.dropout2(
#                 F.relu(
#                   self.dense2(
                        all_features)#))
        return outputs


class DoubleLSTM(nn.Module):

    def __init__(self, input_size, output_size, num_layers, dropout,
                 bidirectional, batch_size, tagger_output_size):
        super(DoubleLSTM, self).__init__()

        # members
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.batch_size = batch_size

        # LSTM layers
        self.lstm_lead = nn.LSTM(
            input_size=input_size,
            hidden_size=output_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.lstm_sublead = nn.LSTM(
            input_size=input_size,
            hidden_size=output_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        #self.mbd = MinibatchDiscrimination(
        #    in_features=3 * (output_size * self.num_directions * 2),
        #    out_features= 3 * (output_size * self.num_directions * 2),
        #    kernel_dims=8)

        # output dense layer
        self.dense = nn.Linear(
            # (3 *) because of torch.cat; (* 2) because of 2 streams
            3 * (output_size * self.num_directions * 2),
            128 #tagger_output_size#128
        )
        self.dropout = nn.Dropout(p=0.5)
        self.dropout1 = nn.Dropout(p=0.5)


    #     self.init_layers()

        self.dense2 = nn.Linear(128, tagger_output_size)

    # def init_layers(self):
    #     nn.init.xavier_uniform(self.dense.weight, gain=0.5)
    #     self.dense.bias.data.zero_()
    #     self.init_lstm()

    # def init_lstm(self):
    #     for l in [self.lstm_lead, self.lstm_sublead]:
    #         # recurrent kernel init
    #         nn.init.orthogonal(l.weight_hh_l0)

    #         # in -> h kernel init
    #         nn.init.xavier_uniform(l.weight_ih_l0)

    #         # Initialize the forget gate to one
    #         for bias in [l.bias_hh_l0, l.bias_ih_l0]:
    #             bias.data.zero_()
    #             # bias.data[l.hidden_size:2 * l.hidden_size] = 1.0

    def forward(self, leading_jets, subleading_jets, lengths,
                batch_weights, batch_size):

        leading_jets = leading_jets[:, :, [5, 9]] # pt and dr
        subleading_jets = subleading_jets[:, :, [5, 9]] # pt and dr

        if torch.cuda.is_available():
            leading_jets.cuda()
            subleading_jets.cuda()
            lengths.cuda()

        lead_len = lengths[:, 0]
        sublead_len = lengths[:, 1]

        # sort the data by length of sequence in decreasing order (because PyTorch)
        lead_len, lead_sort = torch.sort(lead_len, descending=True)
        sublead_len, sublead_sort = torch.sort(sublead_len, descending=True)

        _, lead_unsort = torch.sort(lead_sort)
        _, sublead_unsort = torch.sort(sublead_sort)

        leading_jets = leading_jets[lead_sort]
        subleading_jets = subleading_jets[sublead_sort]

        # Pack sequences
        packed_leading_jets = nn.utils.rnn.pack_padded_sequence(
            input=leading_jets,
            batch_first=True,
            lengths=lead_len.tolist()
        )
        packed_subleading_jets = nn.utils.rnn.pack_padded_sequence(
            input=subleading_jets,
            batch_first=True,
            lengths=sublead_len.tolist()
        )
        
        # LSTMs return (nb_layers * directions, batch, features)
        _, (h_lead, _) = self.lstm_lead(packed_leading_jets)
        # _, (h_lead, _) = self.lstm_lead(leading_jets)
        _, (h_sublead, _) = self.lstm_sublead(packed_subleading_jets)
        # _, (h_sublead, _) = self.lstm_sublead(subleading_jets) # n_layers * n_dir X batch_size X n_feat

        # Only need the last LSTM layer, and want batch first
        # Turns to (batch, directions, features)
        h_lead = h_lead[-self.num_directions:].transpose(0, 1)
        h_sublead = h_sublead[-self.num_directions:].transpose(0, 1)

        # Turns to (batch, directions * features)
        h_lead = h_lead.contiguous().view(batch_size, -1)
        h_sublead = h_sublead.contiguous().view(batch_size, -1)

        # h_lead = get_last_h(self.lstm_lead, packed_leading_jets, lead_len)
        # h_sublead = get_last_h(self.lstm_sublead, packed_subleading_jets, sublead_len)

        # sort back
        h_lead = h_lead[lead_unsort]
        h_sublead = h_sublead[sublead_unsort]

        # concatenate outputs of the 2 LSTMs
        hidden = torch.cat([h_lead, h_sublead], 1)
        # hidden = F.relu(hidden)

        batch_mean = batch_weights.mm(hidden).expand(
             batch_size, hidden.shape[-1])
        batch_second_moment = batch_weights.mm(
             torch.pow(hidden, 2)).expand(batch_size, hidden.shape[-1])
        batch_std = batch_second_moment - torch.pow(batch_mean, 2)

        all_features = torch.cat([batch_mean, batch_std, hidden], 1)

        #mbd_features = self.mbd(hidden, batch_weights)
        #all_features = torch.cat([hidden, mbd_features], 1)

        # self.batch_features = batch_features
        outputs = self.dense2(self.dropout1(F.relu(self.dense(self.dropout(all_features)))))
        # outputs = self.dense(batch_features)
        return outputs


class Conv1DModel(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_output_size, kernel_size,
                 dropout=0.0, bidirectional=False):
        super(Conv1DModel, self).__init__()

        # members
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size)
        self.conv2 = nn.Conv1d(input_size, hidden_size, kernel_size)
        self.rnn1 = nn.GRU(
            input_size=hidden_size,
            hidden_size=rnn_output_size,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.rnn2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=rnn_output_size,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.n_directions = 2 if bidirectional else 1
        self.dense = nn.Linear(
            # (3 *) because of torch.cat; (* 2) because of 2 streams
            3 * (rnn_output_size * self.n_directions * 2),
            16
        )
        self.dropout = nn.Dropout(p=0.5)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dense2 = nn.Linear(16, 1)


    def forward(self, leading_jets, subleading_jets, batch_weights):
        batch_size = batch_weights.shape[1]
        c1 = self.conv1(leading_jets.transpose(1, 2)) # swap seq_length and features
        c2 = self.conv2(subleading_jets.transpose(1, 2))
        
        # take the last time step only
        _, r1 = self.rnn1(c1.transpose(1, 2)) # swap back
        _, r2 = self.rnn2(c2.transpose(1, 2))
        r1 = r1[-self.n_directions:].transpose(0, 1).contiguous().view(batch_size, -1) # [b, dir x feats]
        r2 = r2[-self.n_directions:].transpose(0, 1).contiguous().view(batch_size, -1)
        # concat rnn outputs
        hidden = torch.cat([r1, r2], 1)

        # batch-level features
        batch_mean = batch_weights.mm(hidden).expand(
            batch_size, hidden.shape[-1])
        batch_second_moment = batch_weights.mm(
            torch.pow(hidden, 2)).expand(batch_size, hidden.shape[-1])
        batch_std = batch_second_moment - torch.pow(batch_mean, 2)
        
        all_features = torch.cat([batch_mean, batch_std, hidden], 1)
        
        # fully-connected layers
        outputs = self.dense2(self.dropout1(F.relu(self.dense(self.dropout(all_features)))))
        return outputs


class BeefyConv1DModel(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_output_size, kernel_size,
                 dropout=0.0, bidirectional=False):
        super(BeefyConv1DModel, self).__init__()

        # members
        self.conv1 = nn.Conv1d(input_size, 32, 20)# hidden_size, kernel_size)
        self.conv2 = nn.Conv1d(input_size, 32, 20)# hidden_size, kernel_size)

        self.conv3 = nn.Conv1d(32, 32, 16)# hidden_size, kernel_size)
        self.conv4 = nn.Conv1d(32, 32, 16)# hidden_size, kernel_size)
        



        self.rnn1 = nn.GRU(
            input_size=32, # hidden_size
            hidden_size=48, #rnn_output_size,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.rnn2 = nn.GRU(
            input_size=32, # hidden_size
            hidden_size=48, #rnn_output_size,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.n_directions = 2 if bidirectional else 1
        self.dense = nn.Linear(
            # (3 *) because of torch.cat; (* 2) because of 2 streams
            #3 * (rnn_output_size * self.n_directions * 2),
            3 * 48 * self.n_directions * 2,
            32
        )
        self.dropout = nn.Dropout(p=0.5)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dense2 = nn.Linear(32, 1)


    def forward(self, leading_jets, subleading_jets, batch_weights):
        batch_size = batch_weights.shape[1]
        
        c1 = self.conv1(leading_jets.transpose(1, 2)) # swap seq_length and features
        c2 = self.conv2(subleading_jets.transpose(1, 2))

        c1 = nn.Dropout(p=0.5)(
            F.leaky_relu(
                nn.MaxPool1d(kernel_size=2, stride=2)(c1)))
        c2 = nn.Dropout(p=0.5)(
            F.leaky_relu(
                nn.MaxPool1d(kernel_size=2, stride=2)(c2)))

        c3 = self.conv4(c1)
        c4 = self.conv4(c2)

        c3 = nn.Dropout(p=0.6)(
            F.leaky_relu(
                nn.MaxPool1d(kernel_size=2, stride=2)(c3)))
        c4 = nn.Dropout(p=0.6)(
            F.leaky_relu(
                nn.MaxPool1d(kernel_size=2, stride=2)(c4)))
        
        # take the last time step only
        _, r1 = self.rnn1(c3.transpose(1, 2)) # swap back
        _, r2 = self.rnn2(c4.transpose(1, 2))
        r1 = r1[-self.n_directions:].transpose(0, 1).contiguous().view(batch_size, -1) # [b, dir x feats]
        r2 = r2[-self.n_directions:].transpose(0, 1).contiguous().view(batch_size, -1)
        # concat rnn outputs
        hidden = torch.cat([r1, r2], 1)
        hidden = nn.Dropout(p=0.7)(hidden)

        # batch-level features
        batch_mean = batch_weights.mm(hidden).expand(
            batch_size, hidden.shape[-1])
        batch_second_moment = batch_weights.mm(
            torch.pow(hidden, 2)).expand(batch_size, hidden.shape[-1])
        batch_std = batch_second_moment - torch.pow(batch_mean, 2)
        
        all_features = torch.cat([batch_mean, batch_std, hidden], 1)
        
        # fully-connected layers
        outputs = self.dense2(self.dropout1(F.relu(self.dense(self.dropout(all_features)))))
        return outputs



            
