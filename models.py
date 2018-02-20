import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

class NTrackModel(nn.Module):

    def __init__(self, input_size):
        super(NTrackModel, self).__init__()

        self.dense = nn.Linear(input_size, 32)
        self.dropout = nn.Dropout(p=0.2)
#         self.dense0 = nn.Linear(32, 64)
#         self.dropout0 = nn.Dropout(p=0.7)
#         self.dense1 = nn.Linear(64, 64)
        self.dense1 = nn.Linear(32, 2)
        
#         self.dense2 = nn.Linear(64 * 4, 32) # 4 is from the concat below
        # self.dropout2 = nn.Dropout(p=0.5)
#         self.dense3 = nn.Linear(32, 1)
        self.dense3 = nn.Linear(2 * 2, 1)


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

        batch_mean = batch_weights.mm(hidden).expand(batch_size, hidden.shape[-1])
        batch_second_moment = batch_weights.mm(torch.pow(hidden, 2)).expand(batch_size, hidden.shape[-1])
        batch_std = batch_second_moment - torch.pow(batch_mean, 2)

        batch_features = torch.cat([batch_mean, batch_std], 1)
        self.batch_features = batch_features       
        outputs = self.dense3(
            # self.dropout2(
#                 F.relu(
#                   self.dense2(
                        batch_features)#))
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

        self.init_lstm()

        # output dense layer
        self.dense = nn.Linear(
            # (2 *) because of torch.cat; (* 2) because of 2 streams
            2 * (output_size * self.num_directions * 2),
            128
        )

        self.dense2 = nn.Linear(128, tagger_output_size)

    def init_lstm(self):
        for l in [self.lstm_lead, self.lstm_sublead]:
            # recurrent kernel init
            nn.init.orthogonal(l.weight_hh_l0)

            # in -> h kernel init
            nn.init.xavier_uniform(l.weight_ih_l0)


            # Initialize the forget gate to one
            for bias in [l.bias_hh_l0, l.bias_ih_l0]:
                bias.data.zero_()
                bias.data[l.hidden_size:2 * l.hidden_size] = 1.0

    def forward(self, leading_jets, subleading_jets, lengths,
                batch_weights, batch_size):

        # sort the data by length of sequence in decreasing order (because
        # PyTorch)

        ten_type = torch.cuda.LongTensor if torch.cuda.is_available() \
            else torch.LongTensor

        lengths = torch.from_numpy(lengths).type(ten_type)

        if torch.cuda.is_available():
            leading_jets.cuda()
            subleading_jets.cuda()
            lengths.cuda()

        lead_len = lengths[:, 0]
        sublead_len = lengths[:, 1]

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
        _, (h_sublead, _) = self.lstm_sublead(packed_subleading_jets)

        # Only need the last LSTM layer, and want batch first
        # Turns to (batch, directions, features)
        h_lead = h_lead[-self.num_directions:].transpose(0, 1)
        h_sublead = h_sublead[-self.num_directions:].transpose(0, 1)

        # Turns to (batch, directions * features)
        h_lead = h_lead.contiguous().view(h_lead.size(0), -1)
        h_sublead = h_sublead.contiguous().view(h_sublead.size(0), -1)

        # sort back
        h_lead = h_lead[lead_unsort]
        h_sublead = h_sublead[sublead_unsort]

        # concatenate outputs of the 2 LSTMs
        hidden = torch.cat([h_lead, h_sublead], 1)
        hidden = F.relu(hidden)

        batch_mean = batch_weights.mm(hidden).expand(
            batch_size, hidden.shape[-1])
        batch_second_moment = batch_weights.mm(
            torch.pow(hidden, 2)).expand(batch_size, hidden.shape[-1])
        batch_std = batch_second_moment - torch.pow(batch_mean, 2)

        batch_features = torch.cat([batch_mean, batch_std], 1)
        self.batch_features = batch_features
        outputs = self.dense2(F.relu(self.dense(batch_features)))
        return outputs
