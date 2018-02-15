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
        self.dense1 = nn.Linear(32, 32)
        
#         self.dense2 = nn.Linear(64 * 4, 32) # 4 is from the concat below
        # self.dropout2 = nn.Dropout(p=0.5)
#         self.dense3 = nn.Linear(32, 1)
        self.dense3 = nn.Linear(32 * 2, 1)


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

        # output dense layer
        self.dense = nn.Linear(
            # (2 *) because of torch.cat; (* 2) because of 2 streams
            2 * (output_size * self.num_directions * 2),
            128
        )

        self.dense2 = nn.Linear(128, tagger_output_size)

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
        outputs = self.dense2(F.relu(self.dense(batch_features)))
        return outputs


class OldDoubleLSTM(nn.Module):

    def __init__(self, input_size, output_size, num_layers, dropout, bidirectional,
                 batch_size, tagger_output_size):
        super(DoubleLSTM, self).__init__()
        
        # members
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.batch_size = batch_size

        # LSTM layers
        self.lstm_lead = nn.LSTM(input_size=input_size, hidden_size=output_size,
                                 num_layers=num_layers, batch_first=True,
                                 dropout=dropout, bidirectional=bidirectional)
        
        self.lstm_sublead = nn.LSTM(input_size=input_size, hidden_size=output_size,
                                    num_layers=num_layers, batch_first=True,
                                    dropout=dropout, bidirectional=bidirectional)
        
        # output dense layer
        self.dense = nn.Linear(
            # (2 *) because of torch.cat; (* 2) because of 2 streams
            2 * (output_size * self.num_directions * 2), 
            128)
        self.dense2 = nn.Linear(128, tagger_output_size)

    def forward(self, leading_jets, subleading_jets, lengths, batch_weights, batch_size):
        
        # sort the data by length of sequence in decreasing order (because PyTorch)
        leading_jets = leading_jets[np.argsort(lengths[:, 0])[::-1]]
        subleading_jets = subleading_jets[np.argsort(lengths[:, 1])[::-1]]
        sorted_lengths = np.sort(lengths, axis=0)[::-1]
        sortback_leading = np.argsort([np.argsort(lengths[:, 0])[::-1]])[0]
        sortback_subleading = np.argsort([np.argsort(lengths[:, 1])[::-1]])[0]

        if torch.cuda.is_available():
            leading_jets.cuda()
            subleading_jets.cuda()

        # pack sequences
        packed_leading_jets = nn.utils.rnn.pack_padded_sequence(
            leading_jets, batch_first=True, lengths=sorted_lengths[:, 0]
        )
        packed_subleading_jets = nn.utils.rnn.pack_padded_sequence(
            subleading_jets, batch_first=True, lengths=sorted_lengths[:, 1]
        )

        # LSTMs
        lstm_lead_out, _ = self.lstm_lead(
            packed_leading_jets)
        lstm_sublead_out, _= self.lstm_sublead(
            packed_subleading_jets)
               
        # unpack sequences
        lstm_lead_out, _ = pad_packed_sequence(lstm_lead_out, batch_first=True)
        lstm_sublead_out, _ = pad_packed_sequence(lstm_sublead_out, batch_first=True)

        # sort back
        lstm_lead_out = lstm_lead_out[sortback_leading]
        lstm_sublead_out = lstm_sublead_out[sortback_subleading]
        
        # return output of last timestep (not full sequence)
        if torch.cuda.is_available():
            lead_len = torch.from_numpy(lengths[:, 0] - 1).type(torch.cuda.LongTensor)
            sublead_len = torch.from_numpy(lengths[:, 1] - 1).type(torch.cuda.LongTensor)
        else:
            lead_len = torch.from_numpy(lengths[:, 0] - 1).type(torch.LongTensor)
            sublead_len = torch.from_numpy(lengths[:, 1] - 1).type(torch.LongTensor)
        idx_lead = Variable(
            lead_len, requires_grad=False).view(-1, 1).expand(lstm_lead_out.size(0), lstm_lead_out.size(2)).unsqueeze(1)
        lstm_lead_out = lstm_lead_out.gather(1, idx_lead).squeeze()
        idx_sublead = Variable(
            sublead_len, requires_grad=False).view(-1, 1).expand(lstm_sublead_out.size(0), lstm_sublead_out.size(2)).unsqueeze(1)
        lstm_sublead_out = lstm_sublead_out.gather(1, idx_sublead).squeeze()
        
        # concatenate outputs of the 2 LSTMs
        hidden = torch.cat([lstm_lead_out, lstm_sublead_out], 1)
        hidden = F.relu(hidden)

        batch_mean = batch_weights.mm(hidden).expand(batch_size, hidden.shape[-1])
        batch_second_moment = batch_weights.mm(torch.pow(hidden, 2)).expand(batch_size, hidden.shape[-1])
        batch_std = batch_second_moment - torch.pow(batch_mean, 2)

        batch_features = torch.cat([batch_mean, batch_std], 1)    
        outputs = self.dense2(F.relu(self.dense(batch_features)))
        return outputs
