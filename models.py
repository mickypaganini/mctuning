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
        self.dropout = nn.Dropout(p=0.5)
#         self.dense0 = nn.Linear(32, 64)
#         self.dropout0 = nn.Dropout(p=0.7)
#         self.dense1 = nn.Linear(64, 64)
        self.dense1 = nn.Linear(32, 32)
        
#         self.dense2 = nn.Linear(64 * 4, 32) # 4 is from the concat below
#         self.dropout2 = nn.Dropout(p=0.7)
#         self.dense3 = nn.Linear(32, 1)
        self.dense3 = nn.Linear(32 * 2, 1)


    def forward(self, inputs, batch_weights, batch_size):

        hidden = F.relu(
            self.dense1(
#                 self.dropout0(
#                     F.relu(
#                         self.dense0(
                            self.dropout(
                                F.relu(
                                    self.dense(
                                        inputs)))))#)))

        batch_features = batch_weights.mm(hidden).expand(batch_size, hidden.shape[-1])
        
        weighted_mult = batch_weights.transpose(0, 1).expand(-1, hidden.shape[-1]) * hidden
        std = torch.std(weighted_mult, 0).expand(batch_size, -1)

        batch_features = torch.cat([batch_features, std], 1)         
        outputs = self.dense3(
#             self.dropout2(
#                 F.relu(
#                   self.dense2(
                        batch_features)#)))
        return outputs
        

class DoubleLSTM(nn.Module):

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
            tagger_output_size)
        


    def init_hidden(self, batch_size):
        '''
        Initialize LSTM hidden states to zero at the beginning of each new sequence
        '''
        if torch.cuda.is_available():
            return (Variable(torch.zeros(
                    self.num_layers * self.num_directions,
                    batch_size, 
                    self.output_size).cuda(), requires_grad=False),
                Variable(torch.zeros(
                    self.num_layers * self.num_directions,
                    batch_size,
                    self.output_size).cuda(), requires_grad=False)
               )
        else:
            return (Variable(torch.zeros(
                    self.num_layers * self.num_directions,
                    batch_size, 
                    self.output_size), requires_grad=False),
                Variable(torch.zeros(
                    self.num_layers * self.num_directions,
                    batch_size,
                    self.output_size), requires_grad=False)
               )

    def forward(self, leading_jets, subleading_jets, lengths, batch_weights, batch_size):
                
        # initialize LSTM hidden states to erase history of previous sequence
        hidden_lead = self.init_hidden(batch_size)
        hidden_sublead = self.init_hidden(batch_size)
        
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
            packed_leading_jets, hidden_lead)
        lstm_sublead_out, _= self.lstm_sublead(
            packed_subleading_jets, hidden_sublead)
               
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
        merge = torch.cat([lstm_lead_out, lstm_sublead_out], 1)

        batch_features = batch_weights.mm(merge).expand(batch_size, merge.shape[-1])
        
        weighted_mult = batch_weights.transpose(0, 1).expand(-1, merge.shape[-1]) * merge
        std = torch.std(weighted_mult, 0).expand(batch_size, -1)
        
        batch_features = torch.cat([batch_features, std], 1) 
        outputs = self.dense(batch_features)
        return outputs
