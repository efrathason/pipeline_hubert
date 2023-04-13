from transformers import HubertModel
import torch.nn as nn
import torch
from torch.autograd import Variable 

class LSTMEncoder(nn.Module):
    def __init__(self, num_classes, freeze_updates=1000, lstm_layers=2):
        super(LSTMEncoder, self).__init__()

        # We need an encoder to get representations
        #self.encoder = HubertModel.from_pretrained('ntu-spml/distilhubert')
        self.encoder = HubertModel.from_pretrained('omarxadel/hubert-large-arabic-egyptian')

        # Freeze the convolutional layers at the input. This is standard
        # practice.
        self.encoder.feature_extractor._freeze_parameters()

        #if i want to freeze all the paprams:
        #for param in self.encoder.parameters():
        #    param.requires_grad = False

        #lstm layer:
        self.input_size = self._get_output_dim()
        self.hidden_size = 2000
        self.num_layers = lstm_layers
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,num_layers=self.num_layers, batch_first=True)


        # We need to map these representations to the number of output classes,
        # i.e., the number of bpe units, or characters we will use to model the text
        self.linear = nn.Linear(self.hidden_size, num_classes)
        #print("num class:")
        #print(num_classes)
        #self.linear = nn.Linear(self._get_output_dim(), num_classes)

        # The number of updates to freeze the encoder
        self.freeze_updates = freeze_updates

    def _get_output_dim(self):
        '''
            Get the output size of an encoder. We just pass a dummy tensor
            through the encoder to see what the output size it. The min length
            is 400 since we downsample by 320x.
        '''
        x = torch.rand(1, 400)
        return self.encoder(x).last_hidden_state.size(-1)
    
    # Write the forward function on your own
    def forward(self, mb, iter_num=0):
        '''
            You will take the minibatch which is a dictionary
            mb = {
                'input': torch.FloatTensor() B x T x D (batchsize x length x dim)
                'target': torch.LongTensor() B x T (batchsize x length)
                'input_lens':
            }
            and you will get the output scores for each class. Remember that we
            also need the output lengths.
        '''
        # Pass the minibatch through the encoder (For the students in the lab)
        # You may need to figure out the syntax
        scope = torch.no_grad() if iter_num < self.freeze_updates else torch.enable_grad()
        with scope:
            x = self.encoder(mb['input'], return_dict=False)

        # Pass the encoder outputs to the linear layer (For the students)
        h_0 = Variable(torch.zeros(self.num_layers, x[0].size(0), self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(self.num_layers, x[0].size(0), self.hidden_size).cuda())
        output, (final_hidden_state, final_cell_state) = self.lstm(x[0], (h_0, c_0))
        y = self.linear(output)
        #y = self.linear(output[-1])
        #y = self.linear(x[0])
        # These values are hardcoded for now. They are the widths and strides of
        # the convolutional layers in the "feature extractor". These are the
        # convolutional layers discussed that are responsible for downsampling
        # the audio. I wrote this part for you because it's pretty annoying.
        input_lens = mb['lens']['num_samples']
        for width, stride in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
            # This is the formula for the length of the next layer
            input_lens = torch.floor((input_lens - width) / stride + 1).int()
        return y, input_lens