import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.distributions.beta import Beta

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, fl=12):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.mlp_ample = nn.Sequential(nn.Linear(self.modes1, 2*self.modes1), nn.ReLU(), nn.Linear(2*self.modes1, self.modes1))
        self.update_norm = nn.LayerNorm(self.modes1)
        self.update_drop = nn.Dropout(p=0.2)
        self.ample_blocks = 0
        
    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        '''
        :param x: (batch, input_channels, seq_len)
        :return:
        # ef: (batch, input_channels, seq_len)
        # out_ft: (batch, input_channels, seq_len // 2 + 1)
        '''
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x = torch.cos(x)
        x_fft = torch.fft.rfft(x,norm='ortho')

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat) # (batch, input_channels, seq_len // 2 + 1)
        # keep low frequency feature
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_fft[:, :, :self.modes1], self.weights1)
 
        ample = out_ft[:, :, :self.modes1].abs()    # (batchsize, out_channels, modes1)
        phase = out_ft[:, :, :self.modes1].angle()  # (batchsize, out_channels, modes1)
        
        ef = torch.concat([ample, phase],-1)

        return ef, out_ft

class tf_encoder(nn.Module):
    def __init__(self, args):
        super(tf_encoder, self).__init__()
        self.modes1 = args.fourier_modes   # Number of low-frequency modes to keep
        self.width = args.input_channels
        self.length =  args.sequence_len
        self.device = args.device
        self.freq_feature = SpectralConv1d(self.width, self.width, self.modes1,self.length)  # Frequency Feature Encoder
        self.bn_freq = nn.BatchNorm1d(args.fourier_modes*2)     # It doubles because frequency features contain both amplitude and phase
        # self.cnn = CNN(args).to(args.device)                  # Use cnn Time Feature Encoder
        # self.resnet18 = RESNET18(args).to(args.device)        # Use resnet18 for Time Feature Encoder
        self.tcn = TCN(args).to(args.device)                    # Use tcn for Time Feature Encoder
        self.avg = nn.Conv1d(self.width, 1, kernel_size=3, stride=1, bias=False, padding=(3 // 2))   
        # alpha for Beta distributions
        self.alpha = args.beta_alpha

    def mixup(self, ef, et):
        """
        :param ef: (batch, feature_dim)
        :param et: (batch, feature_dim)
        :return: mixed_feature (batch, feature_dim)
        """
        batch_size = ef.size(0)
        lambda_ = Beta(self.alpha, self.alpha).sample((batch_size,)).to(self.device)
        lambda_ = lambda_.unsqueeze(-1)
        mixed_feature = lambda_ * ef + (1 - lambda_) * et
        return mixed_feature

    def forward(self, x):
        '''
        :param x: (batch, input_channels, seq_len)
        :return:
        # f: (batch, 2*seq_len)
        # out_ft: (batch, input_channels, seq_len // 2 + 1)
        '''
        assert self.length == x.shape[3]
        batch, in_d, num_nodes, seq_len = x.shape
        x = x.reshape(batch, in_d * num_nodes, seq_len)
        ef, out_ft = self.freq_feature(x)
        ef = F.relu(self.bn_freq(self.avg(ef).squeeze()))
        et = self.tcn(x)
        # print(et.shape)
        f = torch.concat([ef,et],-1)
        f = F.normalize(f)

        return f, out_ft

class tf_decoder(nn.Module):
    def __init__(self, args):
        super(tf_decoder, self).__init__()
        self.input_channels, self.sequence_len, self.final_out_channels = args.input_channels, args.sequence_len, args.final_out_channels
        # self.bn1 = nn.BatchNorm1d(self.input_channels, self.sequence_len)
        # self.bn2 = nn.BatchNorm1d(self.input_channels, self.sequence_len)
        self.bn1 = nn.BatchNorm1d(self.input_channels)
        self.bn2 = nn.BatchNorm1d(self.input_channels)
        self.layer_norm = nn.LayerNorm(self.sequence_len)
        self.relu = nn.ReLU()
        # self.convT = torch.nn.ConvTranspose1d(args.final_out_channels, self.sequence_len, self.input_channels, stride=1)
        self.convT = torch.nn.ConvTranspose1d(self.sequence_len, self.final_out_channels, kernel_size=self.input_channels, stride=1)
        self.modes = args.fourier_modes
         # input: 2 * input_channels; output: input_channels;
        self.fusion_conv = nn.Conv1d(2 * self.input_channels, self.input_channels, kernel_size=1)

        self.fusion_blocks = args.fusion_blocks

    def forward(self, f, out_ft):
        '''
        :param f: (batch, 2*seq_len)
        :param out_ft: (batch, input_channels, seq_len // 2 + 1)

        :return:
        # y: (batch, input_dim, num_nodes, seq_len)
        '''
        x_low = self.bn1(torch.fft.irfft(out_ft, n=12))   # reconstruct time series by using low frequency features
        et = f[:, self.modes*2:]
        x_high = F.relu(self.bn2(self.convT(et.unsqueeze(2)).permute(0,2,1))) # reconstruct time series by using time features for high frequency patterns. 
        y = x_low + x_high
        y = torch.unsqueeze(y, dim=1)
        return y

######## CNN ##############################################
class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.width = args.input_channels
        self.channel = args.input_channels
        self.fl =   args.sequence_len
        self.fc0 = nn.Linear(self.channel, self.width) # input channel is 2: (a(x), x)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(args.input_channels, args.mid_channels, kernel_size=8,
                      stride=1, bias=False, padding=(args.kernel_size // 2)),
            nn.BatchNorm1d(args.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(p=0.5)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(args.mid_channels, args.mid_channels , kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(args.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(args.mid_channels , args.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(args.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(args.features_len)

        
    def forward(self, x):
        '''
        :param x: (batch, input_channels, seq_len)
        :return:
        # x_flat: (batch, seq_len)
        '''
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat

######## RESNET ##############################################
class RESNET18(nn.Module):
    def __init__(self, args):
        layers = [2, 2, 2, 2]
        block = BasicBlock
        self.inplanes = args.input_channels
        self.stride = 1
        super(RESNET18, self).__init__()
        self.layer1 = self._make_layer(block, args.mid_channels, layers[0], stride=self.stride)
        self.layer2 = self._make_layer(block, args.mid_channels * 2, layers[1], stride=1)
        self.layer3 = self._make_layer(block, args.final_out_channels, layers[2], stride=1)
        self.layer4 = self._make_layer(block, args.final_out_channels, layers[3], stride=1)

        self.avgpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(args.features_len)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out = residual + out
        out = F.relu(out)
        return out

########## TCN #############################
torch.backends.cudnn.benchmark = True  # might be required to fasten TCN

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCN(nn.Module):
    def __init__(self, args):
        super(TCN, self).__init__()
        self.mid_channels = args.mid_channels
        self.final_out_channels = args.sequence_len
        # TCN features
        self.tcn_layers = [self.mid_channels, self.final_out_channels]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0
        in_channels0 = args.input_channels
        out_channels0 = self.tcn_layers[0]
        in_channels1 = self.tcn_layers[0]
        out_channels1 = self.tcn_layers[1]

        kernel_size = self.tcn_kernel_size
        stride = 1
        dilation0 = 1
        padding0 = (kernel_size - 1) * dilation0
        self.relu = nn.ReLU()
        dilation1 = 2
        padding1 = (kernel_size - 1) * dilation1
        self.downsample0 = nn.Conv1d(in_channels0, out_channels0, 1) if in_channels0 != out_channels0 else None
        self.downsample1 = nn.Conv1d(in_channels1, out_channels1, 1) if in_channels1 != out_channels1 else None
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False, padding=padding0,
                      dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),
            nn.Conv1d(out_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding0, dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(out_channels0, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),
            nn.Conv1d(out_channels1, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),
        )
    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        '''
        :param inputs: (batch, input_channels, seq_len)
        :return:
        # out: (batch, seq_len)
        '''
        x0 = self.conv_block1(inputs)   # (batch, mid_channel, seq_len)
        res0 = inputs if self.downsample0 is None else self.downsample0(inputs) # (batch, mid_channel, seq_len)
        out_0 = self.relu(x0 + res0)    # (batch, mid_channel, seq_len)

        x1 = self.conv_block2(out_0)    # (batch, input_channels, seq_len)
        res1 = out_0 if self.downsample1 is None else self.downsample1(out_0)
        out_1 = self.relu(x1 + res1)    # (batch, input_channels, seq_len)
        out = out_1[:, :, -1]
        return out

if __name__ == "__main__":
    id = 1
    print(id)