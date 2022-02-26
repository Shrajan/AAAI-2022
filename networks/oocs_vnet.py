import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from networks.vnet import (passthrough, ELUCons, LUConv, _make_nConv, InputTransition, DownTransition,
                        UpTransition, OutputTransition)
from networks.OOCS_3dKernels import On_Off_Center_filters_3D

class OOCS_block(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(OOCS_block, self).__init__()
        outChans = inChans
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm3d(outChans)
        self.conv2 = nn.Conv3d(inChans, outChans, kernel_size=5, padding=2)
        self.bn2 = torch.nn.BatchNorm3d(outChans)

        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        self.relu3 = ELUCons(elu, outChans*2)

        self.ops1 = _make_nConv(outChans, nConvs, elu)
        self.ops2 = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, sm_on, sm_off):
        down1 = self.relu1(self.bn1(self.conv1(x)))
        down2 = self.relu2(self.bn2(self.conv2(x)))

        out1 = sm_on + down1
        out2 = sm_off + down2

        out1 = self.ops1(out1)
        out2 = self.ops2(out2)

        out = torch.cat((out1, out2), 1)
        down = torch.cat((down1, down2), 1)
        out = self.relu3(torch.add(out, down))
        return out

##############################################################
#--------------------3D OOCS VNET k=5------------------------#
##############################################################

class OOCS_VNet_k5(nn.Module):
    def __init__(self, opt):
        super(OOCS_VNet_k5, self).__init__()
        self.elu = False
        self.output_channels = opt.output_channels
        self.n_kernels = 16
        self.dropout_rate = opt.dropout_rate
        self.in_channels = opt.input_channels

        device_id = 'cuda:' + str(opt.device_id)
        self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')

        self.conv_On_filters = On_Off_Center_filters_3D(radius=2.0, gamma=2. / 3., in_channels=self.n_kernels,
                                                     out_channels=self.n_kernels, off=False).to(self.device)
        self.conv_Off_filters = On_Off_Center_filters_3D(radius=2.0, gamma=2. / 3., in_channels=self.n_kernels,
                                                      out_channels=self.n_kernels, off=True).to(self.device)

        self.in_tr = InputTransition(self.in_channels, elu=self.elu)
        self.down_conv = nn.Conv3d(self.n_kernels , self.n_kernels , kernel_size=2, stride=2)
        self.down_tr32 = OOCS_block(16, 1, self.elu)
        self.down_tr64 = DownTransition(32, 2, self.elu)
        self.down_tr128 = DownTransition(64, 3, self.elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, self.elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, self.elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, self.elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, self.elu)
        self.up_tr32 = UpTransition(64, 32, 1, self.elu)
        self.out_tr = OutputTransition(32, self.output_channels, self.elu)

        self.weight_init()

    def forward(self, x):

        out16 = self.in_tr(x)
        pool16 = self.down_conv(out16)

        # On and Off surround modulation
        sm_on = self.surround_modulation_DoG_on(pool16) + pool16
        sm_off = self.surround_modulation_DoG_off(pool16) + pool16

        out32 = self.down_tr32(x=pool16, sm_off=sm_off, sm_on=sm_on)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out

    def weight_init(self):
        self.param_count_G = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear)):
                init.orthogonal_(module.weight)
                self.param_count_G += sum([p.data.nelement() for p in module.parameters()])
        print("{} params initialized for model.".format(self.param_count_G))

    def surround_modulation_DoG_on(self, input):
        output = F.conv3d(input, weight=self.conv_On_filters, stride=1, padding=2)
        return F.relu(output, inplace=True)

    def surround_modulation_DoG_off(self, input):
        output = F.conv3d(input, weight=self.conv_Off_filters, stride=1, padding=2)
        return F.relu(output, inplace=True)

##############################################################
#--------------------3D OOCS VNET k=3------------------------#
##############################################################

class OOCS_VNet_k3(nn.Module):
    def __init__(self, opt):
        super(OOCS_VNet_k3, self).__init__()
        self.elu = False
        self.output_channels = opt.output_channels
        self.n_kernels = 16
        self.dropout_rate = opt.dropout_rate
        self.in_channels = opt.input_channels

        device_id = 'cuda:' + str(opt.device_id)
        self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')

        self.conv_On_filters = On_Off_Center_filters_3D(radius=1.0, gamma=1. / 2., in_channels=self.n_kernels,
                                                     out_channels=self.n_kernels, off=False).to(self.device)
        self.conv_Off_filters = On_Off_Center_filters_3D(radius=1.0, gamma=1. / 2., in_channels=self.n_kernels,
                                                      out_channels=self.n_kernels, off=True).to(self.device)

        self.in_tr = InputTransition(self.in_channels, elu=self.elu)
        self.down_conv = nn.Conv3d(self.n_kernels , self.n_kernels , kernel_size=2, stride=2)
        self.down_tr32 = OOCS_block(16, 1, self.elu)
        self.down_tr64 = DownTransition(32, 2, self.elu)
        self.down_tr128 = DownTransition(64, 3, self.elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, self.elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, self.elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, self.elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, self.elu)
        self.up_tr32 = UpTransition(64, 32, 1, self.elu)
        self.out_tr = OutputTransition(32, self.output_channels, self.elu)

        self.weight_init()

    def forward(self, x):

        out16 = self.in_tr(x)
        pool16 = self.down_conv(out16)

        # On and Off surround modulation
        sm_on = self.surround_modulation_DoG_on(pool16) + pool16
        sm_off = self.surround_modulation_DoG_off(pool16) + pool16

        out32 = self.down_tr32(x=pool16, sm_off=sm_off, sm_on=sm_on)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out

    def weight_init(self):
        self.param_count_G = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear)):
                init.orthogonal_(module.weight)
                self.param_count_G += sum([p.data.nelement() for p in module.parameters()])
        print("{} params initialized for model.".format(self.param_count_G))

    def surround_modulation_DoG_on(self, input):
        output = F.conv3d(input, weight=self.conv_On_filters, stride=1, padding=1)
        return F.relu(output, inplace=True)

    def surround_modulation_DoG_off(self, input):
        output = F.conv3d(input, weight=self.conv_Off_filters, stride=1, padding=1)
        return F.relu(output, inplace=True)