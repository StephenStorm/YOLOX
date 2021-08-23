import torch
import numpy as np
from torch import nn
from copy import deepcopy
# from collections import OrderedDict, namedtuple
# import logging
import numpy as np
from .SSPACE import SSPACE
from .micro import FBNetV2Block, Identity, AvgPool, MaxPool, ConvBlock, AdaAvgPool
import netron

theta_file_1 = '/opt/tiger/minist/YOLOX/res_fbnet_config/fbnet_029epoch.txt'
search_micro_1='RESNET_50_3463'


def get_bids(theta_file = theta_file_1, search_micro = search_micro_1):
    bids = []  
    layer = 0
    repeat_layers = []
    last_channel = 0
    if 'RESNET_50' in search_micro:
        repeat_layers = [1, 3, 5, 8]
        repeat_num = [(int(l) - 1) for l in list(search_micro[-4:])] 
    elif 'RESNET_101' in search_micro:
        repeat_layers = [1, 3, 5, 7, 9, 11, 14]
        repeat_num = [(int(l) - 1) for l in list(search_micro[-7:])]


    thetas = []
    with open(theta_file, 'r') as f:            
        assert f, "error open the model theta file"
        for line in f.readlines():
            line = [float(x) for x in line.strip().split(' ')]
            thetas.append(np.array(line))

    for theta in thetas:
        bid = np.argmax(theta)
        bids.append(bid)
        if layer in repeat_layers:
            idx = repeat_layers.index(layer)
            if layer == repeat_layers[-1]:
                # search op/channel/expansion
                if last_channel == 1024:
                    bids[-2], bids[-1] = 0, 1
                elif last_channel == 2048:
                    bids[-2], bids[-1] = 1, 0
                bids.extend(bids[-3:] * repeat_num[idx])
            else:
                # search op/channel
                bids.extend(bids[-2:] * repeat_num[idx])
        layer += 1
    # print("sampling network bids is {}".format(bids))
    return bids; 


class MixedOp(nn.Module):
    def __init__(self, bids, **macro_args):
        super(MixedOp, self).__init__()
        name, block, micro_args = SSPACE['MICRO'][bids]
        self.ops = globals()[block](**macro_args, **micro_args)

    def forward(self, x):
        return self.ops(x)


class FBNetV2(nn.Module):
    def __init__(self,
                 bids,
                #  num_classes,
                 micro = 'RESNET_50_3463',
                #  dim_feature,
                 ):
        '''
        bids = [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1]
        num_classes = 1000
        micro = 'RESNET_50_3463'
        dim_features = ?
        '''
        super(FBNetV2, self).__init__()
        # self.num_classes = num_classes
        layers, self.names = [], []
        TBS_id, l = 0, 0
        next_in_channel = 64
        if 'RESNET_50' in micro:
            llist = [int(l) for l in list(micro[-4:])]
            # llist = [3, 4, 6, 3]
        elif 'RESNET_101' in micro:
            llist = [int(l) for l in list(micro[-7:])]
        else:
            raise ValueError('unsupported micro structure')
        last_search_layer = sum(llist[:-1]) + 2 # 15
        
        sspace = deepcopy(SSPACE[micro])
        for name, layer, params in sspace: 
            #sspace[i]: ["TBS1", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":64, "out_channels": [64, 128, 16]}],
            if layer == 'MixedOp':
                bid_index = bids[TBS_id]
                bid = params['bids'][bid_index]
                del params['bids']
                TBS_id += 1
                params['out_channels'] = params['out_channels'][0] + bids[TBS_id] * params['out_channels'][2]
                TBS_id += 1
                params['in_channels'] = next_in_channel
                if l >= last_search_layer:
                    params['expansion'] = params['expansion'][0] + bids[TBS_id] * params['expansion'][2]
                    TBS_id += 1
                layers.append(globals()[layer](bid, **params))
                next_in_channel = params['out_channels']
            else:
                layers.append(globals()[layer](**params))
            self.names.append(name)
            l += 1
        # add fc in the last layer


        # dim_feature = next_in_channel
        # layers.append(nn.Linear(dim_feature, num_classes))
        # self.names.append('fc%d' % num_classes)
        self.layers = nn.ModuleList(layers)
        # stephen add
        # for layer in self.layers:
            # print(dir(layer))
        # for i in range(len(self.names)):
        #     print(self.names[i], self.layers[i])
        print(self.names) # 20

    def forward(self, x):
        outputs_name = ['TBS7','TBS13','TBS16']
        outputs = []
        N = x.size()[0]

        for idx, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = x.view(N, -1)            
            x = layer(x)
            if self.names[idx] in outputs_name:
                outputs.append(x)
            # print(self.names[idx], x.shape)
        # assert(x.shape[-1] == self.num_classes) 
        return outputs


# if __name__ == '__main__' :
#     num_classes = 1000
#     bids = get_bids()
#     model = FBNetV2(bids, num_classes, search_micro_1)
#     # print(model)
#     # torch.save(model.state_dict(), '/opt/tiger/minist/tmp/res_fbnet.pth')
#     x = torch.zeros((16, 3, 640, 640))
#     y = model(x)
#     for xx in y:
#         print(xx.shape)
    # onnx_path = "/opt/tiger/minist/tmp/res_fbnet.onnx"
    # torch.onnx.export(model, x, onnx_path)

    # print('ok'.center(50, '*'))
    # netron.start(onnx_path)
