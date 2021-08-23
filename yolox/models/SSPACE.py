## search space for each TBS layer
## FBNet V2
SSPACE = {
    "MICRO": [
        ('B1', "FBNetV2Block", {'kernel_size':3, 'use_se': 0, 'groups': 1}), 
        ('B2', "FBNetV2Block", {'kernel_size':3, 'use_se': 0, 'groups': 2}),
        ('B3', "FBNetV2Block", {'kernel_size':3, 'use_se': 0, 'groups': 4}),
        ('B4', "FBNetV2Block", {'kernel_size':3, 'use_se': 1, 'groups': 1}),
        ('B5', "FBNetV2Block", {'kernel_size':3, 'use_se': 1, 'groups': 2}),
        ('B6', "FBNetV2Block", {'kernel_size':3, 'use_se': 1, 'groups': 4}),
        ('B7', "FBNetV2Block", {'kernel_size':3, 'use_se': 0, 'groups': 8}),
        ('B8', "FBNetV2Block", {'kernel_size':3, 'use_se': 0, 'groups': 16}),
        ('B9', "FBNetV2Block", {'kernel_size':3, 'use_se': 0, 'groups': 32}),
        ('B10', "FBNetV2Block", {'kernel_size':3, 'use_se': 1, 'groups': 8}),
        ('B11', "FBNetV2Block", {'kernel_size':3, 'use_se': 1, 'groups': 16}),
        ('B12', "FBNetV2Block", {'kernel_size':3, 'use_se': 1, 'groups': 32}),
        ('B13', "FBNetV2Block", {'kernel_size':5, 'use_se': 0, 'groups': 1}), 
        ('B14', "FBNetV2Block", {'kernel_size':5, 'use_se': 0, 'groups': 2}),
        ('B15', "FBNetV2Block", {'kernel_size':5, 'use_se': 0, 'groups': 4}),
        ('B16', "FBNetV2Block", {'kernel_size':5, 'use_se': 1, 'groups': 1}),
        ('B17', "FBNetV2Block", {'kernel_size':5, 'use_se': 1, 'groups': 2}),
        ('B18', "FBNetV2Block", {'kernel_size':5, 'use_se': 1, 'groups': 4}),
        ('SKIP', "Identity", {}),                                           
    ],

    # resnet34 space
    "RESNET_34_3463": [
        # stage 0, input 224 x 224 x 3
        ["conv_3x3_1", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":3, "out_channels":32, "act":"relu", "bias":False}],
        ["conv_3x3_2", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":32, "out_channels":64, "act":"relu", "bias":False}],
        # stage 1 --- 3 layers
        ["TBS1", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":64, "out_channels": [64, 96, 16]}],   
        ["TBS2", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":96, "out_channels":[64, 96, 16]}],
        ["TBS3", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":96, "out_channels":[64, 96, 16]}],
        # stage 2 --- 4 layers
        ["TBS4", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":1, "in_channels":96, "out_channels":[96, 160, 32]}],
        ["TBS5", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":160, "out_channels":[96, 160, 32]}],
        ["TBS6", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":160, "out_channels":[96, 160, 32]}],
        ["TBS7", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":160, "out_channels":[96, 160, 32]}],
        # stage 3 --- 6 layers
        ["TBS8", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":1, "in_channels":160, "out_channels":[192, 320, 64]}], 
        ["TBS9", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":320, "out_channels":[192, 320, 64]}],
        ["TBS10", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":320, "out_channels":[192, 320, 64]}],
        ["TBS11", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":320, "out_channels":[192, 320, 64]}],
        ["TBS12", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":320, "out_channels":[192, 320, 64]}],
        ["TBS13", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":320, "out_channels":[192, 320, 64]}],
        # stage 4 -- 3 layers
        ["TBS14", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":1, "in_channels":320, "out_channels":[512, 2048, 512]}],
        ["TBS15", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":2048, "out_channels":[512, 2048, 512]}],
        ["TBS16", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":2048, "out_channels":[512, 2048, 512]}],
        ["ap_7x7", "AvgPool", {"kernel_size":7, 'in_channels':2048, 'out_channels':2048}]
    ],

    "RESNET_34_2284": [
        # stage 0, input 224 x 224 x 3
        ["conv_3x3_1", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":3, "out_channels":32, "act":"relu", "bias":False}],
        ["conv_3x3_2", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":32, "out_channels":64, "act":"relu", "bias":False}],
        # stage 1 --- 2 layers
        ["TBS1", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":64, "out_channels": [64, 96, 16]}],   
        ["TBS2", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":96, "out_channels":[64, 96, 16]}],
        # stage 2 --- 2 layers 
        ["TBS3", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":1, "in_channels":96, "out_channels":[96, 128, 16]}],
        ["TBS4", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":128, "out_channels":[96, 128, 16]}],
        # stage 3 --- 8 layers
        ["TBS5", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":1, "in_channels":128, "out_channels":[128, 256, 32]}],
        ["TBS6", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        ["TBS7", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        ["TBS8", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}], 
        ["TBS9", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        ["TBS10", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        ["TBS11", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        ["TBS12", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        # stage 4 -- 4 layers
        ["TBS13", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":1, "in_channels":256, "out_channels":[512, 2048, 512]}],
        ["TBS14", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":2048, "out_channels":[512, 2048, 512]}],
        ["TBS15", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":2048, "out_channels":[512, 2048, 512]}],
        ["TBS16", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":2048, "out_channels":[512, 2048, 512]}],
        ["ap_7x7", "AvgPool", {"kernel_size":7, 'in_channels':2048, 'out_channels':2048}]
    ],
 
    # resnet50 space
    "RESNET_50_3463": [
        # stage 0, input 224 x 224 x 3
        ["conv_3x3_1", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":3, "out_channels":32, "act":"relu", "bias":False}],
        ["conv_3x3_2", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":32, "out_channels":64, "act":"relu", "bias":False}],
        # stage 1 --- 3 layers
        ["TBS1", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":64, "out_channels": [64, 128, 16]}],   
        ["TBS2", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":128, "out_channels":[64, 128, 16]}],
        ["TBS3", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":128, "out_channels":[64, 128, 16]}],
        # stage 2 --- 4 layers
        ["TBS4", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":1, "in_channels":128, "out_channels":[128, 256, 32]}],
        ["TBS5", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        ["TBS6", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        ["TBS7", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        # stage 3 --- 6 layers
        ["TBS8", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":1, "in_channels":256, "out_channels":[256, 512, 64]}], 
        ["TBS9", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS10", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS11", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS12", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS13", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 4 -- 3 layers
        ["TBS14", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":[0.25, 0.5, 0.25], "in_channels":512, "out_channels":[1024, 2048, 1024]}],
        ["TBS15", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        ["TBS16", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        # ["ap_7x7", "AvgPool", {"kernel_size":7, 'in_channels':2048, 'out_channels':2048}]
        # ['AAP', 'AdaAvgPool', {"output_size":(1, 1)}]
    ],

    "RESNET_50_nose_3463": [
        # stage 0, input 224 x 224 x 3
        ["conv_3x3_1", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":3, "out_channels":32, "act":"relu", "bias":False}], # stride = 2
        ["conv_3x3_2", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":32, "out_channels":64, "act":"relu", "bias":False}], # stride = 2
        # stage 1 --- 3 layers
        ["TBS1", "MixedOp", {'bids':[0, 1, 2, 6, 7], 'stride':1, "expansion":1, "in_channels":64, "out_channels": [64, 128, 16]}],   
        ["TBS2", "MixedOp", {'bids':[0, 1, 2, 6, 7], 'stride':1, "expansion":1, "in_channels":128, "out_channels":[64, 128, 16]}],
        ["TBS3", "MixedOp", {'bids':[0, 1, 2, 6, 7], 'stride':1, "expansion":1, "in_channels":128, "out_channels":[64, 128, 16]}],
        # stage 2 --- 4 layers
        ["TBS4", "MixedOp", {'bids':[0, 1, 2, 6, 7, 8], 'stride':2, "expansion":1, "in_channels":128, "out_channels":[128, 256, 32]}], # stride = 2
        ["TBS5", "MixedOp", {'bids':[0, 1, 2, 6, 7, 8], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        ["TBS6", "MixedOp", {'bids':[0, 1, 2, 6, 7, 8], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        ["TBS7", "MixedOp", {'bids':[0, 1, 2, 6, 7, 8], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}], 
        # stage 3 --- 6 layers
        ["TBS8", "MixedOp", {'bids':[0, 1, 2, 6, 7, 8], 'stride':2, "expansion":1, "in_channels":256, "out_channels":[256, 512, 64]}],  # stride = 2
        ["TBS9", "MixedOp", {'bids':[0, 1, 2, 6, 7, 8], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS10", "MixedOp", {'bids':[0, 1, 2, 6, 7, 8], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS11", "MixedOp", {'bids':[0, 1, 2, 6, 7, 8], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS12", "MixedOp", {'bids':[0, 1, 2, 6, 7, 8], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS13", "MixedOp", {'bids':[0, 1, 2, 6, 7, 8], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 4 -- 3 layers
        ["TBS14", "MixedOp", {'bids':[0, 1, 2, 6, 7, 8], 'stride':2, "expansion":[0.25, 0.5, 0.25], "in_channels":512, "out_channels":[1024, 2048, 1024]}], # stride = 2
        ["TBS15", "MixedOp", {'bids':[0, 1, 2, 6, 7, 8], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        ["TBS16", "MixedOp", {'bids':[0, 1, 2, 6, 7, 8], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        ["ap_7x7", "AvgPool", {"kernel_size":7, 'in_channels':2048, 'out_channels':2048}]
    ],

    "RESNET_50_op_3463": [
        # stage 0, input 224 x 224 x 3
        ["conv_3x3_1", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":3, "out_channels":32, "act":"relu", "bias":False}],
        ["conv_3x3_2", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":32, "out_channels":64, "act":"relu", "bias":False}],
        # stage 1 --- 3 layers
        ["TBS1", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17], 'stride':1, "expansion":1, "in_channels":64, "out_channels": 128}],   
        ["TBS2", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17], 'stride':1, "expansion":1, "in_channels":128, "out_channels":128}],
        ["TBS3", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17], 'stride':1, "expansion":1, "in_channels":128, "out_channels":128}],
        # stage 2 --- 4 layers
        ["TBS4", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17], 'stride':2, "expansion":1, "in_channels":128, "out_channels":192}],
        ["TBS5", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17], 'stride':1, "expansion":1, "in_channels":192, "out_channels":192}],
        ["TBS6", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17], 'stride':1, "expansion":1, "in_channels":192, "out_channels":192}],
        ["TBS7", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17], 'stride':1, "expansion":1, "in_channels":192, "out_channels":192}],
        # stage 3 --- 6 layers
        ["TBS8", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17], 'stride':2, "expansion":1, "in_channels":192, "out_channels":384}], 
        ["TBS9", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17], 'stride':1, "expansion":1, "in_channels":384, "out_channels":384}],
        ["TBS10", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17], 'stride':1, "expansion":1, "in_channels":384, "out_channels":384}],
        ["TBS11", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17], 'stride':1, "expansion":1, "in_channels":384, "out_channels":384}],
        ["TBS12", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17], 'stride':1, "expansion":1, "in_channels":384, "out_channels":384}],
        ["TBS13", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17], 'stride':1, "expansion":1, "in_channels":384, "out_channels":384}],
        # stage 4 -- 3 layers
        ["TBS14", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17], 'stride':2, "expansion":0.25, "in_channels":384, "out_channels":2048}],
        ["TBS15", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17], 'stride':1, "expansion":0.25, "in_channels":2048, "out_channels":2048}],
        ["TBS16", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17], 'stride':1, "expansion":0.25, "in_channels":2048, "out_channels":2048}],
        ["ap_7x7", "AvgPool", {"kernel_size":7, 'in_channels':2048, 'out_channels':2048}]
    ],

    "RESNET_50_2284": [
        # stage 0, input 224 x 224 x 3
        ["conv_3x3_1", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":3, "out_channels":32, "act":"relu", "bias":False}],
        ["conv_3x3_2", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":32, "out_channels":64, "act":"relu", "bias":False}],
        # stage 1 --- 2 layers
        ["TBS1", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":64, "out_channels": [64, 128, 16]}],   
        ["TBS2", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":128, "out_channels":[64, 128, 16]}],
        # stage 2 --- 2 layers 
        ["TBS3", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":1, "in_channels":128, "out_channels":[128, 256, 32]}],
        ["TBS4", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        # stage 3 --- 8 layers
        ["TBS5", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":1, "in_channels":256, "out_channels":[256, 512, 64]}],
        ["TBS6", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS7", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS8", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}], 
        ["TBS9", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS10", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS11", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS12", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 4 -- 4 layers
        ["TBS13", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":[0.25, 0.5, 0.25], "in_channels":512, "out_channels":[1024, 2048, 1024]}],
        ["TBS14", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        ["TBS15", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        ["TBS16", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        ["ap_7x7", "AvgPool", {"kernel_size":7, 'in_channels':2048, 'out_channels':2048}]
    ],

    # resnet101 space
    "RESNET_101_3338443": [
        # stage 0, input 224 x 224 x 3
        ["conv_3x3_1", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":3, "out_channels":32, "act":"relu", "bias":False}],
        ["conv_3x3_2", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":32, "out_channels":64, "act":"relu", "bias":False}],
        # stage 1 --- 3 layers
        ["TBS1", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":64, "out_channels": [64, 128, 16]}],   
        ["TBS2", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":128, "out_channels":[64, 128, 16]}],
        ["TBS3", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":128, "out_channels":[64, 128, 16]}],
        # stage 2 --- 3 layers
        ["TBS4", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":1, "in_channels":128, "out_channels":[128, 256, 32]}],
        ["TBS5", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        ["TBS6", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        # stage 3 --- 3 layers 
        ["TBS7", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":1, "in_channels":256, "out_channels":[256, 512, 64]}],
        ["TBS8", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}], 
        ["TBS9", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 4 --- 8 layers 
        ["TBS10", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS11", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS12", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS13", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS14", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS15", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS16", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS17", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 5 -- 4 layers
        ["TBS18", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS19", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS20", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS21", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 6 -- 4 layers
        ["TBS22", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS23", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS24", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS25", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 7 -- 3 layers
        ["TBS26", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":[0.25, 0.5, 0.25], "in_channels":512, "out_channels":[1024, 2048, 1024]}],
        ["TBS27", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        ["TBS28", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        ["ap_7x7", "AvgPool", {"kernel_size":7, 'in_channels':2048, 'out_channels':2048}]
    ],

    "RESNET_101_group_3338443": [
        # stage 0, input 224 x 224 x 3
        ["conv_3x3_1", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":3, "out_channels":32, "act":"relu", "bias":False}],
        ["conv_3x3_2", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":32, "out_channels":64, "act":"relu", "bias":False}],
        # stage 1 --- 3 layers
        ["TBS1", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":64, "out_channels": [64, 128, 16]}],
        ["TBS2", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":128, "out_channels":[64, 128, 16]}],
        ["TBS3", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":128, "out_channels":[64, 128, 16]}],
        # stage 2 --- 3 layers
        ["TBS4", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":1, "in_channels":128, "out_channels":[128, 256, 32]}],
        ["TBS5", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        ["TBS6", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        # stage 3 --- 3 layers 
        ["TBS7", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':2, "expansion":1, "in_channels":256, "out_channels":[256, 512, 64]}],
        ["TBS8", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS9", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 4 --- 8 layers
        ["TBS10", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS11", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS12", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS13", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS14", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS15", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS16", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS17", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 5 -- 4 layers
        ["TBS18", "MixedOp", {'bids':[2, 5, 6, 7, 9, 10], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS19", "MixedOp", {'bids':[2, 5, 6, 7, 9, 10], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS20", "MixedOp", {'bids':[2, 5, 6, 7, 9, 10], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS21", "MixedOp", {'bids':[2, 5, 6, 7, 9, 10], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 6 -- 4 layers
        ["TBS22", "MixedOp", {'bids':[6, 7, 8, 9, 10, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS23", "MixedOp", {'bids':[6, 7, 8, 9, 10, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS24", "MixedOp", {'bids':[6, 7, 8, 9, 10, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS25", "MixedOp", {'bids':[6, 7, 8, 9, 10, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 7 -- 3 layers
        ["TBS26", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":[0.25, 0.5, 0.25], "in_channels":512, "out_channels":[1024, 2048, 1024]}],
        ["TBS27", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        ["TBS28", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        ["ap_7x7", "AvgPool", {"kernel_size":7, 'in_channels':2048, 'out_channels':2048}]
    ],

   # group=2/8/32
   "RESNET_101_LG_3338443": [
        # stage 0, input 224 x 224 x 3
        ["conv_3x3_1", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":3, "out_channels":32, "act":"relu", "bias":False}],
        ["conv_3x3_2", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":32, "out_channels":64, "act":"relu", "bias":False}],
        # stage 1 --- 3 layers
        ["TBS1", "MixedOp", {'bids':[1, 4, 6, 9], 'stride':1, "expansion":1, "in_channels":64, "out_channels": [64, 128, 16]}],
        ["TBS2", "MixedOp", {'bids':[1, 4, 6, 9], 'stride':1, "expansion":1, "in_channels":128, "out_channels":[64, 128, 16]}],
        ["TBS3", "MixedOp", {'bids':[1, 4, 6, 9], 'stride':1, "expansion":1, "in_channels":128, "out_channels":[64, 128, 16]}],
        # stage 2 --- 3 layers
        ["TBS4", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':2, "expansion":1, "in_channels":128, "out_channels":[128, 256, 32]}],
        ["TBS5", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        ["TBS6", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        # stage 3 --- 3 layers 
        ["TBS7", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':2, "expansion":1, "in_channels":256, "out_channels":[256, 512, 64]}],
        ["TBS8", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}], 
        ["TBS9", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 4 --- 8 layers 
        ["TBS10", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS11", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS12", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS13", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS14", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS15", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS16", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS17", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 5 -- 4 layers
        ["TBS18", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS19", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS20", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS21", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 6 -- 4 layers
        ["TBS22", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS23", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS24", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS25", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 7 -- 3 layers
        ["TBS26", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':2, "expansion":[0.25, 0.5, 0.25], "in_channels":512, "out_channels":[1024, 2048, 1024]}],
        ["TBS27", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        ["TBS28", "MixedOp", {'bids':[1, 4, 6, 8, 9, 11], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        ["ap_7x7", "AvgPool", {"kernel_size":7, 'in_channels':2048, 'out_channels':2048}]
    ],

    "RESNET_101_group_3338844": [
        # stage 0, input 224 x 224 x 3
        ["conv_3x3_1", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":3, "out_channels":32, "act":"relu", "bias":False}],
        ["conv_3x3_2", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":32, "out_channels":64, "act":"relu", "bias":False}],
        # stage 1 --- 3 layers
        ["TBS1", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":64, "out_channels": [64, 128, 16]}],
        ["TBS2", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":128, "out_channels":[64, 128, 16]}],
        ["TBS3", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":128, "out_channels":[64, 128, 16]}],
        # stage 2 --- 3 layers
        ["TBS4", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":1, "in_channels":128, "out_channels":[128, 256, 32]}],
        ["TBS5", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        ["TBS6", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        # stage 3 --- 3 layers 
        ["TBS7", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':2, "expansion":1, "in_channels":256, "out_channels":[256, 512, 64]}],
        ["TBS8", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS9", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 4 --- 8 layers
        ["TBS10", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS11", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS12", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS13", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS14", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS15", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS16", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS17", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 5 -- 8 layers
        ["TBS18", "MixedOp", {'bids':[2, 5, 6, 7, 9, 10], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS19", "MixedOp", {'bids':[2, 5, 6, 7, 9, 10], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS20", "MixedOp", {'bids':[2, 5, 6, 7, 9, 10], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS21", "MixedOp", {'bids':[2, 5, 6, 7, 9, 10], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS22", "MixedOp", {'bids':[2, 5, 6, 7, 9, 10], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS23", "MixedOp", {'bids':[2, 5, 6, 7, 9, 10], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS24", "MixedOp", {'bids':[2, 5, 6, 7, 9, 10], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS25", "MixedOp", {'bids':[2, 5, 6, 7, 9, 10], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}], 
        # stage 6 -- 4 layers
        ["TBS26", "MixedOp", {'bids':[6, 7, 8, 9, 10, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS27", "MixedOp", {'bids':[6, 7, 8, 9, 10, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS28", "MixedOp", {'bids':[6, 7, 8, 9, 10, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS29", "MixedOp", {'bids':[6, 7, 8, 9, 10, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 7 -- 4 layers
        ["TBS30", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":[0.25, 0.5, 0.25], "in_channels":512, "out_channels":[1024, 2048, 1024]}],
        ["TBS31", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        ["TBS32", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        ["TBS33", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        ["ap_7x7", "AvgPool", {"kernel_size":7, 'in_channels':2048, 'out_channels':2048}]
    ],

    "RESNET_101_group_3338664": [
        # stage 0, input 224 x 224 x 3
        ["conv_3x3_1", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":3, "out_channels":32, "act":"relu", "bias":False}],
        ["conv_3x3_2", "ConvBlock", {"kernel_size":3, "stride":2, "padding":1, "in_channels":32, "out_channels":64, "act":"relu", "bias":False}],
        # stage 1 --- 3 layers
        ["TBS1", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":64, "out_channels": [64, 128, 16]}],
        ["TBS2", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":128, "out_channels":[64, 128, 16]}],
        ["TBS3", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":128, "out_channels":[64, 128, 16]}],
        # stage 2 --- 3 layers
        ["TBS4", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":1, "in_channels":128, "out_channels":[128, 256, 32]}],
        ["TBS5", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        ["TBS6", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":1, "in_channels":256, "out_channels":[128, 256, 32]}],
        # stage 3 --- 3 layers 
        ["TBS7", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':2, "expansion":1, "in_channels":256, "out_channels":[256, 512, 64]}],
        ["TBS8", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS9", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 4 --- 8 layers
        ["TBS10", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS11", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS12", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS13", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS14", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS15", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS16", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS17", "MixedOp", {'bids':[1, 2, 4, 5, 6, 9], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 5 -- 6 layers
        ["TBS18", "MixedOp", {'bids':[2, 5, 6, 7, 9, 10], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS19", "MixedOp", {'bids':[2, 5, 6, 7, 9, 10], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS20", "MixedOp", {'bids':[2, 5, 6, 7, 9, 10], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS21", "MixedOp", {'bids':[2, 5, 6, 7, 9, 10], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS22", "MixedOp", {'bids':[2, 5, 6, 7, 9, 10], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS23", "MixedOp", {'bids':[2, 5, 6, 7, 9, 10], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 6 -- 6 layers
        ["TBS24", "MixedOp", {'bids':[6, 7, 8, 9, 10, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS25", "MixedOp", {'bids':[6, 7, 8, 9, 10, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS26", "MixedOp", {'bids':[6, 7, 8, 9, 10, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS27", "MixedOp", {'bids':[6, 7, 8, 9, 10, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS28", "MixedOp", {'bids':[6, 7, 8, 9, 10, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        ["TBS29", "MixedOp", {'bids':[6, 7, 8, 9, 10, 11], 'stride':1, "expansion":1, "in_channels":512, "out_channels":[256, 512, 64]}],
        # stage 7 -- 4 layers
        ["TBS30", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':2, "expansion":[0.25, 0.5, 0.25], "in_channels":512, "out_channels":[1024, 2048, 1024]}],
        ["TBS31", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        ["TBS32", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        ["TBS33", "MixedOp", {'bids':[0, 1, 2, 3, 4, 5], 'stride':1, "expansion":[0.25, 0.5, 0.25], "in_channels":2048, "out_channels":[1024, 2048, 1024]}],
        ["ap_7x7", "AvgPool", {"kernel_size":7, 'in_channels':2048, 'out_channels':2048}]
    ],
 
}

