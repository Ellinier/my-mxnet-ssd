import find_mxnet
import mxnet as mx
import importlib
import argparse
import sys

parser = argparse.ArgumentParser(description='network visualization')
# parser.add_argument('--network', type=str, default='vgg16_reduced',
#                     choices = ['vgg16_reduced'],
#                     help = 'the cnn to use')
parser.add_argument('--network', type=str, default='vgg16_reduced', help = 'the cnn to use')
parser.add_argument('--num-classes', type=int, default=20,
                    help='the number of classes')
parser.add_argument('--data-shape', type=int, default=300,
                    help='set image\'s shape')
parser.add_argument('--train', action='store_true', default=False, help='show train net')
args = parser.parse_args()

sys.path.append('../model')

# train net
# net = mx.symbol.load('ssd_300-symbol.json')
net = mx.symbol.load('../model/' + args.network)



cls_preds = net.get_internals()["multibox_cls_pred_output"]
loc_preds = net.get_internals()["multibox_loc_pred_output"]
anchor_boxes = net.get_internals()["multibox_anchors_output"]

cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
    name='cls_prob')
# group output
out = mx.symbol.Group([loc_preds, cls_preds, anchor_boxes])
out = mx.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    variances=(0.1, 0.1, 0.2, 0.2))

a = mx.viz.plot_network(net, shape={"data":(1,3,args.data_shape,args.data_shape)}, \
    node_attrs={"shape":'rect', "fixedsize":'false'})

a.render("ssd_" + args.network)

# b = mx.viz.print_summary(net)