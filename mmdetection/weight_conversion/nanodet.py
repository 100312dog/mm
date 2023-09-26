import torch
from collections import OrderedDict
from arg_parse import parse_args

args = parse_args()

origin_state_dict = torch.load(args.input_weight_path)['state_dict']
origin_param_names = origin_state_dict.keys()

new_state_dict = OrderedDict()

# backbone
for i, origin_param_name in enumerate([n for n in origin_param_names if n.startswith("backbone")]):
    new_param_name = origin_param_name
    if "conv1.0" in new_param_name:
        new_param_name = new_param_name.replace("conv1.0", "conv1.conv")
    if "conv1.1" in new_param_name:
        new_param_name = new_param_name.replace("conv1.1", "conv1.bn")

    if "branch1.0" in new_param_name:
        new_param_name = new_param_name.replace("branch1.0", "branch1.depthwise_conv.conv")
    if "branch1.1" in new_param_name:
        new_param_name = new_param_name.replace("branch1.1", "branch1.depthwise_conv.bn")
    if "branch1.2" in new_param_name:
        new_param_name = new_param_name.replace("branch1.2", "branch1.pointwise_conv.conv")
    if "branch1.3" in new_param_name:
        new_param_name = new_param_name.replace("branch1.3", "branch1.pointwise_conv.bn")

    if "branch2.0" in new_param_name:
        new_param_name = new_param_name.replace("branch2.0", "branch2.0.conv")
    if "branch2.1" in new_param_name:
        new_param_name = new_param_name.replace("branch2.1", "branch2.0.bn")
    if "branch2.3" in new_param_name:
        new_param_name = new_param_name.replace("branch2.3", "branch2.1.depthwise_conv.conv")
    if "branch2.4" in new_param_name:
        new_param_name = new_param_name.replace("branch2.4", "branch2.1.depthwise_conv.bn")
    if "branch2.5" in new_param_name:
        new_param_name = new_param_name.replace("branch2.5", "branch2.1.pointwise_conv.conv")
    if "branch2.6" in new_param_name:
        new_param_name = new_param_name.replace("branch2.6", "branch2.1.pointwise_conv.bn")

    if "conv5.0" in new_param_name:
        new_param_name = new_param_name.replace("conv5.0", "stage4.conv5.conv")
    if "conv5.1" in new_param_name:
        new_param_name = new_param_name.replace("conv5.1", "stage4.conv5.bn")

    new_state_dict[new_param_name] = origin_state_dict[origin_param_name]

# neck
for i, origin_param_name in enumerate([n for n in origin_param_names if n.startswith("fpn")]):
    new_param_name = origin_param_name.replace("fpn.", "neck.")

    new_param_name = new_param_name.replace("depthwise", "depthwise_conv.conv")
    new_param_name = new_param_name.replace("dwnorm", "depthwise_conv.bn")
    new_param_name = new_param_name.replace("pointwise", "pointwise_conv.conv")
    new_param_name = new_param_name.replace("pwnorm", "pointwise_conv.bn")
    if "reduce_layers" in new_param_name:
        new_param_name = new_param_name.replace("reduce_layers", "lateral_convs")
    if "top_down_blocks" in new_param_name:
        new_param_name = new_param_name.replace("top_down_blocks", "fpn_convs")
    if "bottom_up_blocks" in new_param_name:
        new_param_name = new_param_name.replace("bottom_up_blocks", "pafpn_convs")
    if "primary_conv" in new_param_name:
        new_param_name = new_param_name.replace("primary_conv.0", "primary_conv.conv")
        new_param_name = new_param_name.replace("primary_conv.1", "primary_conv.bn")
    if "cheap_operation" in new_param_name:
        new_param_name = new_param_name.replace("cheap_operation.0", "cheap_operation.conv")
        new_param_name = new_param_name.replace("cheap_operation.1", "cheap_operation.bn")
    if "shortcut" in new_param_name:
        new_param_name = new_param_name.replace("shortcut.0", "shortcut.depthwise_conv.conv")
        new_param_name = new_param_name.replace("shortcut.1", "shortcut.depthwise_conv.bn")
        new_param_name = new_param_name.replace("shortcut.2", "shortcut.pointwise_conv.conv")
        new_param_name = new_param_name.replace("shortcut.3", "shortcut.pointwise_conv.bn")
    if "downsamples" in new_param_name:
        new_param_name = new_param_name.replace("downsamples", "downsample_convs")
    if "extra_lvl_in_conv" in new_param_name:
        new_param_name = new_param_name.replace("extra_lvl_in_conv.0", "first_top_conv")
    if "extra_lvl_out_conv" in new_param_name:
        new_param_name = new_param_name.replace("extra_lvl_out_conv.0", "second_top_conv")

    new_state_dict[new_param_name] = origin_state_dict[origin_param_name]

# head
for i, origin_param_name in enumerate([n for n in origin_param_names if n.startswith("head")]):
    new_param_name = origin_param_name.replace("head.", "bbox_head.")

    if "distribution" in new_param_name:
        new_param_name = "bbox_head.integral.project"
    new_param_name = new_param_name.replace("depthwise", "depthwise_conv.conv")
    new_param_name = new_param_name.replace("dwnorm", "depthwise_conv.bn")
    new_param_name = new_param_name.replace("pointwise", "pointwise_conv.conv")
    new_param_name = new_param_name.replace("pwnorm", "pointwise_conv.bn")

    new_state_dict[new_param_name] = origin_state_dict[origin_param_name]

output_weight_path = args.output_weight_path \
    if args.output_weight_path else args.input_weight_path.replace(".pth", "_new.pth")
torch.save(new_state_dict, output_weight_path)

