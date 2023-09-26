import torch
from collections import OrderedDict
from arg_parse import parse_args

args = parse_args()

origin_state_dict = torch.load(args.input_weight_path)
origin_param_names = origin_state_dict.keys()

new_state_dict = OrderedDict()

# backbone
for i, origin_param_name in enumerate(origin_param_names):
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

output_weight_path = args.output_weight_path \
    if args.output_weight_path else args.input_weight_path.replace(".pth", "_new.pth")
torch.save(new_state_dict, output_weight_path)

