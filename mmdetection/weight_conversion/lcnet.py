import torch
import re
from collections import OrderedDict
import paddle
from arg_parse import parse_args

args = parse_args()

paddle_state_dict = paddle.load(args.input_weight_path)
paddle_param_names = paddle_state_dict.keys()

torch_backbone_state_dict = OrderedDict()

for i, paddle_param_name in enumerate([n for n in paddle_param_names]):
    if "backbone" in paddle_param_name:
        torch_param_name = paddle_param_name.replace("backbone.", "")
    else:
        torch_param_name = paddle_param_name
    if "mean" in torch_param_name:
        torch_param_name = torch_param_name.replace("_mean", "running_mean")
    if "variance" in torch_param_name:
        torch_param_name = torch_param_name.replace("_variance", "running_var")
    if "dw_conv" in torch_param_name:
        torch_param_name = torch_param_name.replace("dw_conv", "depthwise_conv")
    if "pw_conv" in torch_param_name:
        torch_param_name = torch_param_name.replace("pw_conv", "pointwise_conv")
    if "se.conv" in torch_param_name:
        torch_param_name = re.sub(r'se\.conv\d', lambda match: match.group() + ".conv", torch_param_name)
    if "conv_t" in torch_param_name:
        torch_param_name = torch_param_name.replace("conv_t", "lateral_convs")
    if torch_param_name == "fc.weight":
        torch_backbone_state_dict[torch_param_name] = torch.from_numpy(
            paddle_state_dict[paddle_param_name].detach().cpu().numpy().T)
    else:
        torch_backbone_state_dict[torch_param_name] = torch.from_numpy(
            paddle_state_dict[paddle_param_name].detach().cpu().numpy())

output_weight_path = args.output_weight_path \
    if args.output_weight_path else args.input_weight_path.replace(".pdparams", ".pth")
torch.save(torch_backbone_state_dict, output_weight_path)