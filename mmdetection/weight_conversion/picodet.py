import torch
import re
from collections import OrderedDict
import paddle
from arg_parse import parse_args

args = parse_args()

paddle_state_dict = paddle.load(args.input_weight_path)
paddle_param_names = paddle_state_dict.keys()

torch_state_dict = OrderedDict()

# backbone
for i, paddle_param_name in enumerate([n for n in paddle_param_names if "backbone" in n]):
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
    torch_state_dict[torch_param_name] = torch.from_numpy(
        paddle_state_dict[paddle_param_name].detach().cpu().numpy())

# neck
for i, paddle_param_name in enumerate([n for n in paddle_param_names if "neck" in n]):
    torch_param_name = paddle_param_name
    if "mean" in torch_param_name:
        torch_param_name = torch_param_name.replace("_mean", "running_mean")
    if "variance" in torch_param_name:
        torch_param_name = torch_param_name.replace("_variance", "running_var")
    if "conv_t" in torch_param_name:
        torch_param_name = torch_param_name.replace("conv_t.convs", "lateral_convs")
    if "top_down_blocks" in torch_param_name:
        torch_param_name = torch_param_name.replace("top_down_blocks", "fpn_convs")
    if "bottom_up_blocks" in torch_param_name:
        torch_param_name = torch_param_name.replace("bottom_up_blocks", "pafpn_convs")
    if "dw_conv" in torch_param_name:
        torch_param_name = torch_param_name.replace("dw_conv", "depthwise_conv")
    if "pw_conv" in torch_param_name:
        torch_param_name = torch_param_name.replace("pw_conv", "pointwise_conv")
    if "downsamples" in torch_param_name or "first_top" in torch_param_name or "second_top" in torch_param_name:
        if "bn1" in torch_param_name:
            torch_param_name = torch_param_name.replace("bn1", "depthwise_conv.bn")
        if "bn2" in torch_param_name:
            torch_param_name = torch_param_name.replace("bn2", "pointwise_conv.bn")
        if "dwconv" in torch_param_name:
            torch_param_name = torch_param_name.replace("dwconv", "depthwise_conv.conv")
        if "pwconv" in torch_param_name:
            torch_param_name = torch_param_name.replace("pwconv", "pointwise_conv.conv")

        if "downsamples" in torch_param_name:
            torch_param_name = torch_param_name.replace("downsamples", "downsample_convs")
    torch_state_dict[torch_param_name] = torch.from_numpy(
        paddle_state_dict[paddle_param_name].detach().cpu().numpy())

# head
for i, paddle_param_name in enumerate([n for n in paddle_param_names if "head" in n]):
    if paddle_param_name in ["head.p3_feat.scale_reg", "head.p4_feat.scale_reg", "head.p5_feat.scale_reg",
                             "head.p6_feat.scale_reg",
                             "head.distribution_project.project"]:
        continue
    torch_param_name = paddle_param_name.replace("head.", "bbox_head.")
    if "mean" in torch_param_name:
        torch_param_name = torch_param_name.replace("_mean", "running_mean")
    if "variance" in torch_param_name:
        torch_param_name = torch_param_name.replace("_variance", "running_var")
    if ".norm" in torch_param_name:
        torch_param_name = torch_param_name.replace(".norm", ".bn")
    if "fc" in torch_param_name:
        torch_param_name = torch_param_name.replace("fc", "attention.fc")
    torch_param_name = re.sub(r'cls_conv_[d,p]w\d.\d',
                              lambda match: match.group()[-3] + "." + match.group()[:-3] + match.group()[-1],
                              torch_param_name)
    if "conv_feat" in torch_param_name:
        if "se" in torch_param_name:
            torch_param_name = torch_param_name.replace("conv_feat.se", "se")
        else:
            torch_param_name = torch_param_name.replace("conv_feat", "cls_convs")
    if "head_cls" in torch_param_name or "head_reg" in torch_param_name:
        torch_param_name = re.sub(r'head_(cls|reg)\d',
                                  lambda match: "pico_" + match.group()[5:-1] + "." + match.group()[-1],
                                  torch_param_name)
    if "cls_align" in torch_param_name:
        torch_param_name = torch_param_name.replace("cls_align", "pico_cls_align")
        if "bn1" in torch_param_name:
            torch_param_name = torch_param_name.replace("bn1", "depthwise_conv.bn")
        if "bn2" in torch_param_name:
            torch_param_name = torch_param_name.replace("bn2", "pointwise_conv.bn")
        if "dwconv" in torch_param_name:
            torch_param_name = torch_param_name.replace("dwconv", "depthwise_conv.conv")
        if "pwconv" in torch_param_name:
            torch_param_name = torch_param_name.replace("pwconv", "pointwise_conv.conv")

    torch_state_dict[torch_param_name] = torch.from_numpy(
        paddle_state_dict[paddle_param_name].detach().cpu().numpy())

torch_state_dict['bbox_head.integral.project'] = torch.linspace(0, 7, 8)

output_weight_path = args.output_weight_path \
    if args.output_weight_path else args.input_weight_path.replace(".pdparams", ".pth")
torch.save(torch_state_dict, output_weight_path)
