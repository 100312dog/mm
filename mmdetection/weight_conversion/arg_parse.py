import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='weight conversion')
    parser.add_argument('input_weight_path', type=str, help='input weight path')
    parser.add_argument('-o', '--output_weight_path', type=str, help='output weight path', required=False)
    args = parser.parse_args()
    return args

# args = parse_args()
# print(args.input_weight_path)
# print(args.output_weight_path)