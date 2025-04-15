import argparse
import torch
import os


def load_tensor(file_path, file_name):
    path = os.path.join(file_path, file_name)
    tensor = torch.load(path, map_location=torch.device('cpu'))
    return tensor.squeeze(0)


def main(args):
    base_dir = './artifacts'
    env_dir = args.env
    model_dir = args.model
    tensor_dir = 'tensors'

    model_artifacts_path = os.path.join(base_dir, env_dir, model_dir)

    tensor_path = os.path.join(model_artifacts_path, tensor_dir)

    for tensor_file in os.listdir(tensor_path):
        # state_embeddings shape: [20,512]
        tensor = load_tensor(tensor_path, tensor_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main parser')
    parser.add_argument('--env', type=str, help='env folder')
    parser.add_argument('--model', type=str, help='model folder')

    main_args = parser.parse_args()
    main(main_args)