import os
import torch
import sys
import warnings
sys.path.append('./options')
from AudioReader import AudioReader, write_wav, read_wav
import argparse
from Conv_TasNet import ConvTasNet
from utils import get_logger
from options.option import parse

warnings.simplefilter(action='ignore', category=FutureWarning)

class Separation():
    def __init__(self, mix_path, yaml_path, model, gpuid):
        super(Separation, self).__init__()
        self.mix = read_wav(mix_path)  # Load the mixed audio
        opt = parse(yaml_path, is_train=False)  # Load the model options
        net = ConvTasNet(**opt['net_conf'])  # Initialize the ConvTasNet model
        dicts = torch.load(model, map_location='cpu')  # Load the model state
        net.load_state_dict(dicts["model_state_dict"])  # Load the weights
        self.logger = get_logger(__name__)
        self.logger.info('Load checkpoint from {}, epoch {: d}'.format(model, dicts["epoch"]))
        self.net = net.eval()  # Set the model to evaluation mode
        self.device = torch.device('cuda:{}'.format(gpuid[0]) if len(gpuid) > 0 else 'cpu')
        self.gpuid = tuple(gpuid)

    def inference(self, file_path):
        with torch.no_grad():
            egs = self.mix
            norm = torch.norm(egs, float('inf'))
            ests = self.net(egs.to(self.device))  # Ensure audio tensor is on the right device
            spks = [torch.squeeze(s.detach().cpu()) for s in ests]  # Get the separated sources
            index = 0
            for s in spks:
                s = s[:egs.shape[0]]
                s = s * norm / torch.max(torch.abs(s))  # Normalize
                index += 1
                os.makedirs(os.path.join(file_path, f'spk{index}'), exist_ok=True)
                filename = os.path.join(file_path, f'spk{index}', 'test.wav')
                write_wav(filename, s.unsqueeze(0), 8000)  # Save the separated audio
        self.logger.info("Compute over {:d} utterances".format(len(self.mix)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mix_scp', type=str, default=r'C:\Users\samco\WPI\MQP\separation_test\sample.wav', help='Path to mix audio file.')
    parser.add_argument('-yaml', type=str, default=r'C:\Users\samco\WPI\MQP\separation_test\ConvTasNet\options\train.yml', help='Path to yaml file.')
    parser.add_argument('-model', type=str, default=r'C:\Users\samco\WPI\MQP\separation_test\ConvTasNet\best.pt', help="Path to model file.")
    parser.add_argument('-gpuid', type=str, default='0', help='Enter GPU id number')
    parser.add_argument('-save_path', type=str, default=r'C:\Users\samco\WPI\MQP\separation_test\output', help='Save result path')
    args = parser.parse_args()
    gpuid = [int(i) for i in args.gpuid.split(',')]
    separation = Separation(args.mix_scp, args.yaml, args.model, gpuid)
    separation.inference(args.save_path)

if __name__ == "__main__":
    main()
