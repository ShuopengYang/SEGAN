import argparse
import glob
import os
from scipy.io import loadmat
import numpy as np
from generator import generatorNet
import torch

class RealESRGANer():
    """A helper class for simulation enhancement with SEGAN.
    """
    def __init__(self, model_path, model = None):
        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loadnet = torch.load(model_path, map_location = torch.device('cpu'))
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict = True)
        model.eval()
        self.model = model.to(self.device)
    
    def process(self):
        self.output = self.model(self.data)
   
    @torch.no_grad()
    def enhance(self, data):
        data = torch.from_numpy(data).float()
        data = img.unsqueeze(0) 
        self.data = data.to(self.device)
        self.process()
        return output

def main():
    """Inference demo for SEGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-iPCoarse', '--inputPC', type = str, default = '/inputsPC', help = 'Input NdarrayMatrix or folder')
    parser.add_argument('-iKRefined', '--inputKR', type = str, default = '/inputsKR', help = 'Input NdarrayMatrix or folder')
    parser.add_argument(
        type = str,
        default = 'x16',
        help = ('Model names: x16 | x64 | x256))
    parser.add_argument('-o', '--output', type = str, default = '/results', help = 'Output folder')
    args = parser.parse_args()
    args.model_name = args.model_name.split('.')[0]
    if args.model_name in ['x16', 'x64', 'x256']:
        model = RRDBNet(num_in_ch = 2, num_out_ch = 1, num_feat = 64, num_block = 6, num_grow_ch = 9)
    # determine model paths
    model_path = os.path.join('/pretrained_models', args.model_name + '.pth')
    sampler = SEGANer(
        model_path = model_path,
        model = model)
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.inputPC):
        pathsPC = [args.inputPC]
    else:
        pathsPC = sorted(glob.glob(os.path.join(args.inputPC, '*')))
    if os.path.isfile(args.inputKR):
        pathsKR = [args.inputKR]
    else:
        pathsKR = sorted(glob.glob(os.path.join(args.inputKR, '*')))

    for idxPC, pathPC in enumerate(pathsPC):
        for idxKR, pathKR in enumerate(pathsKX):
            if idxPC == idxKR:
                namePC, extensionPC = os.path.splitext(os.path.basename(pathPC))
                nameKR, extensionKR = os.path.splitext(os.path.basename(pathKR))
                print('From PC and KR Predicting Prefined', idxPC, namePC, idxKR, nameKR)
                NdarrayMatrixPC = loadmat(pathPC)
                NdarrayMatrixKR = loadmat(pathKR)
                numpyArrayPC = np.array(NdarrayMatrixPC['PP2'])
                numpyArrayKR = np.array(NdarrayMatrixKR['K'])
                dataPC = torch.from_numpy(numpyArrayPC).double()
                dataKR = torch.from_numpy(numpyArrayKR).double()
                #x4 sample
                tranPC = torch.rand(1, 200, 200).double()
                for i in range(0,50):
                    for j in range(0,50):
                        for k in range(0,4):
                            for l in range(0,4):
                                tranPC[0][4 * i + k][4 * j + l] = dataPC[i][j]
                data = torch.cat([tranPC, dataKR], dim=0)
                data = data.cpu().numpy()
                data = data.astype(np.float64)
                output = sampler.enhance(data)
                extension_csv = 'csv'
                save_path_csv = os.path.join(args.output, f'{idxPC}_{namePC}.{extension_csv}')
                output = output.squeeze(0) 
                output = output.cpu().numpy()
                output_float64 = output.astype(np.float64)
                origin_data = output_float64[0]
                np.savetxt(save_path_csv, origin_data, delimiter=',')
if __name__ == '__main__':
    main()
