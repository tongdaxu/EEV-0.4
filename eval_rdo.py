import os
import argparse
import torch
import random
import math
import datetime
import numpy as np
from models.Network import *
from torch.utils.data import DataLoader
from dataload.dataset import CTS
from pytorch_msssim import MS_SSIM
from compressai.zoo import cheng2020_anchor, cheng2020_attn, mbt2018, mbt2018_mean, bmshj2018_hyperprior
from tqdm import tqdm
from utils.info import classes_dict
from models.stream_helper import *
from models.rdo import ScaleHyperpriorSGA

metric_list = ['mse', 'ms-ssim']
parser = argparse.ArgumentParser(description='DMVC evaluation')

intra_models_zoo = [
    'cheng2020_anchor',
    'cheng2020_attn',
    'bmshj2018_hyperprior',
    'bmshj2018_hyperprior_rdo',
    'mbt2018',
    'vtm',
    'x265',
    'bpg',
]

test_class_list = [
    'ClassB',
    'ClassC',
    'ClassD',
    'ClassE',
    'ClassF',
    'UVG',
    'MCLJCV',
]

parser.add_argument('--pretrain', default = '', help='Load pretrain model')
parser.add_argument('--img_dir', default = '')
parser.add_argument('--eval_lambda', default = 256, type = int, help = '[256, 512, 1024, 2048] for MSE, [8, 16, 32, 64] for MS-SSIM')
parser.add_argument('--metric', default = 'mse', choices = metric_list, help = 'mse or ms-ssim')
parser.add_argument('--intra_model', choices = intra_models_zoo, help = 'The intra coding method')
parser.add_argument('--test_class', default = 'ClassD', type = str, choices = test_class_list, help = 'Choose from the test dataset')
parser.add_argument('--gop_size', default = '0', type = int, help = 'The length of the gop')
parser.add_argument('--write_stream', default = False, type = bool)
parser.add_argument('--bin_path', default = './bin/', type = str)
args = parser.parse_args()

if args.metric == "mse":
    if args.intra_model == 'cheng2020_anchor':
        lambda_to_qp_dict = {2048: 6, 1024: 5, 512: 4, 256: 3, 128: 2}
    elif args.intra_model == 'cheng2020_attn':
        lambda_to_qp_dict = {2048: 6, 1024: 5, 512: 4, 256: 3, 128: 2}
    elif args.intra_model == 'mbt2018' or 'bmshj2018_hyperprior' or 'bmshj2018_hyperprior_rdo':
        lambda_to_qp_dict = {2048: 6, 1024: 5, 512: 4, 256: 3, 128: 2}
    elif args.intra_model == 'mbt2018_mean':
        lambda_to_qp_dict = {2048: 6, 1024: 5, 512: 4, 256: 3, 128: 2}
    elif args.intra_model == 'vtm':
        lambda_to_qp_dict = {2048: 25, 1024: 27, 512: 31, 256: 33}
    elif args.intra_model == 'x265':
        lambda_to_qp_dict = {2048: 20, 1024: 23, 512: 26, 256: 29}
    elif args.intra_model == 'bpg':
        lambda_to_qp_dict = {2048: 24, 1024: 28, 512: 32, 256: 36}
else:
    if args.intra_model == 'cheng2020_anchor':
        lambda_to_qp_dict = {64: 6, 32: 5, 16: 4, 8: 3, 4: 2}
    elif args.intra_model == 'mbt2018' or 'bmshj2018_hyperprior' or 'bmshj2018_hyperprior_rdo':
        lambda_to_qp_dict = {64: 6, 32: 5, 16: 4, 8: 3, 4: 2}
    elif args.intra_model == 'mbt2018_mean':
        lambda_to_qp_dict = {64: 6, 32: 5, 16: 4, 8: 3, 4: 2}
    elif args.intra_model == 'vtm':
        lambda_to_qp_dict = {64: 25, 32: 27, 16: 31, 8: 33}
    elif args.intra_model == 'x265':
        lambda_to_qp_dict = {64: 20, 32: 23, 16: 26, 8: 29}
    elif args.intra_model == 'bpg':
        lambda_to_qp_dict = {64: 24, 32: 28, 16: 32, 8: 36}

if args.intra_model == 'vtm':
    images_folder = 'images_intra'
elif args.intra_model == 'x265':
    images_folder = 'h265'
elif args.intra_model == 'bpg':
    images_folder = 'bpg'
else:
    images_folder = 'images_intra'

return_intra_status = True if args.intra_model == 'vtm' or args.intra_model == 'x265' or args.intra_model == 'bpg' else False
test_dataset = CTS(args.img_dir, args.test_class, return_intra_status, args.intra_model, None, lambda_to_qp_dict[args.eval_lambda])
test_loader = DataLoader(dataset = test_dataset, shuffle = False, num_workers = 1, batch_size = 1)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def eval_model(net):
    print("Evaluating...")
    net.eval()

    sum_psnr = 0
    sum_bpp = 0
    sum_intra_bpp = 0
    sum_inter_bpp = 0
    sum_intra_psnr = 0
    sum_inter_psnr = 0
    sum_ms_ssim = 0
    t0 = datetime.datetime.now()
    cnt = 0

    if args.intra_model == 'cheng2020_anchor':
        intra_model = cheng2020_anchor(quality = lambda_to_qp_dict[args.eval_lambda], pretrained = True, metric = args.metric)
    elif args.intra_model == 'cheng2020_attn':
        intra_model = cheng2020_attn(quality = lambda_to_qp_dict[args.eval_lambda], pretrained = True, metric = args.metric)
    elif args.intra_model == 'mbt2018':
        intra_model = mbt2018(quality = lambda_to_qp_dict[args.eval_lambda], pretrained = True, metric = args.metric)
    elif args.intra_model == 'mbt2018_mean':
        intra_model = mbt2018_mean(quality = lambda_to_qp_dict[args.eval_lambda], pretrained = True, metric = args.metric)
    elif args.intra_model == 'bmshj2018_hyperprior':
        intra_model = bmshj2018_hyperprior(quality = lambda_to_qp_dict[args.eval_lambda], pretrained = True, metric = args.metric)
    elif args.intra_model == 'bmshj2018_hyperprior_rdo':
        # manually...
        from compressai.zoo import load_state_dict
        def psnr(mse):
            return 10*torch.log10((255**2) / mse)
        tot_it = 2000
        lr = 5e-3
        model_root = '/home/xutd/.cache/torch/hub/checkpoints/'
        q = lambda_to_qp_dict[args.eval_lambda]
        Ns, Ms = [128,128,128,128,128,192,192,192], [192,192,192,192,192,320,320,320]
        model_names = ["bmshj2018-hyperprior-1-7eb97409.pth.tar",
                       "bmshj2018-hyperprior-2-93677231.pth.tar",
                       "bmshj2018-hyperprior-3-6d87be32.pth.tar",
                       "bmshj2018-hyperprior-4-de1b779c.pth.tar",
                       "bmshj2018-hyperprior-5-f8b614e1.pth.tar",
                       "bmshj2018-hyperprior-6-1ab9c41e.pth.tar",
                       "bmshj2018-hyperprior-7-3804dcbd.pth.tar",
                       "bmshj2018-hyperprior-8-a583f0cf.pth.tar"]
        lams = [0.0018,0.0035,0.0067,0.0130,0.0250,0.0483,0.0932,0.1800]
        lam = args.eval_lambda / (255 * 255)
        model_path = os.path.join(model_root, model_names[q-1])
        N, M = Ns[q-1], Ms[q-1]
        intra_model = ScaleHyperpriorSGA(N, M)
        model_dict = load_state_dict(torch.load(model_path))
        intra_model.load_state_dict(model_dict)

    if not return_intra_status:
        intra_model.cuda()
        intra_model.eval()

    sum_bpp_i = 0
    sum_bpp_m = 0
    sum_bpp_r = 0

    for batch_idx, (frames, intra_bpp, gop_size, i_frames) in enumerate(test_loader):
        batch_size, frame_length, _, h, w = frames.shape
        if args.gop_size > 0:
            gop_size = args.gop_size
        else:
            gop_size = gop_size.item()

        rec_frames = []
        ms_ssim_module = MS_SSIM(data_range = 1, size_average= True, channel = 3)

        last_state = None
        bin_path = args.bin_path + args.test_class + "/" + classes_dict[args.test_class]["sequence_name"][batch_idx] + "/"
        if not os.path.exists(bin_path):
            os.makedirs(bin_path)
        for frame_idx in tqdm(range(frame_length)):
            with torch.no_grad():
                if frame_idx % gop_size == 0:
                    if frame_idx:
                        frames = frames[:, gop_size : ]
                    
                    if not return_intra_status:
                        if args.write_stream == False:
                            if 'rdo' in args.intra_model:
                                # init
                                img = frames[:, frame_idx % gop_size].cuda()
                                img_h, img_w = img.shape[2], img.shape[3]
                                img_pixnum = img_h * img_w
                                # first round
                                ret_dict = intra_model(img, "init")
                                bpp_init = torch.sum(-torch.log2(ret_dict["likelihoods"]["y"])) / (img_pixnum) +\
                                        torch.sum(-torch.log2(ret_dict["likelihoods"]["z"])) / (img_pixnum)
                                mse_init = F.mse_loss(img, ret_dict["x_hat"]) * (255 ** 2)
                                rd_init = bpp_init + lam * mse_init
                                psnr_init = psnr(mse_init)
                                with torch.enable_grad():
                                    y, z = nn.parameter.Parameter(ret_dict["y"]), nn.parameter.Parameter(ret_dict["z"])
                                    # opt = torch.optim.Adam([y], lr=lr)
                                    opt = torch.optim.Adam([y], lr=lr)
                                    for it in range(tot_it):
                                        opt.zero_grad()   
                                        ret_dict = intra_model(img, "sga", y, z, it, tot_it)
                                        bpp_y = torch.sum(-torch.log2(ret_dict["likelihoods"]["y"])) / (img_pixnum)
                                        bpp_z = torch.sum(-torch.log2(ret_dict["likelihoods"]["z"])) / (img_pixnum)
                                        bpp = bpp_y + bpp_z
                                        mse = F.mse_loss(img, ret_dict["x_hat"]) * (255 ** 2)
                                        rdcost = bpp + lam * mse
                                        rdcost.backward()
                                        opt.step()
                                        # print("y bpp: {0:.4f}, z bpp: {1:.4f}".format(bpp_y, bpp_z))

                                ret_dict = intra_model(img, "round", y, z)

                                bpp_post = torch.sum(-torch.log2(ret_dict["likelihoods"]["y"])) / (img_pixnum) +\
                                        torch.sum(-torch.log2(ret_dict["likelihoods"]["z"])) / (img_pixnum)
                                mse_post = F.mse_loss(img, ret_dict["x_hat"]) * (255 ** 2)
                                rd_post = bpp_post + lam * mse_post
                                psnr_post = psnr(mse_post)

                                print("img: {0}, psnr init: {1:.4f}, bpp init: {2:.4f}, rd init: {3:.4f}, psnr post: {4:.4f}, bpp post: {5:.4f}, rd post: {6:.4f}"\
                                    .format(frame_idx, psnr_init, bpp_init, rd_init, psnr_post, bpp_post, rd_post))

                                intra_bits = torch.sum(-torch.log2(ret_dict["likelihoods"]["y"])) +\
                                        torch.sum(-torch.log2(ret_dict["likelihoods"]["z"]))
                                x_hat = ret_dict["x_hat"]
                                intra_mse = torch.mean((x_hat - frames[:, frame_idx % gop_size].cuda()).pow(2))
                                intra_psnr = torch.mean(10 * (torch.log(1. / intra_mse) / np.log(10))).cpu().detach().numpy()
                                intra_ms_ssim = ms_ssim_module(x_hat, frames[:, frame_idx % gop_size].cuda())
                                sum_bpp += intra_bits.detach().cpu().numpy().item() / (batch_size * h * w)
                                sum_intra_bpp += intra_bits.detach().cpu().numpy().item() / (batch_size * h * w)
                                sum_bpp_i += intra_bits.detach().cpu().numpy().item() / (batch_size * h * w)

                            else:
                                result = intra_model(frames[:, frame_idx % gop_size].cuda())
                                intra_bits = sum((torch.log(likelihoods).sum() / (-math.log(2))) for likelihoods in result["likelihoods"].values())
                                x_hat = result["x_hat"].clamp(0., 1.)
                                intra_mse = torch.mean((x_hat - frames[:, frame_idx % gop_size].cuda()).pow(2))
                                intra_psnr = torch.mean(10 * (torch.log(1. / intra_mse) / np.log(10))).cpu().detach().numpy()
                                intra_ms_ssim = ms_ssim_module(x_hat, frames[:, frame_idx % gop_size].cuda())
                                sum_bpp += intra_bits.detach().cpu().numpy().item() / (batch_size * h * w)
                                sum_intra_bpp += intra_bits.detach().cpu().numpy().item() / (batch_size * h * w)
                                sum_bpp_i += intra_bits.detach().cpu().numpy().item() / (batch_size * h * w)
                        else:
                            result = intra_model.compress(frames[:, frame_idx % gop_size].cuda())
                            encode_i(result["strings"], result["shape"], bin_path + str(frame_idx) + ".bin")
                            intra_bits = torch.tensor(filesize(bin_path + str(frame_idx) + ".bin") * 8)
                            strings, shape = decode_i(bin_path + str(frame_idx) + ".bin")
                            x_hat = intra_model.decompress(strings, shape)["x_hat"].clamp(0., 1.)
                            intra_mse = torch.mean((x_hat - frames[:, frame_idx % gop_size].cuda()).pow(2))
                            intra_psnr = torch.mean(10 * (torch.log(1. / intra_mse) / np.log(10))).cpu().detach().numpy()
                            intra_ms_ssim = ms_ssim_module(x_hat, frames[:, frame_idx % gop_size].cuda())
                            sum_bpp += intra_bits.detach().cpu().numpy().item() / (batch_size * h * w)
                            sum_intra_bpp += intra_bits.detach().cpu().numpy().item() / (batch_size * h * w)
                            sum_bpp_i += intra_bits.detach().cpu().numpy().item() / (batch_size * h * w)
                    else:
                        if args.write_stream == False:
                            intra_id = frame_idx // gop_size
                            intra_mse = torch.mean((i_frames[:, intra_id].cuda() - frames[:, frame_idx % gop_size].cuda()).pow(2))
                            intra_psnr = torch.mean(10 * (torch.log(1. / intra_mse) / np.log(10))).cpu().detach().numpy()
                            intra_ms_ssim = ms_ssim_module(i_frames[:, intra_id].cuda(), frames[:, frame_idx % gop_size].cuda())
                            sum_bpp += intra_bpp.detach().numpy().item()
                            sum_intra_bpp += intra_bpp.detach().numpy().item()
                            sum_bpp_i += intra_bpp.detach().cpu().numpy().item() / (batch_size * h * w)
                        else:
                            print("The intra model does not support stream writing.")
                            return

                    sum_psnr += intra_psnr
                    sum_intra_psnr += intra_psnr
                    sum_ms_ssim += intra_ms_ssim
                    cnt += 1
                    
                    if frame_idx == 0:
                        if not return_intra_status:
                            rec_frames = x_hat.unsqueeze(1)
                        else:
                            rec_frames = i_frames[:, intra_id].unsqueeze(1).cuda()
                    else:
                        if not return_intra_status:
                            rec_frames = torch.cat([rec_frames, x_hat.unsqueeze(1)], 1)
                        else:
                            rec_frames = torch.cat([rec_frames, i_frames[:, intra_id].unsqueeze(1).cuda()], 1)

                    continue

                x_curr = frames[:, frame_idx % gop_size].cuda()

                if args.write_stream == False:
                    x_hat, last_state, recon_loss, warp_next_loss, pred_next_loss, pred_loss, bpp, bpp_y, bpp_z, bpp_h, bpp_hp = net(x_curr, rec_frames, return_state = True)
                else:
                    x_hat, last_state, recon_loss, warp_next_loss, pred_next_loss, pred_loss, bpp, bpp_y, bpp_z, bpp_h, bpp_hp = net.encode_decode(x_curr, rec_frames, return_state = True, output_path = bin_path + str(frame_idx) + ".bin")
                
                pred_psnr = 10 * (torch.log(1 * 1 / pred_loss) / np.log(10)).cpu().detach().numpy()
                pred_next_psnr = 10 * (torch.log(1 * 1 / pred_next_loss) / np.log(10)).cpu().detach().numpy()
                recon_psnr = 10 * (torch.log(1 * 1 / recon_loss) / np.log(10)).cpu().detach().numpy()
                warp_psnr = 10 * (torch.log(1 * 1 / warp_next_loss) / np.log(10)).cpu().detach().numpy()
                
                sum_bpp_m += (bpp_h + bpp_hp)
                sum_bpp_r += (bpp_y + bpp_z)

                rec_frames = torch.cat([rec_frames, x_hat.unsqueeze(1).clamp(0., 1.).detach()], 1)
                inter_ms_ssim = ms_ssim_module(x_hat, x_curr)

                if rec_frames.size(1) > 3:
                    rec_frames = rec_frames[:, -3 :]

                cnt += 1
                sum_psnr += recon_psnr
                sum_bpp += bpp.cpu().detach().numpy()

                sum_inter_bpp += bpp.cpu().detach().numpy()
                sum_inter_psnr += recon_psnr
                sum_ms_ssim += inter_ms_ssim
        
        intra_frame_length = frame_length // gop_size
        inter_frame_length = frame_length - intra_frame_length
        
        sum_intra_psnr = 0
        sum_inter_psnr = 0
        sum_intra_bpp = 0
        sum_inter_bpp = 0

    t1 = datetime.datetime.now()
    deltatime = t1 - t0
    print("recon_psnr:{:.4f} ms_ssim:{:.6f} bpp:{:.6f} time:{:.4f}".format(sum_psnr / cnt, sum_ms_ssim / cnt, sum_bpp / cnt, (deltatime.seconds + 1e-6 * deltatime.microseconds) / cnt))
    sum_tmp = sum_bpp_i + sum_bpp_m + sum_bpp_r
    
def check_dir_exist(check_dir):
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)

def main():
    print(args)

    model = DMVC()
    model.cuda()

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('The total number of the learnable parameters:', num_params)

    if args.pretrain != '':
        print('Load the model from {}'.format(args.pretrain))
        pretrained_dict = torch.load(args.pretrain)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    eval_model(model)

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    main()
