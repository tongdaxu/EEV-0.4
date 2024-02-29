# EEV-0.4
Pytorch implementation of the paper "MPAI-EEV: Standardization Efforts of Artificial Intelligence based End-to-End Video Coding"

## Requirements

- Python==3.7
- PyTorch==1.11
- torchac
- ninja

## Data Preparation

### Training Dataset

- [Vimeo-90k dataset](http://toflow.csail.mit.edu/)

### Test Dataset

This method focuses on the P-frame compression. In terms of I frames, we apply CompressAI to compress them. The test datasets include: 
-  HEVC common test sequences 
- [UVG dataset](http://ultravideo.cs.tut.fi/#testsequences_x) (1080p/8bit/YUV) 
- [MCLJCV dataset](http://mcl.usc.edu/mcl-jcv-dataset/) (1080p/8bit/YUV)

Basically, the test sequences are cropped. After that, both the width and height are the multiplier of 64. Subsequently, we split them into consecutive pictures by ffmpeg. Taking UVG as example, the data process is shown as follows. 

1. Crop Videos from 1920x1080 to 1920x1024.
    ```
    ffmpeg -pix_fmt yuv420p  -s 1920x1080 -i ./videos/xxxx.yuv -vf crop=1920:1024:0:0 ./videos_crop/xxxx.yuv
    ```
2. Convert YUV files to images.
    ```
    ffmpeg -s 1920x1024 -pix_fmt yuv420p -i ./videos_crop/xxxx.yuv ./images_crop/xxxx/im%3d.png
    ```

ffmpeg -pix_fmt yuv420p  -s 416x240 -i BasketballPass_416x240_50.yuv -vf crop=384:192:0:0 BasketballPass_384x192_50.yuv
ffmpeg -pix_fmt yuv420p  -s 416x240 -i BlowingBubbles_416x240_50.yuv -vf crop=384:192:0:0 BlowingBubbles_384x192_50.yuv
ffmpeg -pix_fmt yuv420p  -s 416x240 -i BQSquare_416x240_60.yuv -vf crop=384:192:0:0 BQSquare_384x192_60.yuv
ffmpeg -pix_fmt yuv420p  -s 416x240 -i RaceHorses_416x240_30.yuv -vf crop=384:192:0:0 RaceHorses_384x192_30.yuv


ffmpeg -s 384x192 -pix_fmt yuv420p -i BasketballPass_384x192_50.yuv ../../images_crop/BasketballPass_384x192_50/im%3d.png
ffmpeg -s 384x192 -pix_fmt yuv420p -i BlowingBubbles_384x192_50.yuv ../../images_crop/BlowingBubbles_384x192_50/im%3d.png
ffmpeg -s 384x192 -pix_fmt yuv420p -i BQSquare_384x192_60.yuv ../../images_crop/BQSquare_384x192_60/im%3d.png
ffmpeg -s 384x192 -pix_fmt yuv420p -i RaceHorses_384x192_30.yuv ../../images_crop/RaceHorses_384x192_30/im%3d.png

## Evaluation
We respectively train four differnt models for PSNR metric, where $\lambda$ equals to 256, 512, 1024 and 2048. As for MS-SSIM metric, we set $\lambda$ as 8, 16, 32 and 64. Our pretrained models are provided on [Google Drive](https://drive.google.com/drive/folders/16Ga9GRgydm1IIj6TNuP2xlxWrxZlD839?usp=sharing). 

    python eval.py --eval_lambda 256 --metric mse --intra_model cheng2020_anchor --test_class ClassD --gop_size 10 --pretrain ./checkpoints/dmvc_psnr_256.model

    python eval.py --eval_lambda 8 --metric ms-ssim --intra_model cheng2020_anchor --test_class ClassD --gop_size 10 --pretrain ./checkpoints/dmvc_msssim_8.model

## Related links
 * CompressAI: https://github.com/InterDigitalInc/CompressAI
 * Benchmark: https://github.com/ZhihaoHu/PyTorchVideoCompression
 * OpenDVC: https://github.com/RenYang-home/OpenDVC
 * DCVC: https://github.com/DeepMC-DCVC/DCVC
 * M-LVC: https://github.com/JianpingLin/M-LVC_CVPR2020
 * RLVC: https://github.com/RenYang-home/RLVC
 * DMVC: https://github.com/Linkeyboard/DMVC (The project is mainly based on DMVC)

python eval.py --img_dir /NEW_EDS/JJ_Group/xutd/common_datasets/HEVC_CTC_raw/images_crop/ClassD --eval_lambda 256 --metric mse --intra_model mbt2018 --test_class ClassD --gop_size 10 --pretrain ./checkpoints/dmvc_psnr_256.model


python eval_rdo.py --img_dir /NEW_EDS/JJ_Group/xutd/common_datasets/HEVC_CTC_raw/images_crop/ClassD --eval_lambda 256 --metric mse --intra_model mbt2018_rdo --test_class ClassD --gop_size 10 --pretrain ./checkpoints/dmvc_psnr_256.model


python eval_rdo.py --img_dir /NEW_EDS/JJ_Group/xutd/common_datasets/HEVC_CTC_raw/images_crop/ClassD --eval_lambda 512 --metric mse --intra_model mbt2018 --test_class ClassD --gop_size 10 --pretrain ./checkpoints/dmvc_psnr_512.model

python eval_rdo.py --img_dir /NEW_EDS/JJ_Group/xutd/common_datasets/HEVC_CTC_raw/images_crop/ClassD --eval_lambda 256 --metric mse --intra_model mbt2018_rdo --test_class ClassD --gop_size 10 --pretrain ./checkpoints/dmvc_psnr_256.model


python eval_rdo.py --img_dir /NEW_EDS/JJ_Group/xutd/common_datasets/HEVC_CTC_raw/images_crop/ClassD --eval_lambda 2048 --metric mse --intra_model mbt2018 --test_class ClassD --gop_size 10 --pretrain ./checkpoints/dmvc_psnr_2048.model


CUDA_VISIBLE_DEVICES=1 python eval_rdo.py --img_dir /NEW_EDS/JJ_Group/xutd/common_datasets/HEVC_CTC_raw/images_crop/ClassD --eval_lambda 1024 --metric mse --intra_model bmshj2018_hyperprior_rdo --test_class ClassD --gop_size 10 --pretrain ./checkpoints/dmvc_psnr_1024.model

CUDA_VISIBLE_DEVICES=2 python eval_rdo.py --img_dir /NEW_EDS/JJ_Group/xutd/common_datasets/HEVC_CTC_raw/images_crop/ClassD --eval_lambda 512 --metric mse --intra_model bmshj2018_hyperprior_rdo --test_class ClassD --gop_size 10 --pretrain ./checkpoints/dmvc_psnr_512.model


CUDA_VISIBLE_DEVICES=2 python eval_rdo.py --img_dir /NEW_EDS/JJ_Group/xutd/common_datasets/HEVC_CTC_raw/images_crop/ClassD --eval_lambda 512 --metric mse --intra_model bmshj2018_hyperprior --test_class ClassD --gop_size 10 --pretrain ./checkpoints/dmvc_psnr_512.model


ffmpeg -pix_fmt yuv420p  -s 1920x1080 -i Kimono1_1920x1080_24.yuv -vf crop=1920:1024:0:0 ../../videos_crop/ClassB/Kimono1_1920x1024_24.yuv

ffmpeg -s 1920x1024 -pix_fmt yuv420p -i ParkScene_1920x1024_24.yuv ../../images_crop/ClassB/ParkScene/im%3d.png

BasketballDrive_1920x1024_50.yuv', 'BQTerrace_1920x1024_60.yuv', 'Cactus_1920x1024_50.yuv',
                        'Kimono_1920x1024_24.yuv', 'ParkScene_1920x1024_24.yuv


python eval_rdo.py --img_dir /NEW_EDS/JJ_Group/xutd/common_datasets/HEVC_CTC_raw/images_crop/ClassD --eval_lambda 2048 --metric mse --intra_model mbt2018 --test_class ClassD --gop_size 10 --pretrain ./checkpoints/dmvc_psnr_2048.model


CUDA_VISIBLE_DEVICES=0 python eval_rdo.py --img_dir /NEW_EDS/JJ_Group/xutd/common_datasets/HEVC_CTC_raw/images_crop/ClassB --eval_lambda 256 --metric mse --intra_model bmshj2018_hyperprior --test_class ClassB --gop_size 10 --pretrain ./checkpoints/dmvc_psnr_256.model

CUDA_VISIBLE_DEVICES=3 python eval_rdo.py --img_dir /NEW_EDS/JJ_Group/xutd/common_datasets/HEVC_CTC_raw/images_crop/ClassB --eval_lambda 2048 --metric mse --intra_model bmshj2018_hyperprior --test_class ClassB --gop_size 10 --pretrain ./checkpoints/dmvc_psnr_2048.model

## ClassB
### Original
* lambda 256
* recon_psnr:31.3466 ms_ssim:0.939801 bpp:0.043404 time:0.5798
* lambda 512
* recon_psnr:32.6366 ms_ssim:0.954450 bpp:0.066135 time:0.5709
* lambda 1024
* recon_psnr:33.8013 ms_ssim:0.964655 bpp:0.101598 time:0.5633
* lambda 2048
* recon_psnr:34.9428 ms_ssim:0.971617 bpp:0.169011 time:0.5700

### Y rdo
* lambda 256
* recon_psnr:31.5261 ms_ssim:0.941339 bpp:0.041755 time:10.1082
* lambda 512
* recon_psnr:32.7868 ms_ssim:0.955243 bpp:0.063771 time:10.2269
* lambda 1024
* recon_psnr:33.9155 ms_ssim:0.964922 bpp:0.097747 time:10.0776
* lambda 2048
* recon_psnr:35.0287 ms_ssim:0.971822 bpp:0.162778 time:23.4840
