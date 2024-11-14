
# VTF-Net: A Visual Temporal Feature Network for Robust Retinal OCT Image Segmentation
### [Project page](https://github.com/IMOP-lab/VTF-Net-Pytorch) | [Our laboratory page](https://github.com/IMOP-lab)
by Xingru Huang, Zhengyao Jiang, Zhao Huang, Yihao Guo, Jian Huang, Changpeng Yue, Jin Liu, Zhiwen Zheng, Xiaoshuai Zhang

Hangzhou Dianzi University IMOP-lab
![Figure1：Detailed network structure of our proposed VTF-NET](https://github.com/IMOP-lab/VTF-Net-Pytorch/blob/main/figures/Fig2.png)
Figure1: Detailed network structure of our proposed VTF-NET


## Methods

### VTFE
![Fugure2](https://github.com/IMOP-lab/VTF-Net-Pytorch/blob/main/figures/Fig3.png)
Figure2: Visual representation of the VTFE structure. 


### MSAF
![Fugure3](https://github.com/IMOP-lab/VTF-Net-Pytorch/blob/main/figures/Fig4.png)
Figure3: Struture of MSAF 



## Installation
The hardware configuration consisted of a desktop system equipped with two NVIDIA 3080 GPUs, an Intel E5-2690V4 CPU, and 256 GB of RAM. The software environment was constituted of Python 3.9, PyTorch 2.0.0, and CUDA 11.8, with the training framework being realized through PyTorch's DistributedDataParallel (DDP) implementation.

## Experiment

### Datasets
|Datasets	| Quantity |  Training Set |	Validation Set | Testing Set|
|-|-|-|-|-|
|CMED-18k|10000|7200|800|2000|

### baseline
We provide GitHub links pointing to the PyTorch implementation code for all networks compared in this experiment here, so you can easily reproduce all these projects.

[UNet](https://github.com/milesial/Pytorch-UNet);[FCN8s](https://github.com/wkentaro/pytorch-fcn); [SegNet](https://github.com/vinceecws/SegNet_PyTorch?tab=readme-ov-file); [PSPNet](https://github.com/Lextal/pspnet-pytorch); [ENet](https://github.com/davidtvs/PyTorch-ENet); [ICNet](https://github.com/hszhao/ICNet); [UNet+AttGate](https://github.com/EdgarLefevre/Attention_Unet_Pytorch) [DANet](https://github.com/junfu1115/DANet); [LEDNet](https://github.com/sczhou/LEDNet); [DUNet](https://github.com/Tramac/awesome-semantic-segmentation-pytorch); [CENet](https://github.com/Guzaiwang/CE-Net); [CGNet](https://github.com/wutianyiRosun/CGNet); [OCNet](https://github.com/openseg-group/OCNet.pytorch); [GCN](https://github.com/tkipf/pygcn), 
### Results
![Table1](https://github.com/IMOP-lab/VTF-Net-Pytorch/blob/main/figures/Table1.jpg)
Table1: The results of segmentation performance of the proposed method against 14 baseline models, evaluated on the CMED-18K dataset. Metrics include dice coefficient, HD, HD95, NCC, and Kappa statistic. The highest performance values for each metric are highlighted in red, with the second highest marked in blue.

![Figure2](https://github.com/IMOP-lab/VTF-Net-Pytorch/blob/main/figures/Fig5.png)
Figure4: Illustration of results between VTF-Net and 14 baseline models. The first row presents the original input images, followed by corresponding results, including zoomed-in views of edema regions to highlight segmentation detail.

All experiments were executed under identical conditions, and the results are detailed in Table1 and Figure2. VTF-Net showed competitive results across various evaluation metrics.

## Abaltion study

### Key components of VTFE

![Table2](https://github.com/IMOP-lab/VTF-Net-Pytorch/blob/main/figures/Table2.jpg)
Table2: Ablation study results for the VTF-Net architecture, comparing the impact of individual modules—VTFE, MSAF, AFRP, and EFRE—on segmentation performance across multiple metrics, including Dice coefficient, HD, HD95, NCC, and Kappa. The highest performance values are highlighted in red, while the second-highest are marked in blue, demonstrating the relative contributions of each module to the overall network efficacy.

![Table3](https://github.com/IMOP-lab/VTF-Net-Pytorch/blob/main/figures/Table3.jpg)
Table3: Ablation study results for various attention fusion strategies within the MSAF module, illustrating their differential impacts on segmentation performance across multiple quantitative metrics. The CA and EMA attention mechanisms represent Coordinate Attention and Efficient Multi-Scale Attention, respectively, while CB denotes Convolutional Block Attention Module. FFT refers to the Fast Fourier Transform, LSK indicates a Large Selective Kernel Network, and CA EMA signifies the serial concatenation of CA and EMA outputs. The configuration labeled FFT + CA EMA demonstrates a parallel fusion of FFT and CA EMA outputs, and FSA represents a frequency split attention strategy. Red text highlights the highest values, whereas the second-highest scores are marked in blue, signifying performance optima across configurations.

![Table4](https://github.com/IMOP-lab/VTF-Net-Pytorch/blob/main/figures/Table4.jpg)
Table4: Detailed results conducted on the VTFE module, evaluating the impact of variations in architectural parameters number of layers, kernel sizes, and convolutional blocks on segmentation metrics. Multiple configurations were tested to determine the optimal combination of these parameters, with the highest metric values marked in red and the second-highest in blue. The configuration employing 4 layers, 5x5 kernels, and 2 convolutional blocks demonstrated the most favorable performance, indicating the importance of deeper feature hierarchies and larger receptive fields for capturing complex patterns in retinal OCT segmentation.

## Question
If you have any question, please concat 'zhengyao.jiang@hdu.edu.cn.
