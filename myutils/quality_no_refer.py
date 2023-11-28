### UCIQE: https://ieeexplore.ieee.org/abstract/document/7300447
### UIQM https://ieeexplore.ieee.org/abstract/document/7305804
import numpy as np

import math
from scipy import ndimage
from skimage import color, filters, io
import os
import cv2
from PIL import Image
import pyiqa
import torch
from tqdm import tqdm
from torchvision import transforms


## come form https://github.com/Owen718/Image-quality-measure-method/blob/main/uqim_utils.py
def _uiconm(x, window_size):
    """
      Underwater image contrast measure
      https://github.com/tkrahn108/UIQM/blob/master/src/uiconm.cpp
      https://ieeexplore.ieee.org/abstract/document/5609219
    """
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size
    # weight
    w = -1./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[0:int(blocksize_y*k2), 0:int(blocksize_x*k1)]
    # entropy scale - higher helps with randomness
    alpha = 1
    val = 0
    k1 = int(k1)
    k2 = int(k2)
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_-min_
            bot = max_+min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
            else: val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
            #try: val += plip_multiplication((top/bot),math.log(top/bot))
    return w*val

def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # sort pixels by intensity - for clipping
    x = sorted(x)
    # get number of pixels
    K = len(x)
    # calculate T alpha L and T alpha R
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    # calculate mu_alpha weight
    weight = (1/(K-T_a_L-T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s   = int(T_a_L+1)
    e   = int(K-T_a_R)
    val = sum(x[s:e])
    val = weight*val
    return val

def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel-mu), 2)
    return val/len(x)

def _uicm(x):
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
    r = math.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)

def sobel(x):
    dx = ndimage.sobel(x,0)
    dy = ndimage.sobel(x,1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag) 
    return mag

def _uism(x):
    """
      Underwater Image Sharpness Measure
    """
    # get image channels
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]
    # first apply Sobel edge detector to each RGB component
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    # multiply the edges detected for each channel by the channel itself
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    # get eme for each channel
    r_eme = eme1(R_edge_map, 10)
    g_eme = eme1(G_edge_map, 10)
    b_eme = eme1(B_edge_map, 10)
    # coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)

def eme1(x, window_size):
    """
      Enhancement measure estimation
      x.shape[0] = height
      x.shape[1] = width
    """
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size
    # weight
    w = 2./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[0:int(blocksize_y*k2), 0:int(blocksize_x*k1)]
    val = 0
    k1 = int(k1)
    k2 = int(k2)
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            max_ = np.max(block)
            min_ = np.min(block)
            # bound checks, can't do log(0)
            if min_ == 0.0: val += 0
            elif max_ == 0.0: val += 0
            else: val += math.log(max_/min_)
    return w*val

def getUIQM1(x):
    x = np.asarray(x).astype(np.float32)
    
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x, 10)
    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
    return uiqm


## come form https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement/blob/main/.ipynb_checkpoints/test_%E6%97%A0%E5%8F%82%E8%80%83-checkpoint.ipynb
def eme2(ch, blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]

            blockmin = np.float32(np.min(block))
            blockmax = np.float32(np.max(block))

            # # old version
            # if blockmin == 0.0: eme += 0
            # elif blockmax == 0.0: eme += 0
            # else: eme += w * math.log(blockmax / blockmin)

            # new version
            if blockmin == 0: blockmin+=1
            if blockmax == 0: blockmax+=1
            eme += w * math.log(blockmax / blockmin)
    return eme

def plipsum(i,j,gamma=1026):
    return i + j - i * j / gamma

def plipsub(i,j,k=1026):
    return k * (i - j) / (k - j)

def plipmult(c,j,gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c

def logamee(ch,blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]
            blockmin = np.float32(np.min(block))
            blockmax = np.float32(np.max(block))

            top = plipsub(blockmax,blockmin)
            bottom = plipsum(blockmax,blockmin)

            m = top/bottom
            if m ==0.:
                s+=0
            else:
                s += (m) * np.log(m)

    return plipmult(w,s)

def getUIQM2(a):  # nan, don't use
    rgb = np.asarray(a)
    gray = color.rgb2gray(a)

    # UIQM
    p1 = 0.0282
    p2 = 0.2953
    p3 = 3.5753

    # 1st term UICM
    rg = rgb[:,:,0] - rgb[:,:,1]
    yb = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
    rgl = np.sort(rg,axis=None)
    ybl = np.sort(yb,axis=None)
    al1 = 0.1
    al2 = 0.1
    T1 = np.int32(al1 * len(rgl))
    T2 = np.int32(al2 * len(rgl))
    rgl_tr = rgl[T1:-T2]
    ybl_tr = ybl[T1:-T2]

    urg = np.mean(rgl_tr)
    s2rg = np.mean((rgl_tr - urg) ** 2)
    uyb = np.mean(ybl_tr)
    s2yb = np.mean((ybl_tr- uyb) ** 2)

    uicm =-0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

    # 2nd term UISM (k1k2=8x8)
    Rsobel = rgb[:,:,0] * filters.sobel(rgb[:,:,0])
    Gsobel = rgb[:,:,1] * filters.sobel(rgb[:,:,1])
    Bsobel = rgb[:,:,2] * filters.sobel(rgb[:,:,2])

    Rsobel=np.round(Rsobel).astype(np.uint8)
    Gsobel=np.round(Gsobel).astype(np.uint8)
    Bsobel=np.round(Bsobel).astype(np.uint8)

    Reme = eme2(Rsobel)
    Geme = eme2(Gsobel)
    Beme = eme2(Bsobel)

    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    # 3rd term UIConM
    uiconm = logamee(gray)

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm
    
    return uiqm

def getUCIQE(img_BGR):  # very large, don't use
    #img_BGR = cv2.imread(img)
    img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB) 
    img_LAB = np.array(img_LAB,dtype=np.float64)
    # Trained coefficients are c1=0.4680, c2=0.2745, c3=0.2576 according to paper.
    coe_Metric = [0.4680, 0.2745, 0.2576]
    
    img_lum = img_LAB[:,:,0]/255.0
    img_a = img_LAB[:,:,1]/255.0
    img_b = img_LAB[:,:,2]/255.0

    # item-1
    chroma = np.sqrt(np.square(img_a)+np.square(img_b))
    sigma_c = np.std(chroma)

    # item-2
    img_lum = img_lum.flatten()
    sorted_index = np.argsort(img_lum)
    top_index = sorted_index[int(len(img_lum)*0.99)]
    bottom_index = sorted_index[int(len(img_lum)*0.01)]
    con_lum = img_lum[top_index] - img_lum[bottom_index]

    # item-3
    chroma = chroma.flatten()
    sat = np.divide(chroma, img_lum, out=np.zeros_like(chroma, dtype=np.float64), where=img_lum!=0)
    avg_sat = np.mean(sat)

    uciqe = sigma_c*coe_Metric[0] + con_lum*coe_Metric[1] + avg_sat*coe_Metric[2]
    return uciqe


## come form https://github.com/JOU-UIP/UCIQE/blob/main/UCIQE.py
def getUCIQE2(loc):
    img_bgr = cv2.imread(loc)        # Used to read image files
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)  # Transform to Lab color space

    coe_metric = [0.4680, 0.2745, 0.2576]      # Obtained coefficients are: c1=0.4680, c2=0.2745, c3=0.2576.
    img_lum = img_lab[..., 0]/255
    img_a = img_lab[..., 1]/255
    img_b = img_lab[..., 2]/255

    img_chr = np.sqrt(np.square(img_a)+np.square(img_b))              # Chroma

    img_sat = img_chr/np.sqrt(np.square(img_chr)+np.square(img_lum))  # Saturation
    aver_sat = np.mean(img_sat)                                       # Average of saturation

    aver_chr = np.mean(img_chr)                                       # Average of Chroma

    var_chr = np.sqrt(np.mean(abs(1-np.square(aver_chr/img_chr))))    # Variance of Chroma

    dtype = img_lum.dtype                                             # Determine the type of img_lum
    if dtype == 'uint8':
        nbins = 256
    else:
        nbins = 65536

    hist, bins = np.histogram(img_lum, nbins)                        # Contrast of luminance
    cdf = np.cumsum(hist)/np.sum(hist)

    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0]-1)/(nbins-1), (ihigh[0][0]-1)/(nbins-1)]
    con_lum = tol[1]-tol[0]

    uciqe = coe_metric[0]*var_chr+coe_metric[1]*con_lum + coe_metric[2]*aver_sat         # get final quality value
    
    return uciqe


def calculate_path(path):
    all_images = os.listdir(path)

    sumuiqm1, sumuciqe, sumniqe = 0., 0., 0.
    num_images = 0

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    niqe_metrics = pyiqa.create_metric('niqe', device=device)

    for item in tqdm(all_images):
        impath = os.path.join(path, item)
        
        imgRGB = Image.open(impath)
        #imgx=cv2.resize(imgx,(256,256))
        uiqm1 = getUIQM1(imgRGB)
        uciqe = getUCIQE2(impath)
        niqe_num = niqe_metrics(impath).item()
        
        sumuiqm1 = sumuiqm1 + uiqm1
        sumuciqe = sumuciqe + uciqe
        sumniqe = sumniqe + niqe_num
        num_images += 1

    muiqm = sumuiqm1 / num_images
    muciqe = sumuciqe / num_images
    mniqe = sumniqe / num_images

    return muiqm, muciqe, mniqe


def calculate_path_NRIQA(path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    musiq_metrics = pyiqa.create_metric('musiq-koniq', device=device) # input 0~1 when test
    uranker_metrics = pyiqa.create_metric('uranker', device=device) # input 0~1
    
    all_images = os.listdir(path)
    sumuranker, summusiq = 0., 0.
    num_images = 0

    for item in tqdm(all_images):
        impath = os.path.join(path, item)
        imgRGB = Image.open(impath).convert('RGB')
        transform = transforms.ToTensor()
        img_tensor = transform(imgRGB).to(device)
        musiq_num = musiq_metrics(img_tensor).item()
        uranker_num = uranker_metrics(img_tensor.unsqueeze(0)).item()

        sumuranker = sumuranker + uranker_num
        summusiq = summusiq + musiq_num

        num_images += 1

    muranker = sumuranker / num_images
    mmusiq = summusiq / num_images
    muiqm, muciqe, mniqe = calculate_path(path)
    print("UCIQE: ", muciqe)
    print("UIQM: ", muiqm)
    print("NIQE: ", mniqe)
    print("URanker: " + str(muranker))
    print("MUSIQ: " + str(mmusiq))


def calculate_All_non_reference_NRIQA(all_path):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    musiq_metrics = pyiqa.create_metric('musiq-koniq', device=device) # input 0~1 when test
    uranker_metrics = pyiqa.create_metric('uranker', device=device) # input 0~1

    for method_name in os.listdir(all_path):
        if os.path.isdir(os.path.join(all_path, method_name)):
            if "U45" in all_path:
                path = os.path.join(all_path, method_name)
            elif "UIEB" in all_path:
                path = os.path.join(all_path, method_name, "C60")
            if os.path.exists(path):
                print("Method: " + method_name)
                all_images = os.listdir(path)
                if len(all_images) == 0:
                    continue

                sumuranker, summusiq = 0., 0.
                num_images = 0

                for item in tqdm(all_images):
                    impath = os.path.join(path, item)
                    imgRGB = Image.open(impath).convert('RGB')
                    transform = transforms.ToTensor()
                    img_tensor = transform(imgRGB).to(device)
                    musiq_num = musiq_metrics(img_tensor).item()
                    uranker_num = uranker_metrics(img_tensor.unsqueeze(0)).item()

                    sumuranker = sumuranker + uranker_num
                    summusiq = summusiq + musiq_num

                    num_images += 1

                muranker = sumuranker / num_images
                mmusiq = summusiq / num_images
                muiqm, muciqe, mniqe = calculate_path(path)
                print("UCIQE: ", muciqe)
                print("UIQM: ", muiqm)
                print("NIQE: ", mniqe)
                print("URanker: " + str(muranker))
                print("MUSIQ: " + str(mmusiq))