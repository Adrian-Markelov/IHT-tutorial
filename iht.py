import numpy as np
import scipy
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import pdb


def main():
    #  Deblurring
    # y = blurred
    # A = kO

    # Open the image form working directory
    blurred = Image.open('tiger_true.jpg.jpg')



    '''
    DWT_decomposition_level = 7

    # setup linear operator
    A_blur = @(x) blur(x, k0)
    A_blur_adj = @(x) blur_adj(x, k0)
    x_init = imsharpen(blurred)
    [~, cbook] = wavedec2(x_init, DWT_decomposition_level, 'db4')

    Psi = @(x) wavedec2(x, DWT_decomposition_level, 'db4')
    Psi_adj = @(x) waverec2(x, cbook, 'db4')

    kfrac = .1
    K = floor(kfrac * numel(blurred))
    deblurred = IHT(blurred, K, Psi, Psi_adj, A_blur, A_blur_adj, 300, x_init)
    print(deblurred)
    '''


def IHT(y, K, Psi, PsiAdj, A, AAdj, MaxIter, x_init):
    eta = .5
    xstar = x_init
    s = Psi(x_init)
    for i in range(MaxIter):
        if i == 8:
            eta = .08
        g = AAdj((y - A(xstar)));
        xstar = xstar + eta * g;
        s = Psi(xstar);
        [sMaxs, idxs] = maxk(s, K);
        s[~idxs] = 0;
        xstar = PsiAdj(s);
    return xstar




def IHT_demos(y, K, Psi, PsiAdj, A, AAdj, MaxIter, x_init):
    eta = .1
    xstar = x_init
    for i in range(MaxIter):
        if i == 8:
            eta = .01
        g = AAdj((y - A(xstar)))
        xstar = xstar + eta * g

        sR = Psi(xstar(:,:, 1))
        sG = Psi(xstar(:,:, 2))
        sB = Psi(xstar(:,:, 3))

        [sMaxs, idxsR] = maxk(sR, K)
        [sMaxs, idxsG] = maxk(sG, K)
        [sMaxs, idxsB] = maxk(sB, K)
        idxs_max = max([idxsR; idxsG; idxsB])
        sR(~idxs_max) = 0
        sG(~idxs_max) = 0
        sB(~idxs_max) = 0

        xstarR = PsiAdj(sR)
        xstarG = PsiAdj(sG)
        xstarB = PsiAdj(sB)
        xstar = cat(3, xstarR, xstarG, xstarB)
    return xstar







# LINEAR FUNCTIONS and their ADJOINTS

def blur(I, k0):
    b = ndimage.convolve(I, k0, mode='constant', cval=0.0)
    return b

def blur_adj(I, k0):
    bT = ndimage.convolve(I, k0[::-1, ::-1], mode='constant', cval=0.0)
    return bT

def corrupt(x, mask):
    currupted = x* mask
    return currupted

def corrupt_adj(x, mask):
    currupted_adj = x * mask
    return currupted_adj

def mosaik_f(I_c, rm, gm, bm):
    mos = np.zeros((I_c.shape[0], I_c.shape[1]))
    R = I_c[:,:, 1]
    G = I_c[:,:, 2]
    B = I_c[:,:, 3]
    mos[rm == 1] = R[rm == 1] # red
    mos[bm == 1] = G[bm == 1] # blue
    mos[gm == 1] = B[gm == 1] # green
    return mos

def mosaik_f_adj(M, rm, gm, bm):
    I_c = np.zeros((M.shape[0], M.shape[1], 3))
    R = I_c[:,:, 1]
    G = I_c[:,:, 2]
    B = I_c[:,:, 3]
    R[rm == 1] = M[rm == 1]
    G[gm == 1] = M[gm == 1]
    B[bm == 1] = M[bm == 1]

    I_c[:,:, 1] = R # red
    I_c[:,:, 2] = G # green
    I_c[:,:, 3] = B # blue

    return I_c