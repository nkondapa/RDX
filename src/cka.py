# inspired by
# https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/blob/master/CKA.py
# fused with https://github.com/Alxmrphi/correcting_CKA_alignment/blob/main/metrics.py

import math
import torch
import numpy as np


class CKA(object):
    def __init__(self, debiased=False):
        self.debiased = debiased

    def centering(self, K):
        if self.debiased:
            n = K.shape[0]
            np.fill_diagonal(K, 0)
            means = np.sum(K, 0, dtype=np.float64) / (n - 2)
            means -= np.sum(means) / (2 * (n - 1))
            K -= means[:, None]
            K -= means[None, :]
            np.fill_diagonal(K, 0)
            return K
        else:
            n = K.shape[0]
            unit = np.ones([n, n])
            I = np.eye(n)
            H = I - unit / n
            return np.dot(np.dot(H, K), H)

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)


class CudaCKA(object):
    def __init__(self, device, debiased=False):
        self.device = device
        self.debiased = debiased

    def centering(self, K):
        if self.debiased:
            n = K.shape[0]
            # Fill the diagonal with zeros
            K.fill_diagonal_(0)
            # Compute the means
            means = torch.sum(K, dim=0, dtype=torch.float64) / (n - 2)
            means -= torch.sum(means) / (2 * (n - 1))
            # Adjust the matrix by subtracting means
            K -= means[:, None]
            K -= means[None, :]
            # Fill the diagonal again with zeros
            K.fill_diagonal_(0)
            return K
        else:
            n = K.shape[0]
            unit = torch.ones([n, n], device=self.device)
            I = torch.eye(n, device=self.device)
            H = I - unit / n
            return torch.matmul(torch.matmul(H, K), H)

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        L_X = self.rbf(X, sigma)
        L_Y = self.rbf(Y, sigma)

        c_L_X = self.centering(L_X)
        c_L_Y = self.centering(L_Y)
        # return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))
        return {'L_X': L_X, 'L_Y': L_Y, 'centered_L_X': c_L_X, 'centered_L_Y': c_L_Y, 'HSIC': torch.sum(c_L_X * c_L_Y)}

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)

        c_L_X = self.centering(L_X)
        c_L_Y = self.centering(L_Y)
        return {'L_X': L_X, 'L_Y': L_Y, 'centered_L_X': c_L_X, 'centered_L_Y': c_L_Y, 'HSIC': torch.sum(c_L_X * c_L_Y)}

        # return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y, return_dict=False):
        xy_hsic_dict = self.linear_HSIC(X, Y)
        xx_hsic_dict = self.linear_HSIC(X, X)
        yy_hsic_dict = self.linear_HSIC(Y, Y)

        var1 = torch.sqrt(xx_hsic_dict['HSIC'])
        var2 = torch.sqrt(yy_hsic_dict['HSIC'])
        hsic = xy_hsic_dict['HSIC']
        cka = (hsic / (var1 * var2))
        out_dict = {
            'xy_hsic': xy_hsic_dict, 'xx_hsic': xx_hsic_dict, 'yy_hsic': yy_hsic_dict,
            'hsic': hsic.item(), 'var1': var1.item(), 'var2': var2.item(), 'cka': cka.item()
        }
        return out_dict if return_dict else cka

    def kernel_CKA(self, X, Y, sigma=None, return_dict=False):
        xy_hsic_dict = self.kernel_HSIC(X, Y, sigma)
        xx_hsic_dict = self.kernel_HSIC(X, X, sigma)
        yy_hsic_dict = self.kernel_HSIC(Y, Y, sigma)
        var1 = torch.sqrt(xx_hsic_dict['HSIC'])
        var2 = torch.sqrt(yy_hsic_dict['HSIC'])
        hsic = xy_hsic_dict['HSIC']
        cka = (hsic / (var1 * var2)).item()

        out_dict = {
            'xy_hsic': xy_hsic_dict, 'xx_hsic': xx_hsic_dict, 'yy_hsic': yy_hsic_dict,
            'hsic': hsic.item(), 'var1': var1.item(), 'var2': var2.item(), 'cka': cka
        }
        return out_dict if return_dict else cka

        # return hsic / (var1 * var2)
