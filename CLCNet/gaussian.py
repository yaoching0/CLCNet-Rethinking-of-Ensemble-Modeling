from matplotlib import pyplot as mp
import numpy as np
import torch

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

'''

Output a matrix, in which each row takes [0,1] as the x-axis, divide it into out_dim parts, 
and generate the out_dim dimension vector matrix of the discrete Gaussian distribution probability density function.
There are in_dim rows in total. The Gaussian function has the mean of (i_th row)/in_dim respectively

'''

#[[out_dim],[out_dim],[out_dim],...]
def generate_V_matrix(in_dim,out_dim):
    #out_put:(in_dim,out_dim)
    x_values = np.linspace(0, 1, out_dim)
    out_martix = []
    for i in range(in_dim):
        y_values =  gaussian(x_values, i/in_dim, sig=0.01)
        out_martix.append(y_values)
    out_martix=np.array(out_martix)
    return torch.tensor(out_martix,requires_grad=False,dtype=torch.float32).cuda()

def cal_self_att_V(gaussian_matrix,input):
    #input:(batch,n,input_size=1)
    #output:(batch,n,out_dim)
    out=gaussian_matrix*input
    return out

if __name__ == '__main__':
    x_values = np.linspace(0, 1, 1000)
    out_martix=generate_V_matrix(in_dim=10,out_dim=1000)

    test_input=torch.ones(8,10,1).cuda()
    result=cal_self_att_V(gaussian_matrix=out_martix,input=test_input)

    mp.plot(x_values, result[0,1].cpu().numpy())
    mp.show()
