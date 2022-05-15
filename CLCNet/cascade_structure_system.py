import torch.nn.functional as F
import torch
import numpy as np

def CLCNet_cascade_system(model_s,model_d,clcnet,input_s,input_d,threshold,call_deep):
    
    result_s=model_s(input_s)
    result_s=F.softmax(result_s, dim=1)
    
    #Sort from largest to smallest
    sorted, indices = torch.sort(result_s, descending=True)
    
    result_c=clcnet.predict(sorted)

    # Find the index of the sample with confidence less than threshold
    fliter_indices=torch.tensor(np.nonzero((result_c<threshold)[:,0])[0]).cuda()

    if len(fliter_indices) !=0:

        result_c=torch.from_numpy(result_c).cuda()

        # Record how many times the deep model was called
        call_deep[0]+=fliter_indices.numel()

        # Make further predict on the deep model
        result_d=model_d(input_d[fliter_indices])
        result_d=F.softmax(result_d, dim=1)
        
        # Give the result of the deep model to CLCNet again
        sorted_d, indices = torch.sort(result_d, descending=True)
        result_c_d=torch.from_numpy(clcnet.predict(sorted_d)).cuda()

        result_d=result_d[(result_c_d>result_c[fliter_indices]).view(-1)]

        # Only assign the results of the deep model with high confidence to the final output
        fliter_indices=fliter_indices[(result_c[fliter_indices]<result_c_d).view(-1)]

        if len(fliter_indices)!=0:
            
            result_s[fliter_indices]=result_d
    
    return result_s