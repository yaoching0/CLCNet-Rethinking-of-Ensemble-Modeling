from asyncio.format_helpers import _format_callback_source
import timm
import torch 
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np 
import os
from timm.data.parsers.parser_image_folder import find_images_and_targets
from timm.data import create_dataset
from CLCNet.loader import create_loader
import copy
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import tzip
from CLCNet.tab_model import CLCNet
import time
import pandas as pd
import glob
import logging
import argparse
from CLCNet.cascade_structure_system import CLCNet_cascade_system

parser = argparse.ArgumentParser()

# base argument
parser.add_argument('--imagenet-split', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'data','imagenet_splits_5'),
                        help='Path of imagenet val set that divided into five for cross validation.')
parser.add_argument('--weight-save', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'weights'),
                        help='Path of CLCNet weights to load.') 
parser.add_argument('--log', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'log.txt'),
                        help='Path of saving log.')  
parser.add_argument('--batch-size', type=int, default=16,
                        help='Inference batch size.')                                               
parser.add_argument('--shallow-model', type=str, default='tf_efficientnet_b0',
                        help='Shallow model of cascade structure system.')  
parser.add_argument('--shallow-model-FLOPs', type=float, default=0.39,
                        help='FLOPs of shallow model of cascade structure system.')   
parser.add_argument('--deep-model', type=str, default='tf_efficientnet_b4',
                        help='Deep model of cascade structure system.')  
parser.add_argument('--deep-model-FLOPs', type=float, default=4.2,
                        help='FLOPs of deep model of cascade structure system.')  
parser.add_argument('--threshold-searching', action='store_true', default=False,
                        help='Obtain the accuracy and FLOPs of the system under different thresholds.')  
parser.add_argument('--threshold', type=float, default=0.5,
                        help='Calculate the FLOPs and accuracy of the system under single threshold (threshold-searching will be disabled).')
parser.add_argument('--performance-result', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'performance-result.csv'),
                        help='The performance of the system under different thresholds.')        

cf = parser.parse_args()                                            


if __name__ == '__main__':

    if cf.threshold_searching is True:
        search_scope=np.linspace(0.05,1.01,49)
    else:
        search_scope=[cf.threshold,]
    
    #k fold
    num_k=5

    # The path where the output log is saved
    log_path=cf.log
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Record the metrics under different threshold
    output=pd.DataFrame(columns=['threshold', 'accuracy', 'FLOPs'])

    # define shallow model
    model_s= timm.create_model(cf.shallow_model, pretrained=True)
    data_config_s=model_s.default_cfg
    model_s.eval()
    print('shallow model config: ',data_config_s)

    # define deep model
    model_d= timm.create_model(cf.deep_model, pretrained=True)
    data_config_d=model_d.default_cfg
    model_d.eval()
    print('deep model config: ',data_config_d)

    # Evaluate with different thresholds
    for s in search_scope:
        total_num=0
        correct_num=0

        # Count how many times the deep model is called
        call_deep=[0,]

        # If the CLCNet output score is less than this threshold, it will be handed over to the deep model for further judgment
        threshold=s

        for k in range(num_k):
            # define CLCNet
            clcnet = CLCNet()
            saved_filepath=glob.glob(os.path.join(cf.weight_save,'CLCNet_imagenet_split_'+str(k+1)+'*'))

            if len(saved_filepath) !=1:
                raise ValueError(f'Searched multiple paths for CLCNet split_{k+1}, please remove the split_{k+1} weights that are not intended to be used.')

            saved_filepath=saved_filepath[0]
            clcnet.load_model(saved_filepath)
            
            
            samples, class_to_idx=find_images_and_targets(folder=os.path.join(cf.imagenet_split,'split_part_'+str(k+1)),class_to_idx=None)

        
            idx_to_class={}
            for key,value in class_to_idx.items():
                idx_to_class[value] = key


            #The transforms of the shallow model and the deep model are likely to be inconsistent, so the loader and dataset should be defined separately
            dataset_eval_s= create_dataset(
                '', root=os.path.join(cf.imagenet_split,'split_part_'+str(k+1)), split='validation', is_training=False)

            dataset_eval_d= create_dataset(
                '', root=os.path.join(cf.imagenet_split,'split_part_'+str(k+1)), split='validation', is_training=False)

            loader_eval_s = create_loader(
                dataset_eval_s,
                input_size=data_config_s['input_size'],
                batch_size=cf.batch_size,
                is_training=False,
                use_prefetcher=True,
                interpolation=data_config_s['interpolation'],
                mean=data_config_s['mean'],
                std=data_config_s['std'],
                num_workers=1,
                distributed=False,
                crop_pct=data_config_s['crop_pct'],
                pin_memory=False,
            )

            loader_eval_d = create_loader(
                dataset_eval_d,
                input_size=data_config_d['input_size'],
                batch_size=cf.batch_size,
                is_training=False,
                use_prefetcher=True,
                interpolation=data_config_d['interpolation'],
                mean=data_config_d['mean'],
                std=data_config_d['std'],
                num_workers=1,
                distributed=False,
                crop_pct=data_config_d['crop_pct'],
                pin_memory=False,
            )

            model_s.cuda()
            model_d.cuda()

            with torch.no_grad():
                for (input_s, target_s),(input_d, target_d) in tqdm(tzip(loader_eval_s,loader_eval_d)):
                    
                    result=CLCNet_cascade_system(model_s=model_s,model_d=model_d,clcnet=clcnet,input_s=input_s,input_d=input_d,threshold=threshold,call_deep=call_deep)
                    result=result.argmax(dim=1)

                    # Record how many predictions are correct
                    correct_num+=torch.nonzero(target_s==result).numel()
                    total_num+=len(input_s)
                    
            
        
        output=output.append(pd.DataFrame([[threshold,round((correct_num/total_num)*100,4),cf.shallow_model_FLOPs+(call_deep[0]/total_num)*cf.deep_model_FLOPs]],columns=['threshold', 'accuracy', 'FLOPs']))
        logger.info(f'threshold:{threshold} accuracy:{round((correct_num/total_num)*100,4)} FLOPs:{cf.shallow_model_FLOPs+(call_deep[0]/total_num)*cf.deep_model_FLOPs}')
        
    output.to_csv(cf.performance_result,index=None)
