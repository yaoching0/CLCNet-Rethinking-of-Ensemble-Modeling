import torch
import pandas as pd
from PIL import ImageDraw,ImageFont,Image
import os
import numpy as np
import timm
from CLCNet.transforms_factory import transforms_imagenet_eval
from timm.data.parsers.parser_image_folder import find_images_and_targets
from timm.data import create_dataset, create_loader
from tqdm import tqdm
import warnings
import argparse
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

# base argument
parser.add_argument('--data-path', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'data','imagenet_splits_5'),
                        help='Path of data to generate feature for CLCNet training.')
parser.add_argument('--output', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'cls_model_feature.csv'),
                        help='Path of saving generated features.')
parser.add_argument('--model', type=str, default='tf_efficientnet_b4',
                        help='The classification model name for generating the features.')
parser.add_argument('--num_classes', type=int, default=1000,
                        help='Number of classification.')
parser.add_argument('--model-weight', type=str, default=None,
                        help='Weight of classification model to load.')   
parser.add_argument('--batch-size', type=int, default=4,
                        help='Inference batch size.')                       

cf = parser.parse_args() 

'''
output format:
       [filename][sorted n-dim(ImageNet:n=1000) classifcation probilities][label (1 or 0)] ,
       [filename][sorted n-dim(ImageNet:n=1000) classifcation probilities][label (1 or 0)] ,
        ...
'''

def model_inference(img_list,model1,idx_to_class,transform1):

    stack_list1=[]

    for img in img_list:
        img1=transform1(img.copy())
        img1=img1.type(torch.FloatTensor)
        stack_list1.append(img1)

    img1=torch.stack(stack_list1,dim=0)

    # Model gets predictions
    predict_result1=model1(img1.cuda())


    # Define softmax
    sm=torch.nn.Softmax(dim=1)

    predict_result1=sm(predict_result1)
    
    return predict_result1


# Get the dictionary of categories and indexes corresponding to the classification model
samples, class_to_idx=find_images_and_targets(folder=cf.data_path,class_to_idx=None)

idx_to_class={}
for key,value in class_to_idx.items():
    idx_to_class[value] = key


if cf.model_weight is not None:
    model= timm.create_model(cf.model,  num_classes=cf.num_classes,checkpoint_path=cf.model_weight)
else:
    model= timm.create_model(cf.model, pretrained=True)

data_config=model.default_cfg
model.eval()
model.cuda()

# The transfom used when classification model inferring
cls_transform=transforms_imagenet_eval(
    img_size=data_config['input_size'][-2:],
    interpolation=data_config['interpolation'],
    mean=data_config['mean'],
    std=data_config['std'],
    crop_pct=data_config['crop_pct'],)


feature=pd.DataFrame()
image_list=[]
temp_picname=[]
class_id=[]


for file in tqdm(samples):
    image=Image.open(file[0]).convert("RGB") # imagenet exists grayscale images
    image_list.append(image)

    classname=file[0].split('\\')[-2]
    filename=file[0].split('\\')[-1]
    temp_picname.append(filename)

    # id of groundTruth
    class_id.append(class_to_idx[classname])

    if(len(image_list)==cf.batch_size):
        predict=model_inference(image_list,model,idx_to_class,cls_transform)
        
        image_list=[]

        predict=predict.cpu().detach().numpy() #.tolist()
  
        # whether the model predicts correctly
        label=pd.DataFrame(np.argmax(predict,axis=1)==class_id) #[True,False,...]
        
        # Sort from largest to smallest
        predict.sort(axis=1)
        predict = predict[:, ::-1]

        predict=pd.DataFrame(predict)
        
        predict=pd.concat([pd.DataFrame(temp_picname),predict,label],axis=1,ignore_index=True)
        predict.iloc[:,-1]=predict.iloc[:,-1].map({True:1,False:0})

        feature=feature.append(predict,ignore_index=True)
        temp_picname=[]
        class_id=[]
        
print('DONE')

feature.to_csv(cf.output,index=False,header=False,encoding="utf_8_sig")

