from typing import Any
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict
from openood.preprocessors.transform import normalization_dict


import numpy as np

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

def gaussian_kernel(x, sigma=0.1):
    # Compute the squared W1 distances
    squared_distances = np.sum(x ** 2, axis=0)
    
    # Apply the Gaussian kernel function
    K = np.exp(-0.5 * squared_distances / (sigma ** 2))
    
    return K

class OTDP(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.T = 3 # original: 3
        
    def w1_distance(self, p, q):
        return wasserstein_distance(p, q)
    
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):

        logits, feat = net(data, return_feature=True)
        preds = logits.argmax(1)
        softmax = F.softmax(logits / self.T, 1).cpu().numpy()
        
        scores = np.zeros(logits.shape[0])

        logits = normalizer(logits.cpu().numpy())
        feat = normalizer(feat.cpu().numpy())
        softmax = normalizer(softmax)
        
        # print("logits shape:{}".format(logits.shape))  # logits shape:(128, 100)

        for i in range(logits.shape[0]):

            logits_w1_distance = np.min(np.array([self.w1_distance(logits[i], 
                                          (1/(logits[i].shape[0]))*np.ones_like(logits[i])
                                          ),
                                        self.w1_distance(logits[i],
                                          np.zeros_like(logits[i])
                                          )])
                                        )
            
            # only used in ablation
            # logits_w1_distance = self.w1_distance(logits[i], 
            #                               (1/(logits[i].shape[0]))*np.ones_like(logits[i])
            #                               ) # using only mean vector
            
            # only used in ablation
            # logits_w1_distance = self.w1_distance(logits[i], 
            #                               np.zeros_like(logits[i])
            #                               )  # using only zero vector
            
            feature_w1_distance =  np.min(np.array([self.w1_distance(feat[i], 
                                           (1/(feat[i].shape[0]))*np.ones_like(feat[i])
                                           ),
                                         self.w1_distance(feat[i],
                                           np.zeros_like(feat[i])
                                           )]
                                                )
                                        )
            
            # only used in ablation
            # feature_w1_distance =  self.w1_distance(feat[i], 
            #                                (1/(feat[i].shape[0]))*np.ones_like(feat[i])
            #                                )  # using only mean vector
            
            # # only used in ablation
            # feature_w1_distance =  self.w1_distance(feat[i], 
            #                                np.zeros_like(feat[i])
            #                                )  # using only zero vector                          
                                        
            
            softmax_w1_distance = np.min(np.array([self.w1_distance(softmax[i],                     
                                          (1/(softmax[i].shape[0]))*np.ones_like(softmax[i])
                                          ),
                                           self.w1_distance(softmax[i],
                                           np.zeros_like(softmax[i]))]
                                          ))
            
            # only used in ablation
            # softmax_w1_distance = self.w1_distance(softmax[i],                     
            #                               (1/(softmax[i].shape[0]))*np.ones_like(softmax[i])
            #                               ) # using only mean vector
            
            # only used in ablation
            # softmax_w1_distance = self.w1_distance(softmax[i],                     
            #                               np.zeros_like(softmax[i])
            #                               ) # using only zero vector
            

            scores[i] = -((1/3)*feature_w1_distance
                          + (1/3)* logits_w1_distance
                          + (1/3)*softmax_w1_distance
                          )  

        return preds, torch.from_numpy(scores)
