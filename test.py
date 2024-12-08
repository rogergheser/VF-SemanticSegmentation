import json 
import torch


content = json.load(open('/home/disi/VF-SemanticSegmentation/SAN/output/ade20k_custom_eval/inference/sem_seg_predictions.json'))

results = torch.load('/home/disi/VF-SemanticSegmentation/SAN/output/ade20k_custom_eval/inference/sem_seg_evaluation.pth')

print(results)