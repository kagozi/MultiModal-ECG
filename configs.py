PROCESSED_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/'
WAVELETS_PATH = '../santosh_lab/shared/KagoziA/wavelets/cwt/processed_wavelets/'
RESULTS_PATH = '../santosh_lab/shared/KagoziA/wavelets/cwt/processed_wavelets/results/'
DATA_PATH = '../datasets/ECG/'

#Define model configurations to train
configs = [
        
        {'mode': 'scalogram', 'model': 'CWT2DCNN', 'name': 'Scalogram-2DCNN-BCE', 'loss': 'bce'},
        {'mode': 'scalogram', 'model': 'CWT2DCNN', 'name': 'Scalogram-2DCNN-Focal', 'loss': 'focal'},
        {'mode': 'scalogram', 'model': 'CWT2DCNN', 'name': 'Scalogram-2DCNN-Focal-Weighted', 'loss': 'focal_weighted'},
        {'mode': 'phasogram', 'model': 'CWT2DCNN', 'name': 'Phasogram-2DCNN-BCE', 'loss': 'bce'},
        {'mode': 'phasogram', 'model': 'CWT2DCNN', 'name': 'Phasogram-2DCNN-Focal', 'loss': 'focal'},
        {'mode': 'phasogram', 'model': 'CWT2DCNN', 'name': 'Phasogram-2DCNN-Focal-Weighted', 'loss': 'focal_weighted'},
        
        {'mode': 'fusion', 'model': 'CWT2DCNN', 'name': 'Fusion-2DCNN-BCE', 'loss': 'bce'},
        {'mode': 'fusion', 'model': 'CWT2DCNN', 'name': 'Fusion-2DCNN-Focal', 'loss': 'focal'},
        {'mode': 'fusion', 'model': 'CWT2DCNN', 'name': 'Fusion-2DCNN-Focal-Weighted', 'loss': 'focal_weighted'},
        {'mode': 'both', 'model': 'DualStream', 'name': 'DualStream-CNN-Focal-Weighted', 'loss': 'focal_weighted'},
        {'mode': 'both', 'model': 'DualStream', 'name': 'DualStream-CNN-BCE', 'loss': 'bce'},
        {'mode': 'both', 'model': 'DualStream', 'name': 'DualStream-CNN-Focal', 'loss': 'focal'},
             
        {'mode': 'fusion', 'model': 'SwinTransformerEarlyFusion', 'name': 'EarlyFusion-Swin-Focal-Learned', 'loss': 'focal'},
        {'mode': 'fusion', 'model': 'SwinTransformerEarlyFusion', 'name': 'EarlyFusion-Swin-BCE-Learned', 'loss': 'bce'},
        {'mode': 'both', 'model': 'SwinTransformerLateFusion', 'name': 'LateFusion-Swin-BCE-Learned', 'loss': 'bce'},
        
        # ResNet50 variants
        {'mode': 'scalogram', 'model': 'ResNet50ECG', 'name': 'Scalogram-ResNet50-Learned', 'loss': 'focal_weighted'},
         {'mode': 'scalogram', 'model': 'ResNet50ECG', 'name': 'Scalogram-ResNet50-Focal', 'loss': 'focal'},
        {'mode': 'phasogram', 'model': 'ResNet50ECG', 'name': 'Phasogram-ResNet50-Focal', 'loss': 'focal'},
        {'mode': 'scalogram', 'model': 'ResNet50ECG', 'name': 'Scalogram-ResNet50-BCE', 'loss': 'bce'},
        {'mode': 'phasogram', 'model': 'ResNet50ECG', 'name': 'Phasogram-ResNet50-BCE', 'loss': 'bce'},
        {'mode': 'scalogram', 'model': 'ResNet50ECG', 'name': 'Scalogram-ResNet50-Focal-Weighted', 'loss': 'focal_weighted'},
        {'mode': 'phasogram', 'model': 'ResNet50ECG', 'name': 'Phasogram-ResNet50-Focal-Weighted', 'loss': 'focal_weighted'},
        
        # ResNet50 Early Fusion
        {'mode': 'fusion', 'model': 'ResNet50EarlyFusion', 'name': 'EarlyFusion-ResNet50-Learned', 'loss': 'focal_weighted'},
        {'mode': 'fusion', 'model': 'ResNet50EarlyFusion', 'name': 'EarlyFusion-ResNet50-Focal', 'loss': 'focal'},
        {'mode': 'fusion', 'model': 'ResNet50EarlyFusion', 'name': 'EarlyFusion-ResNet50-BCE', 'loss': 'bce'},
        {'mode': 'both', 'model': 'ResNet50LateFusion', 'name': 'LateFusion-ResNet50-BCE', 'loss': 'bce'},
        {'mode': 'both', 'model': 'ResNet50LateFusion', 'name': 'LateFusion-ResNet50-Focal', 'loss': 'focal'},
        {'mode': 'both', 'model': 'ResNet50LateFusion', 'name': 'LateFusion-ResNet50-Focal-Weighted', 'loss': 'focal_weighted'},
        
        # EfficientNet ECG variants
        {'mode': 'scalogram', 'model': 'EfficientNetECG', 'name': 'Scalogram-EfficientNet-ECG-Focal', 'loss': 'focal'},
        {'mode': 'scalogram', 'model': 'EfficientNetECG', 'name': 'Scalogram-EfficientNet-ECG-BCE', 'loss': 'bce'},
        {'mode': 'phasogram', 'model': 'EfficientNetECG', 'name': 'Phasogram-EfficientNet-ECG-BCE', 'loss': 'bce'},
        {'mode': 'phasogram', 'model': 'EfficientNetECG', 'name': 'Phasogram-EfficientNet-ECG-Focal', 'loss': 'focal'},
        {'mode': 'scalogram', 'model': 'EfficientNetECG', 'name': 'Scalogram-EfficientNet-ECG-Focal-Weighted', 'loss': 'focal_weighted'},
        {'mode': 'phasogram', 'model': 'EfficientNetECG', 'name': 'Phasogram-EfficientNet-ECG-Focal-Weighted', 'loss': 'focal_weighted'},


        # EfficientNet Early Fusion variants
        {'mode': 'fusion', 'model': 'EfficientNetEarlyFusion', 'name': 'EfficientNetEarlyFusion-Focal', 'loss': 'focal'},
        {'mode': 'fusion', 'model': 'EfficientNetEarlyFusion', 'name': 'EfficientNetEarlyFusion-BCE', 'loss': 'bce'},
        {'mode': 'fusion', 'model': 'EfficientNetEarlyFusion', 'name': 'EfficientNetEarlyFusion-Focal-Weighted', 'loss': 'focal_weighted'},
        
        ## EfficientNet Late Fusion variants
        {'mode': 'both', 'model': 'EfficientNetLateFusion', 'name': 'LateFusion-EfficientNet-Focal', 'loss': 'focal'},
        {'mode': 'both', 'model': 'EfficientNetLateFusion', 'name': 'LateFusion-EfficientNet-BCE', 'loss': 'bce'},
        {'mode': 'both', 'model': 'EfficientNetLateFusion', 'name': 'LateFusion-EfficientNet-Focal-Weighted', 'loss': 'focal_weighted'},
        
        
        # SwinTransformerECG                 
        {'mode': 'scalogram', 'model': 'SwinTransformerECG', 'name': 'Scalogram-SwinTransformerECG-BCE', 'loss': 'bce'},
        {'mode': 'phasogram', 'model': 'SwinTransformerECG', 'name': 'Phasogram-SwinTransformerECG-BCE', 'loss': 'bce'},
        
        {'mode': 'scalogram', 'model': 'SwinTransformerECG', 'name': 'Scalogram-SwinTransformerECG-Focal', 'loss': 'focal'},
        {'mode': 'phasogram', 'model': 'SwinTransformerECG', 'name': 'Phasogram-SwinTransformerECG-Focal', 'loss': 'focal'},
        
        {'mode': 'scalogram', 'model': 'SwinTransformerECG', 'name': 'Scalogram-SwinTransformerECG-Focal-Weighted', 'loss': 'focal_weighted'},
        {'mode': 'phasogram', 'model': 'SwinTransformerECG', 'name': 'Phasogram-SwinTransformerECG-Focal-Weighted', 'loss': 'focal_weighted'},
        
        # SwinTransformerEarlyFusion     
        {'mode': 'fusion', 'model': 'SwinTransformerEarlyFusion', 'name': 'EarlyFusion-Swin-Focal-Weighted', 'loss': 'focal_weighted'},
        {'mode': 'fusion', 'model': 'SwinTransformerEarlyFusion', 'name': 'EarlyFusion-Swin-Focal', 'loss': 'focal'},
        
        # SwinTransformerLateFusion
        {'mode': 'both', 'model': 'SwinTransformerLateFusion', 'name': 'LateFusion-Swin-Focal-Weighted', 'loss': 'focal_weighted'},
        {'mode': 'both', 'model': 'SwinTransformerLateFusion', 'name': 'LateFusion-Swin-BCE', 'loss': 'bce'},
        {'mode': 'both', 'model': 'SwinTransformerLateFusion', 'name': 'LateFusion-Swin-Focal', 'loss': 'focal'},
         
        # Hybrid Swin variants
        {'mode': 'scalogram', 'model': 'HybridSwinTransformerECG', 'name': 'Scalogram-HybridSwin-Learned', 'loss': 'focal_weighted'},
        {'mode': 'scalogram', 'model': 'HybridSwinTransformerECG', 'name': 'Scalogram-HybridSwin-Focal', 'loss': 'focal'},
        {'mode': 'phasogram', 'model': 'HybridSwinTransformerECG', 'name': 'Phasogram-HybridSwin-Focal', 'loss': 'focal'},
        
        {'mode': 'scalogram', 'model': 'HybridSwinTransformerECG', 'name': 'Scalogram-HybridSwin-BCE', 'loss': 'bce'},
        {'mode': 'phasogram', 'model': 'HybridSwinTransformerECG', 'name': 'Phasogram-HybridSwin-BCE', 'loss': 'bce'},
        
        {'mode': 'fusion', 'model': 'HybridSwinTransformerEarlyFusion', 'name': 'EarlyFusion-HybridSwin-Focal', 'loss': 'focal'},
        {'mode': 'fusion', 'model': 'HybridSwinTransformerEarlyFusion', 'name': 'EarlyFusion-HybridSwin-BCE', 'loss': 'bce'},
        {'mode': 'fusion', 'model': 'HybridSwinTransformerEarlyFusion', 'name': 'EarlyFusion-HybridSwin', 'loss': 'focal_weighted'},
        
        {'mode': 'phasogram', 'model': 'HybridSwinTransformerECG', 'name': 'Scalogram-HybridSwin-Learned', 'loss': 'focal_weighted'},
        {'mode': 'both', 'model': 'HybridSwinTransformerLateFusion', 'name': 'LateFusion-HybridSwin-Focal', 'loss': 'focal'},
        {'mode': 'both', 'model': 'HybridSwinTransformerLateFusion', 'name': 'LateFusion-HybridSwin-BCE', 'loss': 'bce'},
        {'mode': 'both', 'model': 'HybridSwinTransformerLateFusion', 'name': 'LateFusion-HybridSwin-Learned', 'loss': 'focal_weighted'},
]