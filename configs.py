#Define model configurations to train

WAVELETS_PATH = '../santosh_lab/shared/KagoziA/wavelets/cwt/processed_wavelets_ptbxl_scarpiniti/'
RESULTS_PATH = '../santosh_lab/shared/KagoziA/wavelets/cwt/pprocessed_wavelets_ptbxl_scarpiniti/results/'
PROCESSED_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/'

BASELINE_RESULTS_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/results/'
DATA_PATH = '../datasets/ECG/'

configs = [
        {'mode': 'scalogram', 'model': 'CWT2DCNN', 'name': 'Scalogram-2DCNN-BCE', 'loss': 'bce'},
        {'mode': 'scalogram', 'model': 'CWT2DCNN', 'name': 'Scalogram-2DCNN-Focal', 'loss': 'focal'},
        {'mode': 'fusion', 'model': 'CWT2DCNN', 'name': 'Fusion-2DCNN-BCE', 'loss': 'bce'},
        {'mode': 'fusion', 'model': 'CWT2DCNN', 'name': 'Fusion-2DCNN-Focal', 'loss': 'focal'},
        {'mode': 'both', 'model': 'DualStream', 'name': 'DualStream-CNN-BCE', 'loss': 'bce'},
        {'mode': 'both', 'model': 'DualStream', 'name': 'DualStream-CNN-Focal', 'loss': 'focal'},
        
                
        ## ResNet50 variants
        {'mode': 'scalogram', 'model': 'ResNet50ECG', 'adapter': 'learned', 'name': 'Scalogram-ResNet50-BCE', 'loss': 'bce'},
        {'mode': 'fusion', 'model': 'ResNet50EarlyFusion', 'adapter': 'learned', 'name': 'EarlyFusion-ResNet50-BCE', 'loss': 'bce'},
        {'mode': 'both', 'model': 'ResNet50LateFusion', 'adapter': 'learned', 'name': 'LateFusion-ResNet50-BCE', 'loss': 'bce'},
        {'mode': 'phasogram', 'model': 'ResNet50ECG', 'adapter': 'learned', 'name': 'Phasogram-ResNet50-BCE', 'loss': 'bce'},
        
        {'mode': 'scalogram', 'model': 'ResNet50ECG', 'adapter': 'learned', 'name': 'Scalogram-ResNet50-Focal', 'loss': 'focal'},
        {'mode': 'fusion', 'model': 'ResNet50EarlyFusion', 'adapter': 'learned', 'name': 'EarlyFusion-ResNet50-Focal', 'loss': 'focal'},
        {'mode': 'both', 'model': 'ResNet50LateFusion', 'adapter': 'learned', 'name': 'LateFusion-ResNet50-Focal', 'loss': 'focal'},
        {'mode': 'phasogram', 'model': 'ResNet50ECG', 'adapter': 'learned', 'name': 'Phasogram-ResNet50-Focal', 'loss': 'focal'},
        
        ## EfficientNet variants
        {'mode': 'scalogram', 'model': 'EfficientNetFusionECG', 'adapter': 'learned', 'name': 'Scalogram-EfficientNet-BCE', 'loss': 'bce'},
        {'mode': 'fusion', 'model': 'EfficientNetEarlyFusion', 'adapter': 'learned', 'name': 'EarlyFusion-EfficientNet-BCE', 'loss': 'bce'},
        {'mode': 'both', 'model': 'EfficientNetLateFusion', 'adapter': 'learned', 'name': 'LateFusion-EfficientNet-BCE', 'loss': 'bce'},
        {'mode': 'phasogram', 'model': 'EfficientNetFusionECG', 'adapter': 'learned', 'name': 'Phasogram-EfficientNet-BCE', 'loss': 'bce'},
        
        {'mode': 'scalogram', 'model': 'EfficientNetFusionECG', 'adapter': 'learned', 'name': 'Scalogram-EfficientNet-Focal', 'loss': 'focal'},
        {'mode': 'fusion', 'model': 'EfficientNetEarlyFusion', 'adapter': 'learned', 'name': 'EarlyFusion-EfficientNet-Focal', 'loss': 'focal'},
        {'mode': 'both', 'model': 'EfficientNetLateFusion', 'adapter': 'learned', 'name': 'LateFusion-EfficientNet-Focal', 'loss': 'focal'},
        {'mode': 'phasogram', 'model': 'EfficientNetFusionECG', 'adapter': 'learned', 'name': 'Phasogram-EfficientNet-Focal', 'loss': 'focal'},

        
        # {'mode': 'fusion', 'model': 'SwinTransformerEarlyFusion', 'name': 'EarlyFusion-Swin-Focal-Learned', 'loss': 'focal', 'adapter': 'learned'},
        # {'mode': 'fusion', 'model': 'SwinTransformerEarlyFusion', 'name': 'EarlyFusion-Swin-BCE-Learned', 'loss': 'bce', 'adapter': 'learned'},
        # {'mode': 'fusion', 'model': 'SwinTransformerEarlyFusion', 'name': 'EarlyFusion-Swin-Focal-Select', 'loss': 'focal', 'adapter': 'select'},
        # {'mode': 'both', 'model': 'SwinTransformerLateFusion', 'name': 'LateFusion-Swin-Focal-Learned', 'loss': 'focal', 'adapter': 'learned'},
        # {'mode': 'both', 'model': 'SwinTransformerLateFusion', 'name': 'LateFusion-Swin-BCE-Learned', 'loss': 'bce', 'adapter': 'learned'},
        # {'mode': 'both', 'model': 'SwinTransformerLateFusion', 'name': 'LateFusion-Swin-Focal-Select', 'loss': 'focal', 'adapter': 'select'},
        # {'mode': 'both', 'model': 'EfficientNetLateFusion', 'name': 'EfficientNetLateFusion-Focal-Learned', 'loss': 'focal', 'adapter': 'learned'},
        # {'mode': 'both', 'model': 'ViTLateFusion', 'name': 'ViTLateFusion-Focal-Learned', 'loss': 'focal', 'adapter': 'learned'},
        
        # {'mode': 'scalogram', 'model': 'ViTECG', 'name': 'ViT-ECG-BCE-Learned', 'loss': 'bce', 'adapter': 'learned'},
        # {'mode': 'scalogram', 'model': 'EfficientNetFusionECG', 'name': 'EfficientNet-ECG-Focal-Learned', 'loss': 'focal', 'adapter': 'learned'},
         
        # # Hybrid Swin variants
        # {'mode': 'scalogram', 'model': 'HybridSwinTransformerECG', 'adapter': 'learned', 'name': 'Scalogram-HybridSwin-Learned', 'loss': 'focal_weighted'},
        # {'mode': 'fusion', 'model': 'HybridSwinTransformerEarlyFusion', 'name': 'EarlyFusion-HybridSwin', 'loss': 'focal_weighted'},
        

        # {'mode': 'phasogram', 'model': 'HybridSwinTransformerECG', 'adapter': 'learned', 'name': 'Scalogram-HybridSwin-Learned', 'loss': 'focal_weighted'},
        # {'mode': 'both', 'model': 'HybridSwinTransformerLateFusion', 'adapter': 'learned', 'name': 'LateFusion-HybridSwin-Learned', 'loss': 'focal_weighted'},
        


]