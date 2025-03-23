import os
import torch
import argparse
import numpy as np
from btc_model import BTC_model
from utils.hparams import HParams
from utils import logger

def check_model_dimensions(model_file, config):
    """
    Check the dimensions of the model's output layer and analyze 
    why there's a discrepancy between expected and actual output dimensions.
    """
    logger.info(f"Checking dimensions for model: {model_file}")
    
    # Check if model file exists
    if not os.path.isfile(model_file):
        logger.error(f"Model file not found: {model_file}")
        return
    
    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # First try with weights_only=False for PyTorch 2.6+ compatibility
        try:
            checkpoint = torch.load(model_file, map_location=device, weights_only=False)
            logger.info("Loaded checkpoint with weights_only=False")
        except TypeError:
            # Older PyTorch versions don't have weights_only parameter
            checkpoint = torch.load(model_file, map_location=device)
            logger.info("Loaded checkpoint without weights_only parameter")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
        return
    
    # Check output projection weights dimensions
    output_weights = None
    for key, value in checkpoint['model'].items():
        if 'output_layer.output_projection.weight' in key:
            output_weights = value
            break
    
    if output_weights is not None:
        logger.info(f"Output projection weight shape: {output_weights.shape}")
        output_dimension = output_weights.shape[0]
        input_dimension = output_weights.shape[1] 
        logger.info(f"Model is configured for {output_dimension} output classes with input size {input_dimension}")
    else:
        logger.warning("Could not find output projection weights in checkpoint")
    
    # Create model with config settings
    model = BTC_model(config=config.model).to(device)
    
    # Print model's output dimensions before loading checkpoint
    logger.info("=== Model dimensions BEFORE loading checkpoint ===")
    for name, param in model.named_parameters():
        if 'output' in name:
            logger.info(f"{name}: {param.shape}")
    
    # Load weights from checkpoint
    model.load_state_dict(checkpoint['model'])
    
    # Print model's output dimensions after loading checkpoint
    logger.info("=== Model dimensions AFTER loading checkpoint ===")
    for name, param in model.named_parameters():
        if 'output' in name:
            logger.info(f"{name}: {param.shape}")
    
    # Create a test input and check output shape
    timestep = config.model['timestep']
    feature_size = config.model['feature_size']
    batch_size = 1
    
    logger.info(f"Creating test input with shape: [{batch_size}, {timestep}, {feature_size}]")
    test_input = torch.randn(batch_size, timestep, feature_size).to(device)
    
    # Set model to eval mode and process test input
    model.eval()
    with torch.no_grad():
        # Get self attention output
        self_attn_out, _ = model.self_attn_layers(test_input)
        logger.info(f"Self attention output shape: {self_attn_out.shape}")
        
        # TEST 1: Get raw output from linear projection directly
        raw_logits = model.output_layer.output_projection(self_attn_out)
        logger.info(f"Raw output projection result shape: {raw_logits.shape}")
        
        # TEST 2: Check with probs_out=False (normal prediction mode)
        model.probs_out = False
        output_normal = model.output_layer(self_attn_out)
        logger.info(f"Normal output (probs_out=False) type: {type(output_normal)}")
        if isinstance(output_normal, tuple):
            logger.info(f"  - First element shape: {output_normal[0].shape}")
            logger.info(f"  - Second element shape: {output_normal[1].shape}")
        else:
            logger.info(f"  - Output shape: {output_normal.shape}")
        
        # TEST 3: Check with probs_out=True (logits mode)
        model.probs_out = True
        output_logits = model.output_layer(self_attn_out)
        logger.info(f"Logits output (probs_out=True) type: {type(output_logits)}")
        if isinstance(output_logits, tuple):
            logger.info(f"  - First element shape: {output_logits[0].shape}")
            if len(output_logits) > 1:
                logger.info(f"  - Second element shape: {output_logits[1].shape}")
        else:
            logger.info(f"  - Output shape: {output_logits.shape}")
        
        # Analyze the output
        logger.info("\n=== ANALYSIS ===")
        if raw_logits.dim() == 3 and (isinstance(output_logits, torch.Tensor) and output_logits.dim() < 3 or 
                                    isinstance(output_logits, tuple) and output_logits[0].dim() < 3):
            logger.info("ISSUE DETECTED: Raw projection gives 3D tensor but final output is less than 3D")
            logger.info("This means the SoftmaxOutputLayer is removing the third dimension.")
            
            # Look at the SoftmaxOutputLayer implementation
            logger.info("\nChecking SoftmaxOutputLayer.forward code:")
            
            import inspect
            from utils.transformer_modules import SoftmaxOutputLayer
            logger.info(inspect.getsource(SoftmaxOutputLayer.forward))
            
            # Conclusion
            logger.info("\nRECOMMENDATION:")
            logger.info("To get full 3D logits (batch, time, chords), directly use:")
            logger.info("  raw_logits = model.output_layer.output_projection(self_attn_out)")
            logger.info(f"This gives shape {raw_logits.shape}")
        else:
            logger.info("No dimension loss detected between raw projection and final output")

def main():
    parser = argparse.ArgumentParser(description="Check model dimensions")
    parser.add_argument('--model', type=str, default='./test/btc_model_large_voca.pt', 
                       help='Path to model checkpoint file')
    parser.add_argument('--config', type=str, default='run_config.yaml',
                       help='Path to config file')
    parser.add_argument('--voca', action='store_true', default=True,
                       help='Set to use large vocabulary')
    args = parser.parse_args()
    
    # Set up logging
    logger.logging_verbosity(1)
    
    # Load config
    config = HParams.load(args.config)
    if args.voca:
        config.feature['large_voca'] = True
        config.model['num_chords'] = 170
        logger.info("Using large vocabulary with 170 chord classes")
    else:
        config.model['num_chords'] = 25
        logger.info("Using standard vocabulary with 25 chord classes")
    
    # Check model dimensions
    check_model_dimensions(args.model, config)

if __name__ == "__main__":
    main()
