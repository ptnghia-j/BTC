import os
import argparse
import numpy as np
import torch
import librosa
import warnings
from utils.hparams import HParams
from btc_model import BTC_model
from utils.mir_eval_modules import audio_file_to_features, idx2chord, idx2voca_chord, get_audio_paths
from utils import logger

# Filter out the FutureWarning from librosa
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

def main():
    parser = argparse.ArgumentParser(description="Process audio from data/fma_small into labels and spectrograms.")
    parser.add_argument('--voca', default=True, type=lambda x: (str(x).lower()=='true'))
    parser.add_argument('--audio_dir', type=str, default='./data/fma_small')
    parser.add_argument('--save_dir', type=str, default='./data/logits/synth')
    # New argument to indicate dataset type
    parser.add_argument('--dataset', type=str, default='fma', choices=['fma', 'maestro'])
    # New argument to enable saving logits
    parser.add_argument('--save_logits', action='store_true', help='Save teacher logits')
    args = parser.parse_args()
    
    logger.logging_verbosity(1)
    config = HParams.load("run_config.yaml")
    if args.voca:
        config.feature['large_voca'] = True
        config.model['num_chords'] = 170
        model_file = './test/btc_model_large_voca.pt'
        idx_to_chord = idx2voca_chord()
        logger.info("label type: large voca")
    else:
        config.model['num_chords'] = 25
        model_file = './test/btc_model.pt'
        idx_to_chord = idx2chord
        logger.info("label type: Major and minor")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BTC_model(config=config.model).to(device)
    if os.path.isfile(model_file):
        # Remove weights_only parameter as it's not supported in PyTorch 1.9.0
        checkpoint = torch.load(model_file, map_location=device)
        mean = checkpoint['mean']
        std = checkpoint['std']
        model.load_state_dict(checkpoint['model'])
        logger.info("restore model")
    else:
        mean = 0.0
        std = 1.0
    
    model.eval()
    
    # Get project root directory using current working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.getcwd()
    
    # Create base save directories for spectrograms and labels in the root directory
    base_spec_save_dir = os.path.join(project_root, args.save_dir, "spectrograms")
    base_lab_save_dir = os.path.join(project_root, args.save_dir, "labels")
    os.makedirs(base_spec_save_dir, exist_ok=True)
    os.makedirs(base_lab_save_dir, exist_ok=True)
    
    # Create directory for logits if needed
    if args.save_logits:
        base_logits_save_dir = os.path.join(project_root, args.save_dir, "logits")
        os.makedirs(base_logits_save_dir, exist_ok=True)
        logger.info(f"Will save teacher logits to: {base_logits_save_dir}")
    
    # Get all audio files
    audio_paths = get_audio_paths(args.audio_dir)
    logger.info(f"Found {len(audio_paths)} audio files.")
    
    # For maestro, use the dataset folder structure
    if args.dataset == "maestro":
        # Do not pre-create fixed folders; use relative path from audio_dir
        logger.info("Processing maestro dataset with organized subfolders.")
    
    # Create sub-folders for spectrograms and labels
    if args.dataset != "maestro":
        # For non-maestro datasets, pre-generate folders
        for i in range(343):  # Create folders for 000-342 for 343 days of audio
            folder_name = f"{i:03d}"
            os.makedirs(os.path.join(base_spec_save_dir, folder_name), exist_ok=True)
            os.makedirs(os.path.join(base_lab_save_dir, folder_name), exist_ok=True)
            if args.save_logits:
                os.makedirs(os.path.join(base_logits_save_dir, folder_name), exist_ok=True)
    
    # Process each audio file
    processed_count = 0
    skipped_count = 0
    for audio_path in audio_paths:
        logger.info(f"Processing: {audio_path}")
        try:
            # Check file size first to skip obviously problematic files
            file_size = os.path.getsize(audio_path)
            if file_size < 10000:  # Skip files smaller than 10KB
                logger.info(f"Skipping file {audio_path}: File too small ({file_size} bytes)")
                skipped_count += 1
                continue
                
            # Extract feature, time per hop, song length
            feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, config)
            
            # Skip if feature extraction produced nothing useful
            if feature is None or feature.shape[1] < 10:  # Ensure we have at least 10 frames
                logger.info(f"Skipping file {audio_path}: Insufficient audio data extracted")
                skipped_count += 1
                continue
                
        except Exception as e:
            logger.info(f"Skipping file {audio_path}: {e}")
            skipped_count += 1
            continue
        
        # Determine subfolder: for maestro, use the relative path first folder; otherwise group by thousands.
        file_id = os.path.basename(audio_path).split('.')[0]
        if args.dataset == "maestro":
            rel_path = os.path.relpath(os.path.dirname(audio_path), args.audio_dir)
            folder_name = rel_path.split(os.sep)[0]
        else:
            try:
                folder_name = f"{int(file_id)//1000:03d}"
            except ValueError:
                folder_name = "000"
        
        spec_save_dir = os.path.join(base_spec_save_dir, folder_name)
        lab_save_dir = os.path.join(base_lab_save_dir, folder_name)
        os.makedirs(spec_save_dir, exist_ok=True)
        os.makedirs(lab_save_dir, exist_ok=True)
        
        if args.save_logits:
            logits_save_dir = os.path.join(base_logits_save_dir, folder_name)
            os.makedirs(logits_save_dir, exist_ok=True)
        
        # Save spectrogram (transposed) as a numpy file
        spec = feature.T  # CQT spectrogram with shape [time, frequency]
        base = os.path.splitext(os.path.basename(audio_path))[0]
        npy_save_path = os.path.join(spec_save_dir, base + "_spec.npy")
        np.save(npy_save_path, spec)
        
        # Comment out verbose logging
        # logger.info(f"Saved spectrogram to: {npy_save_path}")
        # logger.info(f"Spectrogram shape: {spec.shape}")
        
        # Normalize and pad the feature for model inference
        spec_norm = (spec - mean) / std
        n_timestep = config.model['timestep']
        num_pad = n_timestep - (spec_norm.shape[0] % n_timestep)
        spec_norm = np.pad(spec_norm, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        num_instance = spec_norm.shape[0] // n_timestep
        
        time_unit = feature_per_second
        start_time = 0.0
        lines = []
        
        # Create arrays to store all logits if saving is enabled
        if args.save_logits:
            all_logits = []
        
        spec_tensor = torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            model.eval()
            
            # Store original probs_out setting to restore later
            original_probs_out = model.probs_out
            
            for t in range(num_instance):
                chunk = spec_tensor[:, n_timestep*t:n_timestep*(t+1), :]
                self_attn_out, _ = model.self_attn_layers(chunk)
                
                # Extract logits if saving is enabled, forcing full distribution output.
                if args.save_logits:
                    # Skip the model.output_layer() completely and directly access the raw projection
                    # This gives us the full 3D tensor with all 170 chord dimensions
                    raw_logits = model.output_layer.output_projection(self_attn_out)
                    # Comment out verbose logging
                    # logger.info(f"Raw output projection shape: {raw_logits.shape}")
                    
                    # Use the raw logits directly - this is what we want to save
                    logits = raw_logits
                    all_logits.append(logits.cpu().numpy())
                
                # For label prediction, use the normal path
                try:
                    model.probs_out = True
                    output_for_pred = model.output_layer(self_attn_out)
                    # Handle both cases again
                    if isinstance(output_for_pred, tuple):
                        logits_for_pred = output_for_pred[0]
                    else:
                        logits_for_pred = output_for_pred
                    model.probs_out = original_probs_out
                    
                    # Comment out verbose logging
                    # logger.info(f"Logits shape before argmax: {logits_for_pred.shape}")
                    
                    # If we need 3D predictions but got 2D, reshape using argmax on the raw projection
                    if logits_for_pred.dim() < 3 and raw_logits.dim() == 3:
                        # Comment out verbose logging
                        # logger.info("Using raw logits for prediction since output is not 3D")
                        pred = torch.argmax(raw_logits, dim=-1)  # shape: [batch, time]
                        pred = pred.squeeze()
                    elif logits_for_pred.dim() == 3:
                        pred = torch.argmax(logits_for_pred, dim=-1)
                        pred = pred.squeeze()
                    elif logits_for_pred.dim() == 2:
                        pred = torch.argmax(logits_for_pred, dim=-1)
                    else:
                        logger.info(f"Unexpected logits dimensionality: {logits_for_pred.dim()}")
                        pred = torch.argmax(logits_for_pred.view(1, -1), dim=-1)
                    
                    # Comment out verbose logging
                    # logger.info(f"Prediction shape after argmax: {pred.shape if pred.dim() > 0 else '0-dim scalar'}")
                except Exception as e:
                    logger.error(f"Error during prediction for {audio_path}: {str(e)}")
                    # Use a default prediction
                    pred = torch.tensor([169])  # 'N' in large vocabulary
                
                # Handle different pred shapes
                if pred.dim() == 0:  # It's a scalar
                    # Use same chord value for this entire chunk
                    chord_value = pred.item()
                    for i in range(n_timestep):
                        if t==0 and i==0:
                            prev_chord = chord_value
                            continue
                        # We only change the label if this is a different chord from the previous
                        if chord_value != prev_chord:
                            lines.append('%.3f %.3f %s\n' % (start_time, time_unit*(n_timestep*t+i), idx_to_chord[prev_chord]))
                            start_time = time_unit*(n_timestep*t+i)
                            prev_chord = chord_value
                        if t==num_instance-1 and i+num_pad==n_timestep:
                            if start_time != time_unit*(n_timestep*t+i):
                                lines.append('%.3f %.3f %s\n' % (start_time, time_unit*(n_timestep*t+i), idx_to_chord[prev_chord]))
                            break
                else:  # Normal case with a tensor of predictions
                    for i in range(min(n_timestep, pred.shape[0])):
                        if t==0 and i==0:
                            prev_chord = pred[i].item()
                            continue
                        if pred[i].item() != prev_chord:
                            lines.append('%.3f %.3f %s\n' % (start_time, time_unit*(n_timestep*t+i), idx_to_chord[prev_chord]))
                            start_time = time_unit*(n_timestep*t+i)
                            prev_chord = pred[i].item()
                        if t==num_instance-1 and i+num_pad==n_timestep:
                            if start_time != time_unit*(n_timestep*t+i):
                                lines.append('%.3f %.3f %s\n' % (start_time, time_unit*(n_timestep*t+i), idx_to_chord[prev_chord]))
                            break
        
        # Save label file
        lab_save_path = os.path.join(lab_save_dir, base + ".lab")
        with open(lab_save_path, "w") as f:
            f.writelines(lines)
        # Comment out verbose logging
        # logger.info(f"Saved label file to: {lab_save_path}")
        
        # Save logits if enabled
        if args.save_logits and all_logits:
            # Ensure all_logits is not empty and all elements have compatible shapes
            if len(all_logits) == 0:
                logger.info(f"No logits to save for {audio_path}")
                continue
                
            try:
                # Concatenate all logits chunks along the time axis
                all_logits_array = np.concatenate(all_logits, axis=1)
                
                # Keep only this log statement about final logits shape
                logger.info(f"Logits shape: {all_logits_array.shape}")
                
                # Comment out verbose logging for dimension checking
                if all_logits_array.ndim == 3:
                    expected_num_chords = config.model['num_chords']
                    if all_logits_array.shape[2] != expected_num_chords:
                        # logger.info(f"Unexpected logits chord dimension: {all_logits_array.shape[2]}, expected {expected_num_chords}")
                        pass
                else:
                    # logger.info("Logits array is not 3D; skipping chord dimension check.")
                    pass
                
                # Save the logits
                logits_save_path = os.path.join(logits_save_dir, base + "_logits.npy")
                np.save(logits_save_path, all_logits_array)
                # logger.info(f"Saved logits to: {logits_save_path}")
            except Exception as e:
                logger.error(f"Error saving logits for {audio_path}: {str(e)}")
        
        processed_count += 1
    
    logger.info(f"Processing complete. Processed {processed_count} files, skipped {skipped_count} files.")
    
if __name__ == "__main__":
    main()
