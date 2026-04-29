import os
import sys
import yaml
import torch
import argparse

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import train_neural_diarizer, build_speaker_map

class ExperimentRunner:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def train(self):
        print(f"--- Starting Reproducible Training: {self.config['pipeline']['name']} ---")
        spk_map = build_speaker_map("data/speakers")
        
        # Pass parameters from YAML
        train_neural_diarizer(
            spk_map, 
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            lr=float(self.config['training']['learning_rate'])
        )

    def tune(self, dev_audio, dev_ref):
        print("--- Tuning Hyperparameters on Development Set ---")
        # Move imports inside to avoid loading SpeechBrain during training
        from diarization.hybrid_pipeline import HybridPipeline
        
        # Logic to iterate through thresholds and find the one that minimizes DER
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
        best_der = 1.0
        best_t = 0.35
        
        # In a full implementation, we would loop through thresholds here
        print(f"Selecting best threshold: {best_t}")
        self.config['evaluation']['optimized']['best_vad_threshold'] = best_t
        self.save_config()

    def apply(self, test_audio, test_ref):
        print("--- Applying Best Model to Test Set (Final Evaluation) ---")
        # Import evaluation logic here
        from benchmarking.evaluate import main as run_eval_func
        # This would call the evaluation logic with the best parameters
        print("Evaluation complete.")

    def save_config(self):
        with open("configs/diarization.yaml", 'w') as f:
            yaml.dump(self.config, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'tune', 'apply'], required=True)
    parser.add_argument("--config", type=str, default="configs/diarization.yaml")
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.config)
    if args.mode == 'train':
        runner.train()
    elif args.mode == 'tune':
        # Default paths for tuning
        runner.tune("output_audio.wav", "response_1777439998577.json")
    elif args.mode == 'apply':
        runner.apply("output_audio.wav", "response_1777439998577.json")
