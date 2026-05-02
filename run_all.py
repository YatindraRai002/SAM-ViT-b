import os
import yaml
from train_sam import train_sam
from sam_medical_analysis.experiments.task1_representations import run_task1
from sam_medical_analysis.experiments.task2_separation import run_task2
from sam_medical_analysis.experiments.task3_ablation import run_task3

def main():
    print("Starting SAM Medical Analysis & Training Pipeline...")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Ensure output directories exist
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(config['paths']['figures_dir'], exist_ok=True)
    os.makedirs(config['paths']['metrics_dir'], exist_ok=True)
    os.makedirs(config['paths']['embeddings_dir'], exist_ok=True)

    try:
        # Phase 1: Fine-tuning
        print("\n>>> PHASE 1: Fine-tuning SAM on Medical Data")
        train_sam(config)

        # Phase 2: Representation Analysis
        print("\n>>> PHASE 2: Intermediate Representation Analysis")
        run_task1(config)

        # Phase 3: Modality Separation
        print("\n>>> PHASE 3: Modality Separation Analysis")
        run_task2(config)

        # Phase 4: Ablation Study
        print("\n>>> PHASE 4: Attention Head Ablation")
        run_task3(config)

        print("\nPipeline completed successfully!")
    except Exception as e:
        print(f"\nPipeline failed during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
