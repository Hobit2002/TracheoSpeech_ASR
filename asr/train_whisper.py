### IMPORT LIBRARIES
# Public libraries
from pathlib import Path
from tqdm import tqdm
import argparse,multiprocessing,platform, time
from torch import cuda
# Pytorch lightning
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
# Local
from whisper_config import *
from whisper_datasets import *
from whisper_module import WhisperModelModule
import wandb

# SEED
if __name__ == "__main__":
    seed_everything(SEED, workers=True)

    # Get Configuration
    parser = argparse.ArgumentParser(
                        prog='train_whisper',
                        description='This program trains Whisper of the specified configuration',
                        )
    parser.add_argument('model_configuration') 
    args = parser.parse_args()

    config_dict = {
        "tiny_baseline_regular":TinyBaselineRegularConfig, "tiny_baseline_patient":TinyBaselinePatientConfig,
        "tiny_adapted_regular":TinyAdaptedRegularConfig, "tiny_adapted_quasi_tracheo":TinyAdaptedQuasiTracheoConfig,
        "tiny_adapted_patient":TinyAdaptedPatientConfig, "base_baseline_regular":BaseBaselineRegularConfig,
        "base_baseline_patient":BaseBaselinePatientConfig, "base_adapted_regular":BaseAdaptedRegularConfig,
        "base_adapted_quasi_tracheo":BaseAdaptedQuasiTracheoConfig,"base_adapted_patient": BaseAdaptedPatientConfig,
        "small_adapted_regular": SmallAdaptedRegularConfig, "small_adapted_quasi_tracheo":SmallAdaptedQuasiTracheoConfig,
        "small_adapted_patient":SmallAdaptedPatientConfig

    }

    cfg = config_dict[args.model_configuration]

    ### CREATE LIST OF PAIRED FILES
    train_audio_transcript_pair_list, eval_audio_transcript_pair_list = load_audio_and_annotations()

    ### DEFINE THE MODULE
    ### TRAIN THE WHISPER
    Path(LOG_OUTPUT_DIR).mkdir(exist_ok=True)
    Path(CHECKPOINT_OUTPUT_DIR).mkdir(exist_ok=True)

    # Initialize wandb logging
    wblogger = WandbLogger(
        project="whisper",  # The name of your Wandb project
        name=cfg.train_name,              # The name of your experiment
        save_dir=LOG_OUTPUT_DIR,      # Optional, if you want to save the logs locally
    )


    wandb.init(project=WANDB_PROJECT, name=cfg.train_name)


    # Prepare the checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{CHECKPOINT_OUTPUT_DIR}/checkpoint",
        filename=cfg.train_name,
        save_top_k=1,
        monitor="val/wer_epoch",
        mode="min"
    )

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    model = WhisperModelModule(cfg, cfg.lang, train_audio_transcript_pair_list, eval_audio_transcript_pair_list)

    # Prepare the data loaders

    train_loader, val_loader = model.train_dataloader(), model.val_dataloader()
    
    # Loop loaders to precompute the pickled data
    print("Precomputing training data")
    for _ in tqdm(model.train_dataloader(shuffle=False)):pass

    # Train model
    trainer = Trainer(
        precision='bf16',
        accelerator=DEVICE,
        devices=1,   # Forces single GPU in Lightning's eyes
        strategy="auto",  # Prevents DataParallel or DDP
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=wblogger,
        callbacks=callback_list
    )

    multiprocessing.freeze_support() 
    trainer.fit(model, train_loader, val_loader)

    wandb.finish()