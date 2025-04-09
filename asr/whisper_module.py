import whisper, evaluate, torch, time
import torch.nn as nn
# Transformers
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
# Pytorch lightning
from pytorch_lightning import LightningModule
## Local
from whisper_config import *
from whisper_datasets import *
from torch.utils.checkpoint import checkpoint
import requests
from augmentations import mixstyle
from io import BytesIO
from asr.mlm_teacher import BiLSTMModel
import torch.nn.functional as F
import random

# Different models, different devices



def load_model_from_url(url, checkpoint_dir="artifacts/checkpoint"):
    """
    Load a PyTorch model checkpoint from a URL, checking for a local copy first.

    Args:
        url (str): The URL of the checkpoint file.
        checkpoint_dir (str): Directory where checkpoints are stored locally.

    Returns:
        dict: Loaded checkpoint dictionary.
    """
    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Extract the filename from the URL
    filename = url.split("/")[-1]
    local_path = os.path.join(checkpoint_dir, filename)

    # Check if the file already exists locally
    if os.path.exists(local_path):
        print(f"Checkpoint found locally at {local_path}. Loading...")
        checkpoint = torch.load(local_path)
    else:
        # Download the file from the URL
        print(f"Downloading checkpoint from {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an error if the request fails

        # Save the checkpoint locally
        with open(local_path, "wb") as f:
            f.write(response.content)
        print(f"Checkpoint saved to {local_path}.")

        # Load the checkpoint
        checkpoint = torch.load(local_path)

    return checkpoint

class WhisperModelModule(LightningModule):
    
    def __init__(self, cfg:GeneralConfig, lang="cs", train_dataset=[], eval_dataset=[]) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.model = whisper.load_model(self.cfg.model_name)
        self.cfg = cfg
        # Prepare tokenizer
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language="cs", task=self.options.task)
        # Prepare devices
        if self.cfg.model_name != "tiny":
            if DEVICE == "cpu" or torch.cuda.device_count() < 2: 
                print("You need at least 2 GPUs to run the model of size",self.cfg.model_name)
                self.device0 = torch.device("cuda:1")
                self.device1 = torch.device("cuda:0")

        # Prepare metrics
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        # Prepare datasets
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset
        self.debug = False
        self.save_predictions = False
        
        # Loading whisper model
        # For Whisper training:
        if self.cfg.whisper_checkpoint: 
            checkpoint = torch.load(self.cfg.whisper_checkpoint)
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for k,v in state_dict.items(): new_state_dict[k.replace("model.","")] = v
            self.model.load_state_dict(new_state_dict, strict = False)
            self.model.train()
        # Distribute among GPUs
        if self.cfg.model_name != "tiny":
            self.model.encoder.to(self.device0)
            self.model.decoder.to(self.device1)

        # Load soft-labelling LSTM
        if self.cfg.soft_targets:
            self.lstm_teacher = BiLSTMModel(self.cfg.vocab_size, self.cfg.embedding_dim, self.tokenizer)
            self.lstm_teacher.load_state_dict(torch.load(os.path.join(os.getcwd(),"soft_target_lstm_l6_data_large.pth")))
            self.kd_loss= nn.KLDivLoss(log_target=True, reduction="none")
            self.lstm_teacher.eval()
            if self.cfg.model_name != "tiny": self.lstm_teacher.to(self.device1)

    def forward(self, x):
        # Set devices again
        if self.cfg.model_name != "tiny": 
            self.model.encoder.to(self.device0)
            self.model.decoder.to(self.device1)
            if self.cfg.soft_targets: self.lstm_teacher.to(self.device1)
        # Parse input
        input_ids = x["input_ids"]
        labels = x["labels"].long()
        dec_input_ids = x["dec_input_ids"].long()
        # Predict
        if self.cfg.model_name != "tiny": input_ids = input_ids.to(self.device0)  # Move inputs to encoder GPU
        encoder_output = self.model.encoder(input_ids)
        if self.cfg.model_name != "tiny": 
            encoder_output = encoder_output.to(self.device1)  # Move output to decoder GPU
            dec_input_ids = dec_input_ids.to(self.device1)
        decoder_output = self.model.decoder(dec_input_ids, encoder_output)
        return decoder_output
    
    def training_step(self, batch, batch_id):
        if self.debug: print(f"Training step starts. Batch {batch_id}")
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        if self.cfg.model_name != "tiny": input_ids = input_ids.to(self.device0)
        audio_features = checkpoint(self.model.encoder, input_ids, use_reentrant=False, preserve_rng_state=False)

        if self.cfg.model_name != "tiny": 
            dec_input_ids = dec_input_ids.to(self.device1)
            audio_features = audio_features.to(self.device1)
            labels = labels.to(self.device1)
        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        
        # Apply soft targets
        if self.cfg.soft_targets:
            
            with torch.no_grad():
                if self.cfg.model_name != "tiny": out = out.to(self.device1)
                kd_targets = self.lstm_teacher.get_soft_targets(dec_input_ids, labels, out)
            
            if self.cfg.soft_target_whisper:
                kd_targets = kd_targets + self.cfg.soft_target_whisper_weight * batch["whisper_targets_soft"]

            with torch.amp.autocast("cuda"):
                y_hat_soft = F.log_softmax(out / self.cfg.kd_temperature, dim=-1)
                soft_targets = F.log_softmax(kd_targets, dim=-1)

            kd_loss = self.kd_loss(y_hat_soft.view(-1, y_hat_soft.size(-1)), soft_targets.view(-1, soft_targets.size(-1))).mean()
            kd_loss = kd_loss  * (self.cfg.kd_temperature ** 2)
            self.log("train/kd_loss", kd_loss, on_step=True, prog_bar=True, logger=True)
            loss = self.cfg.kd_lambda * loss + (1 - self.cfg.kd_lambda) * kd_loss

        # Apply layer-wise lr decay
        return loss

    def validation_step(self, batch, batch_id):
        # Set devices again
        if self.cfg.model_name != "tiny": 
            self.model.encoder.to(self.device0)
            self.model.decoder.to(self.device1)
            if self.cfg.soft_targets:self.lstm_teacher.to(self.device1)
        # Let's go!
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        if self.cfg.model_name != "tiny": input_ids = input_ids.to(self.device0)
        audio_features = self.model.encoder(input_ids)
        if self.cfg.model_name != "tiny": 
            dec_input_ids = dec_input_ids.to(self.device1)
            audio_features = audio_features.to(self.device1)
            labels = labels.to(self.device1)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        # Main evaluation loop
        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
            
        # Do the greedy decision
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode(o))
            l_list.append(self.tokenizer.decode(l))
        # Log the results
        if batch_id and not batch_id%50:
            log_data = []
            for i, (refs, outs) in enumerate(zip(l_list, o_list)):
                print(f"Sample {i}:")
                print(f"Reference: {refs}")
                print(f"Outcomes: {outs}")
                log_data.append({"Reference":refs, "Outcomes":outs})


        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)


        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)
        if not self.cfg.pretrain_on_cs and batch_id < 40:
            self.log("val/real_world_loss", loss, on_step=True, prog_bar=True, logger=True)
            self.log("val/real_world_cer", cer, on_step=True, prog_bar=True, logger=True)
            self.log("val/real_world_wer", wer, on_step=True, prog_bar=True, logger=True)

        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        optimizer_grouped_parameters.extend([
            {
                # Parameters for the rest of the model with standard learning rate
                "params": [p for n, p in self.model.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
                "lr":self.cfg.learning_rate
            },
            {
                # Parameters for the rest of the model without weight decay
                "params": [p for n, p in self.model.named_parameters()
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr":self.cfg.learning_rate
            },
        ])

        # Use AdamW optimizer
        optimizer = AdamW(optimizer_grouped_parameters,
                        lr=self.cfg.learning_rate,  # Default learning rate for the main model
                        eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        # Scheduler configuration
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def adjust_learning_rates(self):
        # Loop through all optimizer groups
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            # Calculate the decay based on layer depth (here, simply using group index)
            decay_factor = self.cfg.layer_decay ** ((170 - i)//6)
            # Update the learning rate for this parameter group
            param_group['lr'] = param_group['lr'] * decay_factor

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )
            print("Total update steps:", self.t_total)

    def train_dataloader(self, shuffle = True):
        if self.cfg.pretrain_on_cs:
            dataset = CommonVoiceDataset(self.cfg.pretrain_on_csv,'train.tsv',self.tokenizer) 
            self.__train_dataset = dataset
        else: dataset = JasmiSpeechDataset(self.__train_dataset, self.tokenizer, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset,
                          batch_size=self.cfg.batch_size,
                          drop_last=True, shuffle=shuffle, num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

    def val_dataloader(self):
        if self.cfg.pretrain_on_cs: dataset = CommonVoiceDataset(self.cfg.pretrain_on_csv,'test.tsv',self.tokenizer)
        else: dataset = JasmiSpeechDataset(self.__eval_dataset, self.tokenizer, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset,
                          batch_size=self.cfg.batch_size,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )
    
