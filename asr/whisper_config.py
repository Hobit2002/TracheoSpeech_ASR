import os
import torch

### SET GLOBAL VARIABLES
ANNOTATIONS_PATH = os.path.join(os.getcwd(),"data/TracheoSpeech/metadata.csv")
PATIENT_AUDIO_DIR = os.path.join(os.getcwd(),"data/TracheoSpeech/sessions")
COMPRESSED_AUDIO_DIR = os.path.join(os.getcwd(),"data/compressed")
PATIENT_SEGMENTED_AUDIO_DIR = os.path.join(os.getcwd(),"data/TracheoSpeech/samples")
REGULAR_SPEECH_DIR = os.path.join(os.getcwd(),"data/regular_speech")
QUASI_TRACHEO_DIR = os.path.join(os.getcwd(),"data/quasi_tracheo")
SAMPLE_RATE = 16000
TRAIN_RATE = 0.8
AUDIO_MAX_LENGTH = 480000
TEXT_MAX_LENGTH = 120
SEED = 42
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
LOG_OUTPUT_DIR = "logs"
CHECKPOINT_OUTPUT_DIR = "artifacts"
WANDB_PROJECT = "TracheoSpeech_ASR"

### TRAINING CONFIGURATIONS
# Abstract Configurations
class GeneralConfig:
  weight_decay = 0.01
  adam_epsilon = 1e-8
  warmup_steps = 11
  num_worker = 11
  num_train_epochs = 14
  gradient_accumulation_steps = 1
  sample_rate = SAMPLE_RATE
  lang = "cs"
  vocab_size = 51865
  embedding_dim = 512
  kd_temperature = 2.
  # Catastrophic-forgetting mitigation
  # ... by KD from the old model
  kl_regularization = False
  kl_reg_lambda = 1e-2
  # ... by freezing the first encoder layers
  freeze_encoder_convs = False
  freeze_n_encoder_blocks = 0
  # ... prompt tuning
  prompt_tuning = False
  prompt_lr = 1e-2

# Abstarct Configurations: Baseline vs Adapted
class BaselineConfig(GeneralConfig):
  soft_targets = False

class AdaptedConfig(GeneralConfig):
  soft_targets = True
  kd_lambda = 1e-3

class AdaptedPatientDeviceConfig(GeneralConfig):
  soft_targets = False
  kl_regularization = True
  num_train_epochs = 1
  freeze_encoder_convs = True
  freeze_n_encoder_blocks = 0
  prompt_tuning = True

# Abstarct Configurations: Dataset to train on
class RegularSpeechConfig(GeneralConfig):
  pretrain_on_cs = REGULAR_SPEECH_DIR
  whisper_checkpoint = None

class QuasiTracheoSpeechConfig(GeneralConfig):
  pretrain_on_cs = QUASI_TRACHEO_DIR

class PatientSpeechConfig(GeneralConfig):
  pretrain_on_cs = False

# Abstract Configurations: Model Size
class TinyConfig(GeneralConfig):
  model_name = "tiny"
  batch_size = 8
  learning_rate = 1e-4

class BaseConfig(GeneralConfig):
  model_name = "base"
  batch_size = 4
  learning_rate = 3e-5

class SmallConfig(GeneralConfig):
  model_name = "small"
  batch_size = 4
  learning_rate = 1e-5

# Concrete Configurations
# Tiny Configurations
class TinyBaselineRegularConfig(TinyConfig, BaselineConfig, RegularSpeechConfig):
  train_name = "tiny_baseline_regular"
  learning_rate = 5e-4

class TinyBaselinePatientConfig(TinyConfig, BaselineConfig, PatientSpeechConfig):
  train_name = "tiny_baseline_patient"
  whisper_checkpoint = "tiny_baseline_regular"

class TinyAdaptedRegularConfig(TinyConfig, AdaptedConfig, RegularSpeechConfig):
  train_name = "tiny_adapted_regular"

class TinyAdaptedQuasiTracheoConfig(TinyConfig, AdaptedConfig, QuasiTracheoSpeechConfig):
  train_name = "tiny_adapted_quasi_tracheo"
  whisper_checkpoint = "tiny_adapted_regular"
  kd_lambda = 5e-2

class TinyAdaptedPatientConfig(TinyConfig, AdaptedConfig, PatientSpeechConfig):
  train_name = "tiny_adapted_patient"
  whisper_checkpoint = "tiny_adapted_quasi_tracheo"

# Base Configurations
class BaseBaselineRegularConfig(BaseConfig, BaselineConfig, RegularSpeechConfig):
  train_name = "base_baseline_regular"

class BaseBaselinePatientConfig(BaseConfig, BaselineConfig, PatientSpeechConfig):
  train_name = "base_baseline_patient"
  whisper_checkpoint = "base_baseline_regular"

class BaseAdaptedRegularConfig(BaseConfig, AdaptedConfig, RegularSpeechConfig):
  train_name = "base_adapted_regular"

class BaseAdaptedQuasiTracheoConfig(BaseConfig, AdaptedConfig, QuasiTracheoSpeechConfig):
  train_name = "base_adapted_quasi_tracheo"
  whisper_checkpoint = "base_adapted_regular"
  kd_lambda = 5e-2

class BaseAdaptedPatientConfig(BaseConfig, AdaptedConfig, PatientSpeechConfig):
  train_name = "base_adapted_quasi_patient"
  whisper_checkpoint = "base_adapted_quasi_tracheo"

class BaseAdaptedPatientDeviceConfig(BaseConfig, AdaptedPatientDeviceConfig, PatientSpeechConfig):
  train_name = "base_adapted_patient_device"
  whisper_checkpoint = "base_adapted_patient"
  learning_rate = 1e-5

# Small Configurations
class SmallAdaptedRegularConfig(SmallConfig, BaselineConfig, RegularSpeechConfig):
  learning_rate = 3e-5
  train_name = "small_adapted_regular"

class SmallAdaptedQuasiTracheoConfig(SmallConfig, BaselineConfig, QuasiTracheoSpeechConfig):
  train_name = "small_adapted_quasi_tracheo"
  whisper_checkpoint = "small_adapted_regular"

class SmallAdaptedPatientConfig(SmallConfig, AdaptedConfig, PatientSpeechConfig):
  train_name = "small_adapted_patient_tracheo"
  whisper_checkpoint = "small_adapted_quasi_tracheo"


