#!/usr/bin/env python3
import sys
import torch
import torch.nn.functional as F
import logging
import speechbrain as sb
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.Accuracy import Accuracy
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main

"""Recipe for training a SSL wav2vec2.0 model

Authors
 * Guillermo Cambara 2021
"""

logger = logging.getLogger(__name__)

# Define training procedure
class SSL(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = self.modules.normalize(wavs, wav_lens)
        wavs = wavs.unsqueeze(-1)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Forward pass
        out = self.modules.wav2vec2(wavs, apply_mask=True, return_latent=False)

        feat, quant, mask_indices = out['feat'], out['quant'], out['mask_indices']

        feat_masked = feat[:, mask_indices, :]
        quant_feat = quant['x']
        num_vars = quant['num_vars']
        prob_perplexity = quant['prob_perplexity']

        feat_masked, quant_feat, target = self.modules.wav2vec2.arrange_distractors(feat_masked,
                                                                                    quant_feat,
                                                                                    max_distractors=50)

        return feat_masked, quant_feat, target, num_vars, prob_perplexity

    def compute_objectives(self, feat_masked, quant_feat, num_vars, prob_perplexity, target, batch, stage):
        """Computes the loss given predictions and targets."""

        ids = batch.id

        loss_dict = self.modules.wav2vec2.loss(feat_masked, quant_feat, target, num_vars, prob_perplexity)
        logits = loss_dict['logits']

        # Compute Accuracy using MetricStats
        self.acc_metric.append(
            ids, predict=logits, target=target, lengths=None
        )

        if loss_dict['contrastive_loss']:
            self.loss_metric_contrastive.append(loss_dict['contrastive_loss'])
        if loss_dict['diversity_loss']:
            self.loss_metric_diversity.append(loss_dict['diversity_loss'])

        return loss_dict

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        if self.auto_mix_prec:

            self.wav2vec_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                feat_masked, quant_feat, target, num_vars, prob_perplexity = self.compute_forward(batch, sb.Stage.TRAIN)
                loss_dict = self.compute_objectives(feat_masked, quant_feat, num_vars, prob_perplexity, target, batch, sb.Stage.TRAIN)
                loss = loss_dict['loss']

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.wav2vec_optimizer)

            if self.check_gradients(loss):
                self.scaler.step(self.wav2vec_optimizer)
                self.scaler.step(self.adam_optimizer)

            self.scaler.update()
        else:
            feat_masked, quant_feat, target, num_vars, prob_perplexity = self.compute_forward(batch, sb.Stage.TRAIN)

            loss_dict = self.compute_objectives(feat_masked, quant_feat, num_vars, prob_perplexity, target, batch, sb.Stage.TRAIN)
            loss = loss_dict['loss']
            loss.backward()

            if self.check_gradients(loss):
                self.wav2vec_optimizer.step()

            self.wav2vec_optimizer.zero_grad()
            self.hparams.lr_annealing_wav2vec(self.wav2vec_optimizer)

        self.hparams.tensorboard_train_logger.writer.add_scalar('loss/train_step', loss, 
                                                                self.hparams.lr_annealing_wav2vec.n_steps)
        self.hparams.tensorboard_train_logger.writer.add_scalar('loss_contrastive/train_step', loss_dict['contrastive_loss'], 
                                                                self.hparams.lr_annealing_wav2vec.n_steps)
        self.hparams.tensorboard_train_logger.writer.add_scalar('loss_diversity/train_step', loss_dict['diversity_loss'], 
                                                                self.hparams.lr_annealing_wav2vec.n_steps)
        self.hparams.tensorboard_train_logger.writer.add_scalar('prob_perplexity/train_step', prob_perplexity, 
                                                                self.hparams.lr_annealing_wav2vec.n_steps)
        self.hparams.tensorboard_train_logger.writer.add_scalar('lr/train_step', self.hparams.lr_annealing_wav2vec.current_lr, 
                                                                self.hparams.lr_annealing_wav2vec.n_steps)

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        feat_masked, quant_feat, target, num_vars, prob_perplexity = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss_dict = self.compute_objectives(feat_masked, quant_feat, num_vars, prob_perplexity, target, batch, stage=stage)
            loss = loss_dict['loss']
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""

        # Compute Accuracy using MetricStats
        # Define function taking (prediction, target, length) for eval
        def accuracy_value(predict, target, lengths):
            """Computes Accuracy"""
            predict = F.softmax(predict, dim=1) # logits to probabilities
            predict = torch.log(predict) # probs to log-probs
            predict = predict.unsqueeze(-1)
            nbr_correct, nbr_total = Accuracy(
                predict, target, lengths
            )
            acc = torch.tensor([nbr_correct / nbr_total])
            return acc

        self.acc_metric = sb.utils.metric_stats.MetricStats(
            metric=accuracy_value, n_jobs=1
        )

        self.loss_metric_contrastive = []
        self.loss_metric_diversity = []

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""

        # Compute/store important stats
        stage_stats = {"loss": stage_loss, "acc": self.acc_metric.summarize("average")}

        if self.loss_metric_contrastive != []:
            avg_loss_contrastive = float(sum(self.loss_metric_contrastive) / len(self.loss_metric_contrastive))
            stage_stats['loss_contrastive'] = avg_loss_contrastive
        if self.loss_metric_diversity != []:
            avg_loss_diversity = float(sum(self.loss_metric_diversity) / len(self.loss_metric_diversity))
            stage_stats['loss_diversity'] = avg_loss_diversity

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            if self.hparams.use_tensorboard:
                self.hparams.tensorboard_train_logger.log_stats(
                    stats_meta={"Epoch loaded": self.hparams.epoch_counter.current}, 
                    train_stats=self.train_stats,
                )

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": self.hparams.lr_annealing_wav2vec.current_lr,
                    "n_steps": self.hparams.lr_annealing_wav2vec.n_steps,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            if self.hparams.use_tensorboard:
                self.hparams.tensorboard_train_logger.log_stats(
                    stats_meta={
                    "epoch": epoch,
                    "lr": self.hparams.lr_annealing_wav2vec.current_lr,
                    "n_steps": self.hparams.lr_annealing_wav2vec.n_steps,
                    }, 
                    train_stats=self.train_stats,
                    valid_stats=stage_stats,
                )
            self.checkpointer.save_and_keep_only(
                meta={"acc": stage_stats["acc"]}, max_keys=["acc"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if self.hparams.use_tensorboard:
                self.hparams.tensorboard_train_logger.log_stats(
                    stats_meta={"Epoch loaded": self.hparams.epoch_counter.current}, 
                    train_stats=self.train_stats,
                    test_stats=stage_stats,
                )
            with open(self.hparams.acc_file, "w") as w:
                self.acc_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer"
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )

# Define custom data procedure
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )

    # We also sort the validation data so it is faster to validate
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig"],
    )
    return train_data, valid_data, test_data


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset preparation (parsing CommonVoice)
    from common_voice_prepare import prepare_common_voice  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs"]
        )

    # Due to DDP, we do the preparation ONLY on the main python process
    run_on_main(
        prepare_common_voice,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "train_tsv_file": hparams["train_tsv_file"],
            "dev_tsv_file": hparams["dev_tsv_file"],
            "test_tsv_file": hparams["test_tsv_file"],
            "accented_letters": hparams["accented_letters"],
            "language": hparams["language"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create the datasets objects
    train_data, valid_data, test_data = dataio_prepare(hparams)

    # Trainer initialization
    ssl_brain = SSL(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    ssl_brain.fit(
        ssl_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Test
    ssl_brain.hparams.acc_file = hparams["output_folder"] + "/acc_test.txt"
    ssl_brain.evaluate(
        test_data,
        max_key="acc",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
