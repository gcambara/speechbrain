#!/usr/bin/env python3

import sys
import torch
import torch.nn.functional as F
import logging
import speechbrain as sb
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.Accuracy import Accuracy
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.sampler import DynamicBatchSampler

"""Recipe for training a SSL wav2vec2.0 model

Authors
 * Guillermo Cambara 2021
"""

logger = logging.getLogger(__name__)


# Define training procedure
class PatchiesBrain(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the w2v2 loss."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.unsqueeze(-1)

        if self.hparams.dont_mask_padding:
            x_lens = wav_lens
        else:
            x_lens = None

        if self.distributed_launch:
            out = self.modules.patchies.module(wavs, wav_lens=x_lens,
                                               normalize_wav=self.hparams.normalize_wav,
                                               output_norm=self.hparams.output_norm,
                                               apply_mask=True, stage='train')
        else:
            out = self.modules.patchies(wavs, wav_lens=x_lens, 
                                        normalize_wav=self.hparams.normalize_wav,
                                        output_norm=self.hparams.output_norm,
                                        apply_mask=True, stage='train')

        feat, mask_indices, not_mask_indices, target_patches = out['feat'], out['mask_indices'], out['not_mask_indices'], out['target_patches']

        if self.hparams.upsampling_factor > 1:
            mask_indices = self.modules.patchies.feat_masker.upsample_mask_indices(mask_indices, self.hparams.upsampling_factor)

        pred_masked = self.modules.patchies.feat_masker.get_masked_features(feat, mask_indices)
        target_masked = self.modules.patchies.feat_masker.get_masked_features(target_patches, mask_indices)

        return feat, pred_masked, target_masked

    def compute_objectives(self, pred_masked, target_masked, batch, stage):
        """Computes the loss given predictions and targets."""

        ids = batch.id
        
        if self.distributed_launch:
            avg_loss, patch_losses = self.modules.patchies.module.loss([pred_masked], [target_masked])
        else:
            avg_loss, patch_losses = self.modules.patchies.loss([pred_masked], [target_masked])

        return avg_loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""

        # Here we manage mixed precision
        if self.auto_mix_prec:
            with torch.cuda.amp.autocast():
                feat, pred_masked, target_masked = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(pred_masked, target_masked, batch, sb.Stage.TRAIN)

            # normalize the loss by gradient_accumulation step
            self.scaler.scale(
                loss / self.hparams.gradient_accumulation
            ).backward()

            if self.step % self.hparams.gradient_accumulation == 0:
                # gradient clipping & early stop if loss is not fini
                self.check_gradients(loss)

                self.scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # anneal lr every update
                self.hparams.noam_annealing(self.optimizer)
        else:
            feat, pred_masked, target_masked = self.compute_forward(batch, sb.Stage.TRAIN)

            loss = self.compute_objectives(pred_masked, target_masked, batch, sb.Stage.TRAIN)

            # normalize the loss by gradient_accumulation step
            (loss / self.hparams.gradient_accumulation).backward()

            if self.step % self.hparams.gradient_accumulation == 0:
                # gradient clipping & early stop if loss is not fini
                self.check_gradients(loss)

                self.optimizer.step()
                self.optimizer.zero_grad()

                # anneal lr every update
                self.hparams.noam_annealing(self.optimizer)

                self.hparams.tensorboard_train_logger.writer.add_scalar('loss/train_step', loss.detach(), 
                                                                self.hparams.noam_annealing.n_steps)

                self.hparams.tensorboard_train_logger.writer.add_scalar('lr/train_step', self.hparams.noam_annealing.current_lr, 
                                                                self.hparams.noam_annealing.n_steps)

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        feat, pred_masked, target_masked = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(pred_masked, target_masked, batch, stage=stage)
        return loss.detach()
    # def on_stage_start(self, stage, epoch):
    #     """Gets called at the beginning of each epoch"""
    #     # # Compute Accuracy using MetricStats
    #     # # Define function taking (prediction, target, length) for eval
    #     # def accuracy_value(predict, target, lengths):
    #     #     """Computes Accuracy"""
    #     #     predict = F.softmax(predict, dim=1) # logits to probabilities
    #     #     predict = torch.log(predict) # probs to log-probs
    #     #     predict = predict.unsqueeze(0)
    #     #     target = target.unsqueeze(0)
    #     #     nbr_correct, nbr_total = Accuracy(
    #     #         predict, target, lengths
    #     #     )
    #     #     acc = torch.tensor([nbr_correct / nbr_total])
    #     #     return acc

    #     # self.acc_metric = sb.utils.metric_stats.MetricStats(
    #     #     metric=accuracy_value, n_jobs=1
    #     # )

    #     # self.loss_metric_contrastive = []
    #     # self.loss_metric_diversity = []
    #     # self.loss_metric_latent_l2 = []

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""

        # Compute/store important stats
        #stage_stats = {"loss": stage_loss, "acc": self.acc_metric.summarize("average")}
        stage_stats = {"loss": stage_loss}

        # if self.loss_metric_contrastive != []:
        #     avg_loss_contrastive = float(sum(self.loss_metric_contrastive) / len(self.loss_metric_contrastive))
        #     stage_stats['loss_contrastive'] = avg_loss_contrastive
        # if self.loss_metric_diversity != []:
        #     avg_loss_diversity = float(sum(self.loss_metric_diversity) / len(self.loss_metric_diversity))
        #     stage_stats['loss_diversity'] = avg_loss_diversity
        # if self.loss_metric_latent_l2 != []:
        #     avg_loss_penalization = float(sum(self.loss_metric_latent_l2) / len(self.loss_metric_latent_l2))
        #     stage_stats['loss_latent_l2'] = avg_loss_diversity

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
                    "lr": self.hparams.noam_annealing.current_lr,
                    "n_steps": self.hparams.noam_annealing.n_steps,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            if self.hparams.use_tensorboard:
                self.hparams.tensorboard_train_logger.log_stats(
                    stats_meta={
                    "epoch": epoch,
                    "lr": self.hparams.noam_annealing.current_lr,
                    "n_steps": self.hparams.noam_annealing.n_steps,
                    }, 
                    train_stats=self.train_stats,
                    valid_stats=stage_stats,
                )
            # self.checkpointer.save_and_keep_only(
            #     meta={"acc": stage_stats["acc"]}, max_keys=["acc"],
            # )
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
            key_min_value={"duration": hparams["avoid_if_shorter_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_shorter_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        train_data = train_data.filtered_sorted(
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_shorter_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # We also sort the validation data so it is faster to validate
    #test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data]

    # defining tokenizer and loading it

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        if info.num_channels > 1:
            sig = torch.mean(sig, dim=1)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)

        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig"],
    )

    if hparams["dynamic_batching"]:
        dynamic_hparams = hparams["dynamic_batch_sampler"]
        for index, dataset in enumerate(datasets):
            batch_sampler = DynamicBatchSampler(
                dataset,
                dynamic_hparams["max_batch_len"],
                dynamic_hparams["left_bucket_len"],
                bucket_length_multiplier=dynamic_hparams["multiplier"],
                length_func=lambda x: x["duration"],
                shuffle=dynamic_hparams["shuffle_ex"],
            )

            datasets[index] = SaveableDataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=PaddedBatch,
            )

    return datasets[0], datasets[1]

if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs"]
        )
    # Due to DDP, we do the preparation ONLY on the main python process
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data = dataio_prepare(hparams)

    # Trainer initialization
    patchies_brain = PatchiesBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        opt_class=hparams["opt_class"],
        checkpointer=hparams["checkpointer"],
    )

    # Adding objects to trainer.

    # Training
    patchies_brain.fit(
        patchies_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
#    patchies_brain.hparams.acc_file = hparams["output_folder"] + "/acc_test.txt"
#    patchies_brain.evaluate(
#        test_data,
#        min_key="loss",
#        test_loader_kwargs=hparams["test_dataloader_opts"],
#    )
