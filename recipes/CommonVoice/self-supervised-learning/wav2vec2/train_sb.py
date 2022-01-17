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
class W2VBrain(sb.core.Brain):
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
            out = self.modules.wav2vec2.module(wavs, wav_lens=x_lens,
                                               normalize_wav=self.hparams.normalize_wav,
                                               output_norm=self.hparams.output_norm,
                                               apply_mask=True, 
                                               return_latent=False,
                                               penalize_latent=self.hparams.penalize_latent,
                                               do_final_projection=True,
                                               latent_grad_weight=self.hparams.latent_grad_weight)
        else:
            out = self.modules.wav2vec2(wavs, wav_lens=x_lens, 
                                        normalize_wav=self.hparams.normalize_wav,
                                        output_norm=self.hparams.output_norm,
                                        apply_mask=True, 
                                        return_latent=False,
                                        penalize_latent=self.hparams.penalize_latent,
                                        do_final_projection=True,
                                        latent_grad_weight=self.hparams.latent_grad_weight)

        feat, quant, mask_indices = out['feat'], out['quant'], out['mask_indices']
        latent_l2_loss = out['latent_l2']
        feat_masked = feat[:, mask_indices, :]

        if quant:
            pos_target = quant['x']
            num_vars = quant['num_vars']
            prob_perplexity = quant['prob_perplexity']
        else:
            pos_target = out['cont_target']
            num_vars, prob_perplexity = None, None

        if self.hparams.dynamic_distractor_sampling:
            max_num_negatives = int(len(mask_indices) * self.hparams.distractors_mask_percentage)
            if max_num_negatives > self.hparams.max_num_negatives:
                max_num_negatives = self.hparams.max_num_negatives
        else:
            max_num_negatives = self.hparams.max_num_negatives

        if self.distributed_launch:
            neg_target, _ = self.modules.wav2vec2.module.sample_negatives(pos_target,
                                                                pos_target.size(1),
                                                                padding_count=0,
                                                                num_negatives=max_num_negatives,
                                                                cross_sample_negatives=self.hparams.cross_sample_negatives
                                                                )
        else:
            neg_target, _ = self.modules.wav2vec2.sample_negatives(pos_target,
                                                    pos_target.size(1),
                                                    padding_count=0,
                                                    num_negatives=max_num_negatives,
                                                    cross_sample_negatives=self.hparams.cross_sample_negatives
                                                    )

        return feat_masked, pos_target, neg_target, num_vars, prob_perplexity, latent_l2_loss

    def compute_objectives(self, feat_masked, pos_target, neg_target, num_vars, prob_perplexity, 
                           latent_l2_loss, batch, stage):
        """Computes the loss given predictions and targets."""

        ids = batch.id

        if self.distributed_launch:
            loss_dict = self.modules.wav2vec2.module.loss(feat_masked, pos_target, neg_target, 
                                                num_vars, prob_perplexity, latent_l2_loss)
        else:
            loss_dict = self.modules.wav2vec2.loss(feat_masked, pos_target, neg_target, 
                                                num_vars, prob_perplexity, latent_l2_loss)
        logits = loss_dict['logits']
        target = loss_dict['target']

        # Compute Accuracy using MetricStats
        self.acc_metric.append(
            ids, predict=logits, target=target, lengths=None
        )

        if loss_dict['contrastive_loss']:
            self.loss_metric_contrastive.append(loss_dict['contrastive_loss'])
        if loss_dict['diversity_loss']:
            self.loss_metric_diversity.append(loss_dict['diversity_loss'])
        if loss_dict['latent_l2_loss']:
            self.loss_metric_diversity.append(loss_dict['latent_l2_loss'])

        return loss_dict

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""

        # Here we manage mixed precision
        if self.auto_mix_prec:
            with torch.cuda.amp.autocast():
                feat_masked, pos_target, neg_target, num_vars, prob_perplexity, latent_l2_loss = self.compute_forward(batch, sb.Stage.TRAIN)
                loss_dict = self.compute_objectives(feat_masked, pos_target, neg_target, num_vars,
                                                    prob_perplexity, latent_l2_loss,
                                                    batch, sb.Stage.TRAIN)
                loss = loss_dict['loss']

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
            feat_masked, pos_target, neg_target, num_vars, prob_perplexity, latent_l2_loss = self.compute_forward(batch, sb.Stage.TRAIN)

            loss_dict = self.compute_objectives(feat_masked, pos_target, neg_target, num_vars,
                                                prob_perplexity, latent_l2_loss,
                                                batch, sb.Stage.TRAIN)
            loss = loss_dict['loss']

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
                if loss_dict['contrastive_loss']:
                    self.hparams.tensorboard_train_logger.writer.add_scalar('loss_contrastive/train_step', loss_dict['contrastive_loss'].detach(), 
                                                                    self.hparams.noam_annealing.n_steps)
                if loss_dict['diversity_loss']:
                    self.hparams.tensorboard_train_logger.writer.add_scalar('loss_diversity/train_step', loss_dict['diversity_loss'].detach(), 
                                                                    self.hparams.noam_annealing.n_steps)

                if loss_dict['latent_l2_loss']:
                    self.hparams.tensorboard_train_logger.writer.add_scalar('loss_latent_l2/train_step', loss_dict['latent_l2_loss'].detach(), 
                                                                    self.hparams.noam_annealing.n_steps)

                if prob_perplexity:
                    self.hparams.tensorboard_train_logger.writer.add_scalar('prob_perplexity/train_step', prob_perplexity, 
                                                                    self.hparams.noam_annealing.n_steps)
                self.hparams.tensorboard_train_logger.writer.add_scalar('lr/train_step', self.hparams.noam_annealing.current_lr, 
                                                                self.hparams.noam_annealing.n_steps)

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        feat_masked, pos_target, neg_target, num_vars, prob_perplexity, latent_l2_loss = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss_dict = self.compute_objectives(feat_masked, pos_target, neg_target, num_vars, prob_perplexity, latent_l2_loss, batch, stage=stage)
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
            predict = predict.unsqueeze(0)
            target = target.unsqueeze(0)
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
        self.loss_metric_latent_l2 = []

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
        if self.loss_metric_latent_l2 != []:
            avg_loss_penalization = float(sum(self.loss_metric_latent_l2) / len(self.loss_metric_latent_l2))
            stage_stats['loss_latent_l2'] = avg_loss_diversity

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
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_shorter_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        train_data = train_data.filtered_sorted(
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_shorter_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

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

    return datasets[0], datasets[1], datasets[2]

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
            "language": hparams["language"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_data = dataio_prepare(hparams)

    # Trainer initialization
    w2v_brain = W2VBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        opt_class=hparams["opt_class"],
        checkpointer=hparams["checkpointer"],
    )

    # Adding objects to trainer.

    # Training
    w2v_brain.fit(
        w2v_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Test
    w2v_brain.hparams.acc_file = hparams["output_folder"] + "/acc_test.txt"
    w2v_brain.evaluate(
        test_data,
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
