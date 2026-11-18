"""Observation and acceptance checks around the unmodified experiment entry point."""

from collections import defaultdict
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import time

import torch


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def category(name):
    if name.startswith("ae_module."):
        return "ae"
    if "lora_a" in name or "lora_b" in name:
        return "lora"
    if name.startswith("heads.") or name.startswith("vit.heads."):
        return "head"
    return "backbone"


def snapshot(model):
    digests = defaultdict(hashlib.sha256)
    counts = defaultdict(lambda: {"total": 0, "trainable": 0})
    for name, param in model.named_parameters():
        group = category(name)
        tensor = param.detach().cpu().contiguous()
        digests[group].update(name.encode())
        digests[group].update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digests[group].update(tensor.numpy().tobytes())
        counts[group]["total"] += param.numel()
        counts[group]["trainable"] += param.numel() if param.requires_grad else 0
    return {key: {**counts[key], "sha256": digest.hexdigest()} for key, digest in digests.items()}


def unchanged(before, after, groups):
    for group in groups:
        if before.get(group, {}).get("sha256") != after.get(group, {}).get("sha256"):
            raise RuntimeError(f"Acceptance failed: {group} weights changed during a frozen stage")


def changed(before, after, group):
    if before.get(group, {}).get("sha256") == after.get(group, {}).get("sha256"):
        raise RuntimeError(f"Acceptance failed: expected {group} weights to train")


class Observation:
    """Callbacks record returned values and parameter hashes without altering training."""

    def __init__(self, record):
        self.record = record
        self.step = 0
        self.ga = {}
        self.dft = {}
        self.csa_count = 0
        self.ga_audited = False
        self.dft_audited = False
        self.restores = []

    def metrics(self, values):
        self.step += 1
        self.record.log_metrics(self.step, values)

    def replace(self, owner, name, callback):
        original = getattr(owner, name)
        self.restores.append((owner, name, original))

        @wraps(original)
        def observed(*args, **kwargs):
            arguments = inspect.signature(original).bind(*args, **kwargs)
            arguments.apply_defaults()
            return callback(original, arguments.arguments, args, kwargs)

        setattr(owner, name, observed)

    def install(self):
        import ae_psl.main_AE_PSL_global_evaluation as main
        from ae_psl.ae_trainers.implementations.classification.centralized_ae_trainer import CentralizedAETrainer
        from ae_psl.trainers.implementations.classification.sasl_trainer import SASLTrainer

        self.replace(main, "get_auto_encoder", self.alignment)
        self.replace(main, "get_distributed_dataloaders_from_datasets", self.partition)
        self.replace(CentralizedAETrainer, "train_epoch", self.ga_train)
        self.replace(CentralizedAETrainer, "test_epoch", self.ga_test)
        self.replace(SASLTrainer, "client_specific_alignment", self.csa)
        self.replace(SASLTrainer, "train_epoch", self.dft_train)
        self.replace(SASLTrainer, "test_epoch_global_evaluation", self.dft_test)
        self.replace(main, "save_experiment_results", self.epoch_saved)

    def restore(self):
        for owner, name, original in reversed(self.restores):
            setattr(owner, name, original)

    def alignment(self, original, bound, args, kwargs):
        base = bound["base_model"]
        before = snapshot(base)
        self.record.event("stage_start", stage="ga", backbone=before)
        started = time.monotonic()
        result = original(*args, **kwargs)
        after = snapshot(base)
        unchanged(before, after, before)
        if result[1].get("reused_AE"):
            raise RuntimeError("Acceptance failed: this attempt reused trained AE weights")
        self.record.event("ga_backbone_audit", before=before, after=after,
                          seconds=time.monotonic() - started, pretrained_ae_reused=False)
        return result

    def partition(self, original, bound, args, kwargs):
        loaders = original(*args, **kwargs)
        mapping = bound["train_dataset"].partitioner._partition_id_to_indices
        counts = {str(cid): len(indices) for cid, indices in mapping.items()}
        fingerprint = hashlib.sha256(json.dumps(
            {str(cid): [int(x) for x in indices] for cid, indices in mapping.items()},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        identity = {
            "client_counts": counts, "partition_sha256": fingerprint,
            "train_samples": len(bound["train_dataset"].train_ds),
            "test_samples": len(bound["test_dataset"]),
            "global_batches_per_epoch": len(loaders[0]),
            "client_batch_limit": bound["mini_batch_size"],
            "client_full_batches": {cid: n // bound["mini_batch_size"] for cid, n in counts.items()},
            "client_remainders": {cid: n % bound["mini_batch_size"] for cid, n in counts.items()},
            "drop_last": False, "sampler_shuffle": loaders[0].batch_sampler.shuffle,
            "evaluation_batch_size": loaders[2].batch_size,
        }
        if identity["train_samples"] != 50000 or identity["test_samples"] != 10000 or sum(counts.values()) != 50000:
            raise RuntimeError(f"Acceptance failed: CIFAR-100 split counts differ: {identity}")
        self.record.update_data({"downstream_partition": identity})
        self.record.event("partition", **identity)
        return loaders

    def ga_train(self, original, bound, args, kwargs):
        options = bound["kwargs"]
        model = options["auto_encoder"]
        before = snapshot(model) if not self.ga_audited else None
        started = time.monotonic()
        loss = original(*args, **kwargs)
        if before is not None:
            after = snapshot(model)
            if all(before[key]["sha256"] == after[key]["sha256"] for key in before):
                raise RuntimeError("Acceptance failed: GA did not update AE")
            self.record.event("ga_ae_audit", before=before, after=after)
            self.ga_audited = True
        self.ga[options["epoch_nr"]] = {
            "ga/epoch": options["epoch_nr"] + 1,
            "ga/train_mse": loss,
            "ga/train_seconds": time.monotonic() - started,
            "ga/learning_rate": options["optimizer"].param_groups[0]["lr"],
        }
        return loss

    def ga_test(self, original, bound, args, kwargs):
        options = bound["kwargs"]
        started = time.monotonic()
        loss = original(*args, **kwargs)
        if options["experiment_results"] is not None:
            self.metrics({**self.ga.pop(options["epoch_nr"]), "ga/test_mse": loss,
                          "ga/test_seconds": time.monotonic() - started})
        return loss

    def csa(self, original, bound, args, kwargs):
        clients, server = bound["client_models"], bound["server_model"]
        before = {str(cid): snapshot(model) for cid, model in clients.items()}
        before_server = snapshot(server)
        self.record.event("stage_start", stage="csa")
        started = time.monotonic()
        result = original(*args, **kwargs)
        after = {str(cid): snapshot(model) for cid, model in clients.items()}
        after_server = snapshot(server)
        for cid in before:
            unchanged(before[cid], after[cid], ("backbone", "lora", "head"))
            changed(before[cid], after[cid], "ae")
        unchanged(before_server, after_server, ("backbone", "lora", "head"))
        max_error = 0.0
        for name, actual in server.ae_module.patch_decoder.state_dict().items():
            expected = torch.zeros_like(actual)
            for client in clients.values():
                expected += client.ae_module.patch_decoder.state_dict()[name] / len(clients)
            error = float((expected - actual).abs().max())
            max_error = max(max_error, error)
            if not torch.allclose(expected, actual, rtol=1e-5, atol=1e-6):
                raise RuntimeError(f"Acceptance failed: server decoder is not the CSA mean: {name}")
        self.record.event("csa_weight_audit", clients_before=before, clients_after=after,
                          server_before=before_server, server_after=after_server,
                          decoder_mean_max_abs_error=max_error)
        self.csa_count += 1
        mse = bound["experiment_results"].client_batch_mse
        values = [value for per_epoch in mse.values() for value in per_epoch[str(bound["epoch_nr"])]]
        self.metrics({"csa/epoch": self.csa_count, "csa/mean_batch_mse": sum(values) / len(values),
                      "csa/seconds": time.monotonic() - started})
        return result

    def dft_train(self, original, bound, args, kwargs):
        clients, server = bound["client_models"], bound["server_model"]
        before = {str(cid): snapshot(model) for cid, model in clients.items()} if not self.dft_audited else None
        before_server = snapshot(server) if before is not None else None
        started = time.monotonic()
        result = original(*args, **kwargs)
        if before is not None:
            after = {str(cid): snapshot(model) for cid, model in clients.items()}
            after_server = snapshot(server)
            for cid in before:
                unchanged(before[cid], after[cid], ("backbone", "ae"))
                changed(before[cid], after[cid], "lora")
            unchanged(before_server, after_server, ("backbone", "ae"))
            changed(before_server, after_server, "lora")
            changed(before_server, after_server, "head")
            for state in [*after.values(), after_server]:
                if any(info["trainable"] for group, info in state.items() if group in {"backbone", "ae"}):
                    raise RuntimeError("Acceptance failed: DFT has trainable backbone/AE parameters")
            self.record.event("dft_weight_audit", clients_before=before, clients_after=after,
                              server_before=before_server, server_after=after_server)
            self.dft_audited = True
        self.dft[bound["epoch_nr"]] = {
            "dft/epoch": bound["epoch_nr"] + 1, "dft/train_loss": result[0],
            "dft/train_accuracy": result[1],
            "dft/train_samples": sum(result[2].values()),
            "dft/train_seconds": time.monotonic() - started,
        }
        if sum(result[2].values()) != 50000:
            raise RuntimeError(f"Acceptance failed: DFT did not consume all CIFAR training samples: {result[2]}")
        return result

    def dft_test(self, original, bound, args, kwargs):
        if bound["compression_during_global_eval"]:
            raise RuntimeError("Acceptance failed: global evaluation must bypass the AE")
        started = time.monotonic()
        result = original(*args, **kwargs)
        if bound["experiment_results"] is not None:
            self.dft[bound["epoch_nr"]].update({
                "dft/test_loss": result[0], "dft/test_accuracy": result[1],
                "dft/test_seconds": time.monotonic() - started,
            })
        else:
            self.record.event("final_evaluation", loss=result[0], accuracy=result[1],
                              samples=len(bound["dataloader"].dataset), bypass_ae=True)
        return result

    def epoch_saved(self, original, bound, args, kwargs):
        result = original(*args, **kwargs)
        epoch = bound["experiment_results"].last_completed_epoch
        self.metrics(self.dft.pop(epoch))
        return result
