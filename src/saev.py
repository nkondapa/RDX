"""
Neural network architectures for sparse autoencoders.
"""

import io
import json
import logging
import os
import typing

import einops
import torch
from torch import Tensor
import tqdm


# class Loss(typing.NamedTuple):
#     """The composite loss terms for an autoencoder training batch."""
#
#     mse: Float[Tensor, ""]
#     """Reconstruction loss (mean squared error)."""
#     sparsity: Float[Tensor, ""]
#     """Sparsity loss, typically lambda * L1."""
#     ghost_grad: Float[Tensor, ""]
#     """Ghost gradient loss, if any."""
#     l0: Float[Tensor, ""]
#     """L0 magnitude of hidden activations."""
#     l1: Float[Tensor, ""]
#     """L1 magnitude of hidden activations."""
#
#     def loss(self):
#         """Total loss."""
#         return self.mse + self.sparsity + self.ghost_grad


class SparseAutoencoder(torch.nn.Module):
    """
    Sparse auto-encoder (SAE) using L1 sparsity penalty.
    """

    def __init__(self, input_params):
        super().__init__()

        self.params = input_params
        d_input = input_params['d_input']
        d_sae = input_params['d_sae']
        seed = input_params['seed']

        self.W_enc = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_input, d_sae))
        )
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))

        self.W_dec = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_sae, d_input))
        )
        self.b_dec = torch.nn.Parameter(torch.zeros(d_input))
        self.logger = logging.getLogger(f"sae(seed={seed})")
        self.dataset = None

    def forward(self, x):
        """
        Given x, calculates the reconstructed x_hat, the intermediate activations f_x, and the loss.

        Arguments:
            x: a batch of ViT activations.
        """

        # Remove encoder bias as per Anthropic
        h_pre = (
                torch.einsum(
                    "bd, ds -> bs", x - self.b_dec, self.W_enc
                )
                + self.b_enc
        )
        f_x = torch.nn.functional.relu(h_pre)
        x_hat = self.decode(f_x)

        # Some values of x and x_hat can be very large. We can calculate a safe MSE
        mse_loss = safe_mse(x_hat, x)

        mse_loss = mse_loss.mean()
        l0 = (f_x > 0).float().sum(axis=1).mean(axis=0)
        l1 = f_x.sum(axis=1).mean(axis=0)
        sparsity_loss = self.params['sparsity_coeff'] * l1
        # Ghost loss is included for backwards compatibility.
        ghost_loss = torch.zeros_like(mse_loss)

        return x_hat, f_x, {"mse_loss": mse_loss, "sparsity_loss": sparsity_loss, "ghost_loss": ghost_loss, "l0": l0,
                            "l1": l1, "loss": mse_loss + sparsity_loss + ghost_loss}

    def decode(self, f_x):
        x_hat = (
            # torch.einsum(f_x, self.W_dec, "... d_sae, d_sae d_vit -> ... d_vit")
                torch.einsum("bs, sd -> bd", f_x, self.W_dec)
                + self.b_dec
        )
        return x_hat

    @torch.no_grad()
    def init_b_dec(self, vit_acts):
        # if self.params.get('n_reinit_samples', 0) <= 0:
        #     self.logger.info("Skipping init_b_dec.")
        #     return
        previous_b_dec = self.b_dec.clone().cpu()
        # vit_acts = vit_acts[: self.params['n_reinit_samples']]
        # assert len(vit_acts) == self.params['n_reinit_samples']
        mean = vit_acts.mean(axis=0)
        previous_distances = torch.norm(vit_acts - previous_b_dec, dim=-1)
        distances = torch.norm(vit_acts - mean, dim=-1)
        # self.logger.info(
        #     "Prev dist: %.3f; new dist: %.3f",
        #     previous_distances.median(axis=0).values.mean().item(),
        #     distances.median(axis=0).values.mean().item(),
        # )
        self.b_dec.data = mean.to(self.b_dec.dtype).to(self.b_dec.device)

    @torch.no_grad()
    def normalize_w_dec(self):
        """
        Set W_dec to unit-norm columns.
        """
        if self.params.get('normalize_w_dec', False):
            # print("Normalizing W_dec")
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_parallel_grads(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_vit) shape
        """
        if not self.params.get("remove_parallel_grads", False):
            return

        # print("Removing parallel grads")

        parallel_component = torch.einsum(

            "sd, sd -> s", self.W_dec.grad,
            self.W_dec.data
        )

        self.W_dec.grad -= torch.einsum(
            "s, sd -> sd",
            parallel_component,
            self.W_dec.data,
        )

    def fit(self, input_dict, representation):
        """
        Fit the sparse autoencoder to the data.
        """

        lr = input_dict['lr']
        n_lr_warmup = input_dict['n_lr_warmup']
        sparsity_coeff = input_dict['sparsity_coeff']
        n_sparsity_warmup = input_dict['n_sparsity_warmup']
        sae_batch_size = input_dict['sae_batch_size']
        n_workers = input_dict['n_workers']
        device = input_dict.get('device', "cuda")
        n_epochs = input_dict['n_epochs']

        # mode = "online" if cfg.track else "disabled"
        # tags = [cfg.tag] if cfg.tag else []
        # run = ParallelWandbRun(cfg.wandb_project, cfgs, mode, tags)
        self.train()
        sae = self.to(device)

        optimizer = torch.optim.Adam(self.parameters())
        lr_scheduler = Warmup(0.0, lr, n_lr_warmup)
        sparsity_scheduler = Warmup(0.0, sparsity_coeff, n_sparsity_warmup)

        self.dataset = SAEDataset(representation)
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=sae_batch_size, num_workers=n_workers, shuffle=True
        )

        # dataloader = BatchLimiter(dataloader, n_patches)

        global_step, n_patches_seen = 0, 0
        loss_history = {'mse': [], 'sparsity': [], "l0": [], "l1": []}
        for epoch in tqdm.tqdm(range(n_epochs)):

            for batch in dataloader:
                acts_BD = batch["act"].to(device, non_blocking=True)
                sae.normalize_w_dec()
                # Forward passes
                _, _, losses = sae(acts_BD)

                loss_history["mse"].append(losses["mse_loss"].item())
                loss_history["sparsity"].append(losses["sparsity_loss"].item())
                loss_history["l0"].append(losses["l0"].item())
                loss_history["l1"].append(losses["l1"].item())

                n_patches_seen += len(acts_BD)
                # with torch.no_grad():
                #     if (global_step + 1) % cfg.log_every == 0:
                #         metrics = [
                #             {
                #                 "losses/mse": loss.mse.item(),
                #                 "losses/l1": loss.l1.item(),
                #                 "losses/sparsity": loss.sparsity.item(),
                #                 "losses/ghost_grad": loss.ghost_grad.item(),
                #                 "losses/loss": loss.loss.item(),
                #                 "metrics/l0": loss.l0.item(),
                #                 "metrics/l1": loss.l1.item(),
                #                 "progress/n_patches_seen": n_patches_seen,
                #                 "progress/learning_rate": group["lr"],
                #                 "progress/sparsity_coeff": sae.sparsity_coeff,
                #             }
                #             for loss, sae, group in zip(losses, saes, optimizer.param_groups)
                #         ]
                #         run.log(metrics, step=global_step)
                #
                #         logger.info(
                #             "loss: %.5f, mse loss: %.5f, sparsity loss: %.5f, l0: %.5f, l1: %.5f",
                #             losses[0].loss.item(),
                #             losses[0].mse.item(),
                #             losses[0].sparsity.item(),
                #             losses[0].l0.item(),
                #             losses[0].l1.item(),
                #         )

                losses["loss"].backward()
                sae.remove_parallel_grads()

                optimizer.step()

                # Update LR and sparsity coefficients.
                # for param_group, scheduler in zip(optimizer.param_groups, lr_scheduler):
                optimizer.param_groups[0]['lr'] = lr_scheduler.step()
                # param_group["lr"] = scheduler.step()

                # for sae, scheduler in zip(sae, sparsity_schedulers):

                sae.sparsity_coeff = sparsity_scheduler.step()

                # Don't need these anymore.
                optimizer.zero_grad()

                global_step += 1

            print(
                f"Epoch {epoch}, step {global_step}: mse: {losses['mse_loss'].item()}, sparsity: {losses['sparsity_loss'].item()}, l0: {losses['l0'].item()}, l1: {losses['l1'].item()}")

        return global_step, loss_history


def ref_mse(x_hat, x, norm: bool = True):
    mse_loss = torch.pow((x_hat - x.float()), 2)

    if norm:
        mse_loss /= (x ** 2).sum(dim=-1, keepdim=True).sqrt()
    return mse_loss


def safe_mse(x_hat, x, norm: bool = False):
    upper = x.abs().max()
    x = x / upper
    x_hat = x_hat / upper

    mse = (x_hat - x) ** 2
    # (sam): I am now realizing that we normalize by the L2 norm of x.
    if norm:
        mse /= torch.linalg.norm(x, axis=-1, keepdim=True) + 1e-12
        return mse * upper

    return mse * upper * upper


class SAEDataset(torch.utils.data.Dataset):

    def __init__(self, data: Tensor):
        self.data = data
        # normalize data zero mean, unit variance
        self.mean = self.data.mean(dim=0)
        self.std = self.data.std(dim=0)

    def __len__(self) -> int:
        return len(self.data)

    def unnormalize(self, x: Tensor) -> Tensor:
        """
        Unnormalize the data.
        """
        return x.cpu() * self.std + self.mean

    def __getitem__(self, idx: int) -> Tensor:
        tmp = (self.data[idx] - self.mean) / self.std
        return dict(act=tmp)


class Scheduler:
    def step(self) -> float:
        err_msg = f"{self.__class__.__name__} must implement step()."
        raise NotImplementedError(err_msg)

    def __repr__(self) -> str:
        err_msg = f"{self.__class__.__name__} must implement __repr__()."
        raise NotImplementedError(err_msg)


class Warmup(Scheduler):
    """
    Linearly increases from `init` to `final` over `n_warmup_steps` steps.
    """

    def __init__(self, init: float, final: float, n_steps: int):
        self.final = final
        self.init = init
        self.n_steps = n_steps
        self._step = 0

    def step(self) -> float:
        self._step += 1
        if self._step < self.n_steps:
            return self.init + (self.final - self.init) * (self._step / self.n_steps)

        return self.final

    def __repr__(self) -> str:
        return f"Warmup(init={self.init}, final={self.final}, n_steps={self.n_steps})"


class BatchLimiter:
    """
    Limits the number of batches to only return `n_samples` total samples.
    """

    def __init__(self, dataloader: torch.utils.data.DataLoader, n_samples: int):
        self.dataloader = dataloader
        self.n_samples = n_samples
        self.batch_size = dataloader.batch_size

    def __len__(self) -> int:
        return self.n_samples // self.batch_size

    def __iter__(self):
        self.n_seen = 0
        while True:
            for batch in self.dataloader:
                yield batch

                # Sometimes we underestimate because the final batch in the dataloader might not be a full batch.
                self.n_seen += self.batch_size
                if self.n_seen > self.n_samples:
                    return

            # We try to mitigate the above issue by ignoring the last batch if we don't have drop_last.
            if not self.dataloader.drop_last:
                self.n_seen -= self.batch_size

# def dump(fpath: str, sae: SparseAutoencoder):
#     """
#     Save an SAE checkpoint to disk along with configuration, using the [trick from equinox](https://docs.kidger.site/equinox/examples/serialisation).
#
#     Arguments:
#         fpath: filepath to save checkpoint to.
#         sae: sparse autoencoder checkpoint to save.
#     """
#     kwargs = vars(sae.cfg)
#
#     os.makedirs(os.path.dirname(fpath), exist_ok=True)
#     with open(fpath, "wb") as fd:
#         kwargs_str = json.dumps(kwargs)
#         fd.write((kwargs_str + "\n").encode("utf-8"))
#         torch.save(sae.state_dict(), fd)
#
#
# def load(fpath: str, *, device: str = "cpu") -> SparseAutoencoder:
#     """
#     Loads a sparse autoencoder from disk.
#     """
#     with open(fpath, "rb") as fd:
#         kwargs = json.loads(fd.readline().decode())
#         buffer = io.BytesIO(fd.read())
#
#     cfg = config.SparseAutoencoder(**kwargs)
#     model = SparseAutoencoder(cfg)
#     state_dict = torch.load(buffer, weights_only=True, map_location=device)
#     model.load_state_dict(state_dict)
#     return model
