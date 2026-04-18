import torch
import numpy as np


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        device,
        writer,
        grad_accum_steps=1,
        log_every=10,
        clip=1.0,
        early_stopping=20,
        min_delta = 0.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.writer = writer
        self.grad_accum_steps = grad_accum_steps
        self.log_every = log_every
        self.clip = clip
        self.early_stopping = early_stopping
        self.min_delta = min_delta

        self.no_improve_steps = 0
        self.optim_step = 0
        self.best_loss = float("inf")

        # if bf16 is supported, scaler is disabled  (bf16 preferred for training stability)
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler(enabled=(self.amp_dtype == torch.float16))

    # ------------------------

    def compute_loss(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
            _, loss = self.model(input_ids, labels)

        return loss

    # ------------------------

    def validate(self):
        self.model.eval()
        losses = []

        with torch.no_grad():
            for batch in self.valid_loader:
                loss = self.compute_loss(batch)
                losses.append(loss.item())

        self.model.train()
        return np.mean(losses)

    # ------------------------

    def train(self, max_steps):

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        train_losses = []
        accum_step = 0

        while self.optim_step < max_steps:

            for batch in self.train_loader:

                loss = self.compute_loss(batch)
                train_losses.append(loss.item())

                # scale for grad accumulation before backward (disabled if bf16)
                self.scaler.scale(loss / self.grad_accum_steps).backward()

                accum_step += 1

                # ---- optimizer step ----
                if accum_step % self.grad_accum_steps == 0:

                    # unscale before clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()  # adjusts scale factor for next iter

                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    self.optim_step += 1

                    # ---- logging ----
                    if self.optim_step % self.log_every == 0:

                        avg_train_loss = np.mean(train_losses)
                        train_losses = []

                        lr = self.optimizer.param_groups[0]["lr"]

                        print(
                            f"step {self.optim_step} | "
                            f"train_loss {avg_train_loss:.4f} | lr {lr:.2e}"
                        )

                        self.writer.add_scalar("train/loss", avg_train_loss, self.optim_step)
                        self.writer.add_scalar("train/lr", lr, self.optim_step)

                        # ---- validation ----
                        val_loss = self.validate()

                        print(f"step {self.optim_step} | valid_loss {val_loss:.4f}")

                        self.writer.add_scalar("valid/loss", val_loss, self.optim_step)
                        self.writer.flush()

                        # ---- early stopping logic ----
                        improved = val_loss < (self.best_loss - self.min_delta)

                        if improved:
                            self.best_loss = val_loss
                            self.no_improve_steps = 0
                            # save best checkpoint
                            torch.save({
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "scheduler_state_dict": self.scheduler.state_dict(),
                                "optim_step": self.optim_step,
                                "config": {
                                    "vocab_size": self.model.vocab_size,
                                    "d_model": 384,
                                    "n_layers": 8,
                                    "n_heads": 6,
                                    "d_ff": 4*384,
                                    "max_seq_len": 1024,
                                },
                            }, "./utils/checkpoints/best_checkpoint.pt")

                        else:
                            self.no_improve_steps += 1
                            print(f"No improvement ({self.no_improve_steps}/{self.early_stopping})")

                            if self.no_improve_steps >= self.early_stopping:
                                print("Early stopping triggered")
                                return   # exit from training

                    # ---- stop condition ----
                    if self.optim_step >= max_steps:
                        break

            if self.optim_step >= max_steps:
                break
