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
        cfg
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.writer = writer

        # training behavior params from cfg
        self.grad_accum_steps = cfg.stages.train.grad_accum_steps
        self.log_every = cfg.stages.train.log_every
        self.min_delta = cfg.stages.train.min_delta
        self.early_stopping = cfg.stages.train.early_stopping
        self.clip = cfg.stages.optim.clip
        self.checkpoint_path = cfg.checkpoint.save_path

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

        shift_labels = labels[:, 1:]
        valid_tokens = (shift_labels != -100).sum().item()

        return loss, valid_tokens

    # ------------------------

    def validate(self):
        self.model.eval()
        total_valid_loss = 0.0
        total_valid_tokens = 0

        with torch.no_grad():
            for batch in self.valid_loader:
                loss, valid_tokens = self.compute_loss(batch)
                total_valid_loss += loss.item() * valid_tokens
                total_valid_tokens += valid_tokens

        self.model.train()
        return total_valid_loss / total_valid_tokens

    # ------------------------

    def train(self, max_steps):

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0
        total_tokens = 0
        accum_step = 0

        while self.optim_step < max_steps:

            for batch in self.train_loader:

                loss, valid_tokens = self.compute_loss(batch)
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens

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

                        avg_train_loss = total_loss / total_tokens
                        total_loss = 0
                        total_tokens = 0

                        lr = self.optimizer.param_groups[0]["lr"]

                        print(
                            f"step {self.optim_step} | "
                            f"train_loss {avg_train_loss:.4f} | lr {lr:.2e}"
                        )

                        self.writer.add_scalar("train/loss", avg_train_loss, self.optim_step)

                        # ---- validation ----
                        valid_loss = self.validate()

                        print(f"step {self.optim_step} | valid_loss {valid_loss:.4f}")

                        self.writer.add_scalar("valid/loss", valid_loss, self.optim_step)
                        self.writer.add_scalar("optim/lr", lr, self.optim_step)
                        self.writer.flush()

                        # ---- early stopping logic ----
                        improved = valid_loss < (self.best_loss - self.min_delta)

                        if improved:
                            self.best_loss = valid_loss
                            self.no_improve_steps = 0
                            # save best checkpoint
                            torch.save({
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "scheduler_state_dict": self.scheduler.state_dict(),
                                "optim_step": self.optim_step,
                            }, self.checkpoint_path)

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
