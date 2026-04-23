import os
import tqdm
import torch
from evaluate import load
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from sklearn.metrics import classification_report, roc_auc_score


class Trainer:
    """训练器基类"""

    def __init__(self, model, device, epochs, learning_rate, checkpoint_steps=400):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.checkpoint_steps = checkpoint_steps
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def __call__(self, dataloader, model_params_path=None, writer=None, is_test=False):
        self.dataloader = dataloader
        self.model_params_path = model_params_path
        self.writer = writer
        self.is_test = is_test
        self.model.to(self.device)
        self.global_step = 0

        if is_test:
            for k, v in self.run_epoch("test").items():
                print(f"Test {k}:", v)
            return

        assert self.model_params_path is not None
        use_amp = self.device.type == "cuda"
        scaler = GradScaler() if use_amp else None
        best_valid_loss = float("inf")

        for epoch in range(self.epochs):
            print(f"\n===== Epoch {epoch + 1}/{self.epochs} =====")

            train_metrics = self.run_epoch("train", epoch, use_amp, scaler)
            for k, v in train_metrics.items():
                print(f"Train {k}: {v:.4f}")

            valid_metrics = self.run_epoch("valid", epoch)
            for k, v in valid_metrics.items():
                print(f"Valid {k}: {v:.4f}")

            if valid_metrics["loss"] <= best_valid_loss:
                best_valid_loss = valid_metrics["loss"]
                parent_dir = os.path.dirname(self.model_params_path)
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir)
                torch.save(self.model.state_dict(), self.model_params_path)
                print(f"保存最佳模型到: {self.model_params_path}")

    def run_epoch(self, phase, epoch=0, use_amp=False, scaler=None):
        self.model.train() if phase == "train" else self.model.eval()
        total_loss = 0.0
        total_examples = 0
        records = {}

        with torch.set_grad_enabled(phase == "train"):
            for inputs in tqdm.tqdm(self.dataloader[phase], desc=phase):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with autocast(device_type=self.device.type, enabled=use_amp):
                    outputs, loss = self.forward(inputs, phase)

                if phase == "train":
                    self.optimizer.zero_grad()
                    if scaler:
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

                    if self.writer:
                        self.writer.add_scalar(
                            f"Loss/{phase}", loss.item(), self.global_step
                        )
                    self.global_step += 1

                    if (
                        self.checkpoint_steps
                        and self.global_step % self.checkpoint_steps == 0
                    ):
                        checkpoint_path = str(self.model_params_path) + ".checkpoint"
                        torch.save(self.model.state_dict(), checkpoint_path)

                current_batch_size = inputs["input_ids"].size(0)
                total_loss += loss.item() * current_batch_size
                total_examples += current_batch_size

                if phase != "train":
                    self.update_records(inputs, outputs, records)

        avg_loss = total_loss / total_examples
        metrics = {"loss": avg_loss}

        if phase != "train":
            self.compute_metrics(metrics, records)
            if self.writer:
                for metric_name, value in metrics.items():
                    self.writer.add_scalar(f"{phase}/{metric_name}", value, epoch)
        return metrics

    def forward(self, inputs, phase):
        raise NotImplementedError

    def update_records(self, inputs, outputs, records):
        raise NotImplementedError

    def compute_metrics(self, metrics, records):
        raise NotImplementedError


class SentimentTrainer(Trainer):
    """情感分类训练器"""

    def forward(self, inputs, phase):
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        return outputs, outputs["loss"]

    def update_records(self, inputs, outputs, records):
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1).detach().cpu()
        preds = logits.argmax(dim=1).detach().cpu()
        labels = inputs["labels"].detach().cpu()
        records.setdefault("probs", []).append(probs)
        records.setdefault("preds", []).append(preds)
        records.setdefault("labels", []).append(labels)

    def compute_metrics(self, metrics, records):
        all_probs = torch.cat(records["probs"])
        all_preds = torch.cat(records["preds"])
        all_labels = torch.cat(records["labels"])
        report = classification_report(
            all_labels, all_preds, output_dict=True, zero_division=0
        )
        metrics["accuracy"] = report["accuracy"]
        metrics.update(report["weighted avg"])
        if all_probs.size(1) > 2:
            auc = roc_auc_score(
                all_labels, all_probs, multi_class="ovr"
            )
        else:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        metrics["auc"] = auc


class Seq2SeqTrainer(Trainer):
    """摘要/问答训练器"""

    def forward(self, inputs, phase):
        outputs = {"loss": torch.tensor(0.0)}
        if phase != "test":
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
        if phase != "train":
            outputs["generated_ids"] = self.model.generate(
                inputs["input_ids"], inputs["attention_mask"]
            )
        return outputs, outputs["loss"]

    def update_records(self, inputs, outputs, records):
        preds = self.model.tokenizer.batch_decode(
            outputs["generated_ids"], skip_special_tokens=True
        )
        labels = self.model.tokenizer.batch_decode(
            torch.where(
                inputs["labels"] == -100,
                self.model.tokenizer.pad_token_id,
                inputs["labels"],
            ),
            skip_special_tokens=True,
        )
        records.setdefault("preds", []).extend(preds)
        records.setdefault("labels", []).extend(labels)

    def compute_metrics(self, metrics, records):
        rouge_scores = load("rouge").compute(
            predictions=records["preds"],
            references=records["labels"],
            tokenizer=self.model.tokenizer.tokenize,
        )
        metrics.update(rouge_scores)
