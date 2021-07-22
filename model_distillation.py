import os
import uuid
import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from models.convolutional import BaseCNN
from processing.dataset import SeafoodDataset


class Distiller(object):
    def __init__(self, student, teacher,
        temperature=3, alpha=0.1, experiment_name=None):
        self.student = student
        self.teacher = teacher

        self.optimiser = optim.Adam(self.student.parameters())
        self.temperature = temperature
        self.alpha = alpha

        self.experiment_name = experiment_name
        if experiment_name is None:
            self.experiment_name = uuid.uuid4()

        self.writer = SummaryWriter(log_dir=experiment_name)
        self.loss = F.cross_entropy

    def distill(self, train_dataset, val_dataset=None, epochs=5):

        desc = "Train Epoch: {}"
        val_desc = "Valid Epoch: {}"

        for e in range(epochs):
            train_losses = {
                "student_loss": 0,
                "distillation_loss": 0,
                "combined_loss": 0,
            }
            val_losses = deepcopy(train_losses)
            train_bar = tqdm.tqdm(
                train_dataset,
                desc=desc.format(e+1),
                ncols=50,
                total=len(train_dataset)
            )
            for batch in train_bar:
                losses = self.distill_train_step(batch)
                for loss in train_losses:
                    train_losses[loss] += losses[loss]
                self.display_metrics(train_losses, train_bar)

            self.writer.add_scalars("train", train_losses, e)

            if val_dataset is not None:
                val_bar = tqdm.tqdm(
                    val_dataset,
                    desc=val_desc.format(e+1),
                    ncols=50,
                    total=len(val_dataset)
                )
                for batch in val_bar:
                    losses = self.distill_val_step(batch)
                    for loss in val_losses:
                        val_losses[loss] += losses[loss]
                    self.display_metrics(val_losses, val_bar)

                self.writer.add_scalars("val", val_losses, e)

        torch.save(
            self.student.state_dict(),
            os.path.join(self.experiment_name, "student.pt")
        )

    def display_metrics(self, metrics_dict, progress_bar):
        evaluated_metrics = {
            k: str(v.result().numpy())[:7]
            for k, v in metrics_dict.items()
        }
        progress_bar.set_postfix(**evaluated_metrics)

    def distill_train_step(self, sample):
        x, y = sample

        teacher_pred = self.teacher(x)

        self.optimiser.zero_grad()

        student_pred = self.student(x)
        student_loss = self.loss(y, student_pred)
        d_loss = self.distillation_loss(teacher_pred, student_pred)

        combined_loss = self.alpha * student_loss + (1 - self.alpha) * d_loss
        combined_loss.backward()

        return {
            "student_loss": student_loss.numpy(),
            "distillation_loss": d_loss.numpy(),
            "combined_loss": combined_loss.numpy(),
        }

    def distill_val_step(self, sample):
        x, y = sample

        with torch.no_grad():
            student_pred = self.student(x)
            teacher_pred = self.teacher(x)
            student_loss = self.loss(y, student_pred)
            d_loss = self.distillation_loss(teacher_pred, student_pred)

            combined_loss = self.alpha * student_loss \
                + (1 - self.alpha) * d_loss

        return {
            "student_loss": student_loss.numpy(),
            "distillation_loss": d_loss.numpy(),
            "combined_loss": combined_loss.numpy(),
        }

    def distillation_loss(self, teacher_pred, student_pred):
        d_loss = F.kl_div(
            F.softmax(teacher_pred / self.temperature, dim=-1),
            F.softmax(student_pred / self.temperature, dim=-1)
        )
        return d_loss


class Trainer(object):
    def __init__(self, model, model_name=None, experiment_name=None):
        self.model = model
        self.model_name = model_name
        if self.model_name is None:
            self.model_name = self.model.__name__()

        self.optimiser = optim.Adam(self.model.parameters())
        self.loss = F.cross_entropy

        self.experiment_name = experiment_name
        if experiment_name is None:
            self.experiment_name = uuid.uuid4()

        self.writer = SummaryWriter(log_dir=experiment_name)

    def train(self, train_dataset, val_dataset=None, epochs=5):
        desc = "Train Epoch: {}"
        val_desc = "Valid Epoch: {}"

        for e in range(epochs):
            train_losses = {
                "loss": 0,
            }
            val_losses = deepcopy(train_losses)
            train_bar = tqdm.tqdm(
                train_dataset,
                desc=desc.format(e+1),
                ncols=50,
                total=len(train_dataset)
            )
            for batch in train_bar:
                losses = self.train_step(batch)
                for loss in train_losses:
                    train_losses[loss] += losses[loss]
                self.display_metrics(train_losses, train_bar)

            self.writer.add_scalars("train", train_losses, e)

            if val_dataset is not None:
                val_bar = tqdm.tqdm(
                    val_dataset,
                    desc=val_desc.format(e+1),
                    ncols=50,
                    total=len(val_dataset)
                )
                for batch in val_bar:
                    losses = self.val_teacher_step(batch)
                    for loss in val_losses:
                        val_losses[loss] += losses[loss]
                    self.display_metrics(val_losses, val_bar)

                self.writer.add_scalars("val", val_losses, e)

        return self.model

    def train_step(self, sample):
        x, y = sample

        self.optimiser.zero_grad()
        pred = self.model(x)
        loss = self.loss(y, pred)

        loss.backward()

        return {
            f"{self.model_name}_loss": loss.numpy(),
        }

    def val_step(self, sample):
        x, y = sample

        with torch.no_grad():
            pred = self.model(x)
            loss = self.loss(y, pred)

        return {
            f"{self.model_name}_loss": loss.numpy(),
        }

    def display_metrics(self, metrics_dict, progress_bar):
        evaluated_metrics = {
            k: str(v.result().numpy())[:7]
            for k, v in metrics_dict.items()
        }
        progress_bar.set_postfix(**evaluated_metrics)


def main():
    strides = [2, 2, 2, 2]
    kernel_sizes = [3, 3, 3, 3]
    teacher_filters = [3, 64, 128, 256]
    student_filters = [3, 16, 32, 64]

    num_classes = 10
    epochs = 10
    batch_size = 2

    DATASET_DIR = "data/archive/Fish_Dataset"
    EXPERIMENT_NAME = "distillation"

    dataset = SeafoodDataset(DATASET_DIR)
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size)
    val_loader = DataLoader(val_dataset)

    student = BaseCNN(student_filters, kernel_sizes, strides, num_classes)
    teacher = BaseCNN(teacher_filters, kernel_sizes, strides, num_classes)
    student_copy = BaseCNN(student_filters, kernel_sizes, strides, num_classes)

    student_trainer = Trainer(student_copy, "student", EXPERIMENT_NAME)
    teacher_trainer = Trainer(teacher, "teacher", EXPERIMENT_NAME)

    trained_teacher = teacher_trainer.train(train_loader, val_loader, epochs)
    trained_student = student_trainer.train(train_loader, val_loader, epochs)

    distill = Distiller(student, trained_teacher, experiment_name=EXPERIMENT_NAME)
    distill.distill(train_loader, val_loader, epochs)

if __name__ == "__main__":
    main()
