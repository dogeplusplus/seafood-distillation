import os
import uuid
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy
from torchvision import tranforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from models.convolutional import BaseCNN
from processing.dataset import SeafoodDataset


class Distiller(object):
    def __init__(self, student, teacher, device,
        temperature=3, alpha=0.1, experiment_name=None, accumulation_steps=16):
        self.student = student
        self.teacher = teacher
        self.device = device

        self.optimiser = optim.Adam(self.student.parameters())
        self.temperature = temperature
        self.alpha = alpha

        self.experiment_name = experiment_name
        if experiment_name is None:
            self.experiment_name = uuid.uuid4()

        self.accumulation_steps = accumulation_steps
        self.writer = SummaryWriter(log_dir=experiment_name)
        self.loss = nn.CrossEntropyLoss()

    def distill(self, train_dataset, val_dataset=None, epochs=5):

        desc = "Train Epoch: {}"
        val_desc = "Valid Epoch: {}"

        self.optimiser.zero_grad()
        for e in range(epochs):
            train_losses = {
                "student_loss": 0,
                "student_accuracy": 0,
                "distillation_loss": 0,
                "combined_loss": 0,
            }
            val_losses = deepcopy(train_losses)
            train_bar = tqdm.tqdm(
                train_dataset,
                desc=desc.format(e+1),
                total=len(train_dataset),
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
            )
            for i, batch in enumerate(train_bar):
                losses = self.distill_train_step(i, batch)
                for loss in train_losses:
                    train_losses[loss] = (train_losses[loss] * i + losses[loss]) / (i+1)
                self.display_metrics(train_losses, train_bar)

            self.writer.add_scalars("train", train_losses, e)

            if val_dataset is not None:
                val_bar = tqdm.tqdm(
                    val_dataset,
                    desc=val_desc.format(e+1),
                    total=len(val_dataset),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                )
                for i, batch in enumerate(val_bar):
                    losses = self.distill_val_step(batch)
                    for loss in val_losses:
                        val_losses[loss] = (val_losses[loss] * i + losses[loss]) / (i+1)
                    self.display_metrics(val_losses, val_bar)

                self.writer.add_scalars("val", val_losses, e)

        torch.save(
            self.student.state_dict(),
            os.path.join(self.experiment_name, "student.pt")
        )
        return self.student

    def display_metrics(self, metrics_dict, progress_bar):
        evaluated_metrics = {
            k: str(v.detach().numpy())[:7]
            for k, v in metrics_dict.items()
        }
        progress_bar.set_postfix(**evaluated_metrics)

    def distill_train_step(self, idx, sample):
        x, y = sample
        x, y = x.to(self.device), y.to(self.device)

        teacher_pred = self.teacher(x)
        student_pred = self.student(x)

        student_loss = self.loss(student_pred, y)
        d_loss = self.distillation_loss(teacher_pred, student_pred)
        classes = torch.argmax(student_pred, dim=-1)
        student_accuracy = (classes == y).sum() / len(y)

        combined_loss = self.alpha * student_loss + (1 - self.alpha) * d_loss
        combined_loss.backward()

        if (idx + 1) % self.accumulation_steps == 0:
            self.optimiser.step()
            self.optimiser.zero_grad()

        return {
            "student_loss": student_loss.cpu().detach().numpy(),
            "student_accuracy": student_accuracy.cpu().detach().numpy(),
            "distillation_loss": d_loss.cpu().detach().numpy(),
            "combined_loss": combined_loss.cpu().detach().numpy(),
        }

    def distill_val_step(self, sample):
        x, y = sample
        x, y = x.to(self.device), y.to(self.device)
        x.to(self.device)

        with torch.no_grad():
            student_pred = self.student(x)
            teacher_pred = self.teacher(x)
            student_loss = self.loss(student_pred, y)
            d_loss = self.distillation_loss(teacher_pred, student_pred)
            classes = torch.argmax(student_pred, dim=-1)
            student_accuracy = (classes == y).sum() / len(y)

            combined_loss = self.alpha * student_loss \
                + (1 - self.alpha) * d_loss

        return {
            "student_loss": student_loss.cpu().detach().numpy(),
            "student_accuracy": student_accuracy.cpu().detach().numpy(),
            "distillation_loss": d_loss.cpu().detach().numpy(),
            "combined_loss": combined_loss.cpu().detach().numpy(),
        }

    def distillation_loss(self, teacher_pred, student_pred):
        d_loss = F.kl_div(
            F.softmax(teacher_pred / self.temperature, dim=-1),
            F.softmax(student_pred / self.temperature, dim=-1)
        )
        return d_loss


class Trainer(object):
    def __init__(self, model, device, model_name=None, experiment_name=None, accumulation_steps=16):
        self.model = model
        self.model_name = model_name
        if self.model_name is None:
            self.model_name = self.model.__name__()
        self.device = device

        self.optimiser = optim.Adam(self.model.parameters())
        self.loss = nn.CrossEntropyLoss()

        self.experiment_name = experiment_name
        if experiment_name is None:
            self.experiment_name = uuid.uuid4()

        self.accumulation_steps = accumulation_steps
        self.writer = SummaryWriter(log_dir=experiment_name)

    def train(self, train_dataset, val_dataset=None, epochs=5):
        desc = "Train Epoch: {}"
        val_desc = "Valid Epoch: {}"

        self.optimiser.zero_grad()
        for e in range(epochs):
            train_losses = {
                f"{self.model_name}_loss": 0,
                f"{self.model_name}_accuracy": 0,
            }
            val_losses = deepcopy(train_losses)
            train_bar = tqdm.tqdm(
                train_dataset,
                desc=desc.format(e+1),
                total=len(train_dataset),
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
            )
            for i, batch in enumerate(train_bar):
                losses = self.train_step(i, batch)
                for loss in train_losses:
                    train_losses[loss] = (train_losses[loss] * i + losses[loss]) / (i+1)
                self.display_metrics(train_losses, train_bar)

            self.writer.add_scalars("train", train_losses, e)

            if val_dataset is not None:
                val_bar = tqdm.tqdm(
                    val_dataset,
                    desc=val_desc.format(e+1),
                    total=len(val_dataset),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                )
                for i, batch in enumerate(val_bar):
                    losses = self.val_step(batch)
                    for loss in val_losses:
                        val_losses[loss] = (val_losses[loss] * i + losses[loss]) / (i+1)
                    self.display_metrics(val_losses, val_bar)

                self.writer.add_scalars("val", val_losses, e)

        return self.model

    def train_step(self, idx, sample):
        x, y = sample
        x, y = x.to(self.device), y.to(self.device)

        pred = self.model(x)
        loss = self.loss(pred, y)
        classes = torch.argmax(pred, dim=-1)
        accuracy = (classes == y).sum() / len(y)

        loss.backward()
        if (idx + 1) % self.accumulation_steps == 0:
            self.optimiser.step()
            self.optimiser.zero_grad()

        return {
            f"{self.model_name}_loss": loss.cpu().detach().numpy(),
            f"{self.model_name}_accuracy": accuracy.cpu().detach().numpy(),
        }

    def val_step(self, sample):
        x, y = sample
        x, y = x.to(self.device), y.to(self.device)

        with torch.no_grad():
            pred = self.model(x)
            loss = self.loss(pred, y)
            classes = torch.argmax(pred, dim=-1)
            accuracy = (classes == y).sum() / len(y)

        return {
            f"{self.model_name}_loss": loss.cpu().detach().numpy(),
            f"{self.model_name}_accuracy": accuracy.cpu().detach().numpy(),
        }

    def display_metrics(self, metrics_dict, progress_bar):
        evaluated_metrics = {
            k: str(v)[:7]
            for k, v in metrics_dict.items()
        }
        progress_bar.set_postfix(**evaluated_metrics)


def main():
    strides = [2, 2, 2]
    kernel_sizes = [3, 3, 3]
    teacher_filters = [3, 32, 64]
    student_filters = [3, 8, 16]

    num_classes = 10
    epochs = 5
    batch_size = 128

    DATASET_DIR = "data/archive/Fish_Dataset"
    EXPERIMENT_NAME = "distillation"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SeafoodDataset(DATASET_DIR)
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])

    augmentation = transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_loader = DataLoader(train_dataset, batch_size, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size, num_workers=8)

    student = BaseCNN(student_filters, kernel_sizes, strides, num_classes)
    teacher = BaseCNN(teacher_filters, kernel_sizes, strides, num_classes)
    student_copy = BaseCNN(student_filters, kernel_sizes, strides, num_classes)

    student.to(device)
    teacher.to(device)
    student_copy.to(device)

    student_trainer = Trainer(student_copy, device, "student", EXPERIMENT_NAME)
    teacher_trainer = Trainer(teacher, device, "teacher", EXPERIMENT_NAME)

    trained_teacher = teacher_trainer.train(train_loader, val_loader, epochs)
    trained_student = student_trainer.train(train_loader, val_loader, epochs)

    distill = Distiller(student, trained_teacher, device, experiment_name=EXPERIMENT_NAME)
    distilled_student = distill.distill(train_loader, val_loader, epochs)

if __name__ == "__main__":
    main()
