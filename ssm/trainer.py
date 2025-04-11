import os
import warnings
import torch
from tqdm import tqdm
from torchmetrics import Accuracy
from .model.block.embedding_block import EmbeddingBlock


class Trainer:
    def __init__(
        self,
        model,
        dataset,
        steps,
        logging_steps,
        device=None,
        test_steps=0,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 1e-3},
        enable_progress_bar=True,
        tensorboard_logger=True,
        logging_dir=None,
    ):
        """
        Initialize the Trainer class.

        :param torch.nn.Module model: The model to be trained.
        :param ssm.dataset.CopyDataset dataset: The dataset to be used for
            training.
        :param int steps: The number of training steps.
        :param int logging_steps: The number of steps between logging.
        :param torch.device device: The device to use for training (CPU or GPU).
        :param int test_steps: The number of test steps.
        :param torch.optim.Optimizer optimizer_class: The optimizer class to use.
        :param dict optimizer_params: The parameters for the optimizer.
        :param bool enable_progress_bar: Whether to show a progress bar during
            training.
        :param bool tensorboard_logger: Whether to use TensorBoard for logging.
        :param str logging_dir: The directory for TensorBoard logging.
        """
        self.dataset = iter(dataset)
        n_classes = dataset.alphabet_size
        self.model = EmbeddingBlock(model, n_classes)
        self.steps = steps
        self.test_steps = test_steps
        self.logging_steps = logging_steps
        self.device = device if device else self.set_device()
        self.optimizer = optimizer_class(
            self.model.parameters(), **optimizer_params
        )
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.enable_progress_bar = enable_progress_bar
        self.accuracy = Accuracy(
            task="multiclass", num_classes=n_classes, ignore_index=-1
        )
        if tensorboard_logger:
            from torch.utils.tensorboard import SummaryWriter

            if logging_dir is None:
                warnings.warn(
                    "No logging directory provided. Using default directory."
                )
                logging_dir = "logging_dir/default"
            if not os.path.exists(logging_dir):
                os.makedirs(logging_dir)
            logging_dir = self.logging_folder(logging_dir)
            os.makedirs(logging_dir)
            self.writer = SummaryWriter(log_dir=logging_dir)

    def fit(self):
        """
        Train the model
        """
        self.move_to_device()
        self.model.train()
        pbar = tqdm(range(self.steps), disable=not self.enable_progress_bar)
        for i in pbar:
            # Get a batch of data
            x, y = next(self.dataset)
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            output = self.model(x).permute(0, 2, 1)

            # Compute loss
            loss = self.loss(output, y)
            accuracy = self.accuracy(output, y)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.logging_steps == 0:
                self.logging(pbar, i, loss, accuracy)
        self.writer.close()
        print("Training complete.")

    def test(self):
        """
        Test the model
        """
        self.move_to_device()
        self.model.eval()
        pbar = tqdm(
            range(self.test_steps), disable=not self.enable_progress_bar
        )
        accuracy = 0
        loss = 0
        with torch.no_grad():
            for i in pbar:
                # Get a batch of data
                x, y = next(self.dataset)
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                output = self.model(x).permute(0, 2, 1)

                # Compute loss
                loss += self.loss(output, y)
                accuracy += self.accuracy(output, y)
        print(f"Test Loss: {loss / self.test_steps}")
        print(f"Test Accuracy: {accuracy / self.test_steps}")

    def move_to_device(self):
        """
        Move the model and loss function to the specified device.
        """
        self.model.to(self.device)
        self.accuracy.to(self.device)

    @staticmethod
    def set_device():
        """
        Determine the device to use for training (CPU or GPU). This method
        checks for the availability of CUDA and Metal Performance Shaders
        (MPS) on macOS. If neither is available, it defaults to CPU.
        :return: The device to use for training.
        :rtype: torch.device
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def logging_folder(base_dir):
        """
        Determine the next available logging folder based on existing
        directories in the specified base directory. The folder names are
        expected to follow the format "version_X", where X is an integer
        representing the version number.
        :param str base_dir: The base directory where the logging folders are
            located.
        :return: The path to the next available logging folder.
        :rtype: str
        """
        idx = [
            name.split("version_")[-1]
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name))
            and name.startswith("version_")
        ]
        idx = [int(i) for i in idx]
        if len(idx) == 0:
            return os.path.join(base_dir, "version_0")
        idx = max(idx) + 1
        return os.path.join(base_dir, f"version_{idx}")

    def logging(self, pbar, steps, loss, accuracy):
        """
        Log the training progress by updating the progress bar and writing
        metrics to TensorBoard.
        :param tqdm pbar: The progress bar object.
        :param int steps: The current step number.
        :param torch.Tensor loss: The current loss value.
        :param torch.Tensor accuracy: The current accuracy value.
        """
        pbar.set_postfix(
            loss=loss.item(),
            accuracy=accuracy.item(),
        )
        self.writer.add_scalar("loss", loss.item(), steps)
        self.writer.add_scalar("accuracy", accuracy.item(), steps)
        self.writer.flush()
