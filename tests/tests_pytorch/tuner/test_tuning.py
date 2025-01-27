# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import BatchSizeFinder, LearningRateFinder
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.tuner.tuning import Tuner


def test_tuner_with_distributed_strategies():
    """Test that an error is raised when tuner is used with multi-device strategy."""
    trainer = Trainer(devices=2, strategy="ddp", accelerator="cpu")
    tuner = Tuner(trainer)
    model = BoringModel()

    with pytest.raises(ValueError, match=r"not supported with distributed strategies"):
        tuner.scale_batch_size(model)


def test_tuner_with_already_configured_batch_size_finder():
    """Test that an error is raised when tuner is already configured with BatchSizeFinder."""
    trainer = Trainer(callbacks=[BatchSizeFinder()])
    tuner = Tuner(trainer)
    model = BoringModel()

    with pytest.raises(ValueError, match=r"Trainer is already configured with a `BatchSizeFinder`"):
        tuner.scale_batch_size(model)


def test_tuner_with_already_configured_learning_rate_finder():
    """Test that an error is raised when tuner is already configured with LearningRateFinder."""
    trainer = Trainer(callbacks=[LearningRateFinder()])
    tuner = Tuner(trainer)
    model = BoringModel()

    with pytest.raises(ValueError, match=r"Trainer is already configured with a `LearningRateFinder`"):
        tuner.lr_find(model)


def test_tuner_batch_size_does_not_exceed_dataset_length(tmp_path):
    """Test that the batch size does not exceed the dataset length."""
    class CustomBoringModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.save_hyperparameters()

    model = CustomBoringModel()
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1)
    tuner = Tuner(trainer)

    # Mock the combined loader to return a small dataset length
    trainer._active_loop = trainer.fit_loop
    trainer._active_loop.setup_data()
    combined_loader = trainer._active_loop._combined_loader
    combined_loader._dataset_length = lambda: 5

    optimal_batch_size = tuner.scale_batch_size(model)
    assert optimal_batch_size == 5, "Batch size should not exceed the dataset length"


def test_tuner_batch_size_scaling_with_small_dataset(tmp_path):
    """Test that the batch size scaling works correctly with a small dataset."""
    class CustomBoringModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.save_hyperparameters()

    model = CustomBoringModel()
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1)
    tuner = Tuner(trainer)

    # Mock the combined loader to return a small dataset length
    trainer._active_loop = trainer.fit_loop
    trainer._active_loop.setup_data()
    combined_loader = trainer._active_loop._combined_loader
    combined_loader._dataset_length = lambda: 10

    optimal_batch_size = tuner.scale_batch_size(model)
    assert optimal_batch_size <= 10, "Batch size should not exceed the dataset length"
