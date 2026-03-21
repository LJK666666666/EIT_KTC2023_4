"""
Abstract base trainer for KTC2023 EIT reconstruction models.

Provides the template for all training workflows with:
  - Auto-incrementing result directories
  - Checkpoint save/load with resume support
  - TensorBoard logging
  - Training log JSON
  - Quick test via max_iters
  - Progress bars
"""

from abc import ABC, abstractmethod
import json
import os
import tempfile
import yaml

import torch
from tqdm import tqdm


class BaseTrainer(ABC):
    """Abstract base class for all KTC2023 model trainers.

    Subclasses must implement:
      build_model():         Create model, optimizer, scheduler
      build_datasets():      Create train_loader, val_data
      train_step(batch):     Process one batch, return {'loss': float}
      validate(epoch):       Run validation, return {'score': float}

    Optional hooks:
      on_epoch_start(epoch)
      on_epoch_end(epoch, metrics)
      get_checkpoint_extra() -> dict
      load_checkpoint_extra(state)
    """

    def __init__(self, config, experiment_name='experiment'):
        self.config = config
        self.device = self._resolve_device(config.device)
        self.experiment_name = experiment_name

        # TPU flag for xla-specific operations
        self._xla = hasattr(self.device, 'type') and str(self.device).startswith('xla')
        if self._xla:
            import torch_xla.core.xla_model as xm
            self._xm = xm

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_data = None

        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        self.training_log = []
        self.writer = None

        # Early stopping state
        self._es_counter = 0
        self._es_best_val_loss = None

    @staticmethod
    def _resolve_device(requested):
        """Resolve device string, with TPU (xla) support.

        Accepted values: 'cuda', 'cpu', 'tpu', 'xla'.
        'tpu' is an alias for 'xla'.
        Falls back to 'cpu' when the requested backend is unavailable.
        """
        if requested in ('tpu', 'xla'):
            try:
                import torch_xla.core.xla_model as xm
                return xm.xla_device()
            except ImportError:
                print('Warning: torch_xla not installed, falling back to cpu. '
                      'Install with: pip install torch_xla')
                return 'cpu'
        if requested == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        if requested == 'cuda':
            print('Warning: CUDA not available, falling back to cpu.')
        return 'cpu'

    def mark_step(self):
        """Trigger XLA graph execution. No-op on CPU/CUDA."""
        if self._xla:
            self._xm.mark_step()

    # ------------------------------------------------------------------
    # Result directory management
    # ------------------------------------------------------------------

    @staticmethod
    def _create_result_dir(experiment_name, base_dir='results'):
        """Create auto-incrementing results directory."""
        num = 1
        while True:
            dir_name = os.path.join(base_dir, f'{experiment_name}_{num}')
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
                return dir_name
            num += 1

    @staticmethod
    def find_latest_result_dir(experiment_name, base_dir='results'):
        """Find most recent result dir for this experiment."""
        num = 1
        latest = None
        while True:
            dir_name = os.path.join(base_dir, f'{experiment_name}_{num}')
            if os.path.exists(dir_name):
                latest = dir_name
                num += 1
            else:
                break
        return latest

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def build_model(self):
        """Create and assign self.model, self.optimizer, self.scheduler."""

    @abstractmethod
    def build_datasets(self):
        """Create and assign self.train_loader, self.val_data."""

    @abstractmethod
    def train_step(self, batch) -> dict:
        """Process one batch. Returns dict with at least 'loss' key."""

    @abstractmethod
    def validate(self, epoch) -> dict:
        """Run validation. Returns dict with at least 'score' key."""

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def on_epoch_start(self, epoch):
        pass

    def on_epoch_end(self, epoch, metrics):
        pass

    def get_checkpoint_extra(self):
        """Return extra state for checkpoint (e.g., EMA)."""
        return {}

    def load_checkpoint_extra(self, state):
        """Load extra state from checkpoint."""
        pass

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        """Main training entry point."""
        # Set up result directory
        resume_path = self.config.training.get('resume_from', None)
        if resume_path:
            # Resume into the same directory as the checkpoint
            self.result_dir = os.path.dirname(resume_path)
        else:
            self.result_dir = self._create_result_dir(self.experiment_name)

        self.build_model()
        self.build_datasets()

        if resume_path:
            self._load_checkpoint(resume_path)

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.result_dir)
        self._save_config()

        max_iters = self.config.training.get('max_iters', None)
        total_epochs = self.config.training.epochs
        save_freq = self.config.training.get('save_freq', 5)

        print(f'Training epochs {self.current_epoch + 1} to {total_epochs}')
        print(f'Results directory: {self.result_dir}')
        if max_iters:
            print(f'Quick test mode: stopping after {max_iters} iterations')

        for epoch in range(self.current_epoch, total_epochs):
            self.current_epoch = epoch
            self.on_epoch_start(epoch)

            epoch_metrics = self._train_epoch(epoch, max_iters=max_iters)

            # Validate
            val_metrics = {}
            val_freq = self.config.training.get('val_freq', 1)
            if val_freq > 0 and (epoch + 1) % val_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    val_metrics = self.validate(epoch)
                self.model.train()

            all_metrics = {**epoch_metrics, **val_metrics}

            # ReduceLROnPlateau: step with val_loss (preferred) or avg_loss
            if self.scheduler is not None:
                sched_metric = val_metrics.get(
                    'val_loss', epoch_metrics.get('avg_loss'))
                if sched_metric is not None:
                    self.scheduler.step(sched_metric)
            all_metrics['lr'] = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/lr', all_metrics['lr'],
                                   epoch + 1)

            self._log_epoch(epoch, all_metrics)

            # Save last checkpoint periodically to reduce cloud-drive I/O.
            if save_freq > 0 and (epoch + 1) % save_freq == 0:
                self._save_checkpoint('last.pt')

            # Save best if val_loss improved (lower is better)
            if 'val_loss' in val_metrics:
                if (self.best_metric is None
                        or val_metrics['val_loss'] < self.best_metric):
                    self.best_metric = val_metrics['val_loss']
                    self._save_checkpoint('best.pt')
                    print(f'  New best val_loss: '
                          f'{val_metrics["val_loss"]:.5f}')

            # Early stopping based on val_loss (lower=better)
            es_loss = val_metrics.get(
                'val_loss', epoch_metrics.get('avg_loss'))
            if es_loss is not None:
                if (self._es_best_val_loss is None
                        or es_loss < self._es_best_val_loss):
                    self._es_best_val_loss = es_loss
                    self._es_counter = 0
                else:
                    self._es_counter += 1

            self.on_epoch_end(epoch, all_metrics)

            # Early stopping check
            es_patience = self.config.training.get(
                'early_stopping_patience', None)
            if es_patience and self._es_counter >= es_patience:
                print(f'Early stopping: val_loss not improved for '
                      f'{es_patience} epochs '
                      f'(best={self._es_best_val_loss:.4f})')
                break

            if max_iters and self.global_step >= max_iters:
                print(f'Quick test: reached {max_iters} iterations, stopping.')
                break

        # Always persist the final resumable checkpoint once training stops.
        self._save_checkpoint('last.pt')

        if self.writer:
            self.writer.close()
        print(f'Training complete. Results saved to: {self.result_dir}')

    def _train_epoch(self, epoch, max_iters=None):
        """Train one epoch with progress bar."""
        self.model.train()
        total_loss = 0.0
        num_items = 0

        pbar = tqdm(enumerate(self.train_loader),
                    total=len(self.train_loader),
                    desc=f'Epoch {epoch + 1}')

        for idx, batch in pbar:
            loss_dict = self.train_step(batch)
            self.mark_step()
            loss_val = loss_dict['loss']

            batch_size = batch[0].shape[0]
            total_loss += loss_val * batch_size
            num_items += batch_size

            self.global_step += 1

            log_freq = self.config.training.get('log_freq', 50)
            if self.global_step % log_freq == 0:
                for key, val in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', val,
                                           self.global_step)

            pbar.set_postfix(loss=f'{loss_val:.5f}')

            if max_iters and self.global_step >= max_iters:
                break

        avg_loss = total_loss / max(num_items, 1)
        print(f'  Average Loss: {avg_loss:.5f}')
        self.writer.add_scalar('train/epoch_avg_loss', avg_loss, epoch + 1)

        return {'avg_loss': avg_loss}

    # ------------------------------------------------------------------
    # Checkpoint methods
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_torch_save(obj, path):
        """Atomically save a PyTorch object to avoid partial checkpoint files."""
        directory = os.path.dirname(path) or '.'
        os.makedirs(directory, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(path) + '.',
            suffix='.tmp',
            dir=directory,
        )
        try:
            with os.fdopen(fd, 'wb') as f:
                torch.save(obj, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    def _save_checkpoint(self, filename):
        """Save checkpoint with all state for resume."""
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': (self.scheduler.state_dict()
                                     if self.scheduler else None),
            'training_log': self.training_log,
            'es_counter': self._es_counter,
            'es_best_val_loss': self._es_best_val_loss,
        }
        state.update(self.get_checkpoint_extra())
        path = os.path.join(self.result_dir, filename)
        self._atomic_torch_save(state, path)

    def _load_checkpoint(self, path):
        """Load checkpoint for resume training."""
        print(f'Resuming from: {path}')
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.current_epoch = state['epoch'] + 1
        self.global_step = state['global_step']
        self.best_metric = state['best_metric']
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        if self.scheduler and state.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.training_log = state.get('training_log', [])
        self._es_counter = state.get('es_counter', 0)
        self._es_best_val_loss = state.get('es_best_val_loss', None)

        self.load_checkpoint_extra(state)
        print(f'  Resumed at epoch {self.current_epoch}, '
              f'step {self.global_step}')

    def _log_epoch(self, epoch, metrics):
        """Append epoch metrics to training_log.json."""
        record = {'epoch': epoch + 1, **metrics}
        self.training_log.append(record)

        log_path = os.path.join(self.result_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)

    def _save_config(self):
        """Save config YAML to results directory."""
        config_path = os.path.join(self.result_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)
