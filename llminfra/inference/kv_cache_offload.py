"""Teaching interface for on-disk KV cache offloading."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch


class OnDiskKVStore:
    """Simple per-sequence on-disk KV cache store.

    This is an interface simulation, not a production storage engine. It
    writes one ``.pt`` file per sequence using ``torch.save``.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, seq_id: int) -> Path:
        return self.directory / f"seq_{seq_id}.pt"

    def save(self, seq_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Persist key/value tensors for ``seq_id``."""
        torch.save({"key": key, "value": value}, self._path(seq_id))

    def load(self, seq_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load key/value tensors for ``seq_id``."""
        path = self._path(seq_id)
        if not path.exists():
            raise FileNotFoundError(f"KV cache not found: {path}")
        payload = torch.load(path, weights_only=True)
        return payload["key"], payload["value"]

    def delete(self, seq_id: int) -> None:
        """Delete the on-disk KV cache for ``seq_id``."""
        self._path(seq_id).unlink(missing_ok=True)


@dataclass
class KVRecord:
    """Internal tiered-cache record."""

    key: torch.Tensor
    value: torch.Tensor
    last_access: float


class TieredKVCache:
    """Reference HBM/CPU/NVMe KV cache with LRU promotion and eviction.

    HBM is represented by tensors on the caller's device, CPU by detached
    host tensors, and NVMe by :class:`OnDiskKVStore`. The implementation is
    synchronous and sequence-granular; production serving systems should use
    block-granular asynchronous DMA and a scheduler aware of request
    deadlines.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        max_hbm_entries: int = 8,
        max_cpu_entries: int = 32,
        hbm_device: str | torch.device | None = None,
    ) -> None:
        if max_hbm_entries < 1 or max_cpu_entries < 1:
            raise ValueError("tier capacities must be >= 1")
        self.store = OnDiskKVStore(directory)
        self.max_hbm_entries = int(max_hbm_entries)
        self.max_cpu_entries = int(max_cpu_entries)
        if hbm_device is None:
            hbm_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hbm_device = torch.device(hbm_device)
        self.hbm: dict[int, KVRecord] = {}
        self.cpu: dict[int, KVRecord] = {}
        self.nvme: set[int] = set()

    def put(self, seq_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Insert or replace a sequence in the HBM tier."""
        if key.shape != value.shape:
            raise ValueError("key and value shapes must match")
        if seq_id in self.nvme:
            self.store.delete(seq_id)
        self.hbm[seq_id] = KVRecord(
            key.detach().clone().to(self.hbm_device),
            value.detach().clone().to(self.hbm_device),
            time.monotonic(),
        )
        self.cpu.pop(seq_id, None)
        self.nvme.discard(seq_id)
        self._evict_hbm()

    def get(self, seq_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a sequence and promote it to HBM."""
        if seq_id in self.hbm:
            record = self.hbm[seq_id]
        elif seq_id in self.cpu:
            record = self.cpu.pop(seq_id)
            self.hbm[seq_id] = KVRecord(
                record.key.to(self.hbm_device),
                record.value.to(self.hbm_device),
                time.monotonic(),
            )
            self._evict_hbm()
            record = self.hbm[seq_id]
        elif seq_id in self.nvme:
            key, value = self.store.load(seq_id)
            self.store.delete(seq_id)
            self.hbm[seq_id] = KVRecord(
                key.to(self.hbm_device),
                value.to(self.hbm_device),
                time.monotonic(),
            )
            self.nvme.discard(seq_id)
            self._evict_hbm()
            record = self.hbm[seq_id]
        else:
            raise KeyError(f"unknown KV sequence: {seq_id}")
        record.last_access = time.monotonic()
        return record.key, record.value

    def delete(self, seq_id: int) -> None:
        """Remove a sequence from all tiers."""
        self.hbm.pop(seq_id, None)
        self.cpu.pop(seq_id, None)
        if seq_id in self.nvme:
            self.store.delete(seq_id)
            self.nvme.discard(seq_id)

    def _evict_hbm(self) -> None:
        while len(self.hbm) > self.max_hbm_entries:
            seq_id, record = min(self.hbm.items(), key=lambda item: item[1].last_access)
            del self.hbm[seq_id]
            self.cpu[seq_id] = KVRecord(
                record.key.detach().clone().to("cpu"),
                record.value.detach().clone().to("cpu"),
                record.last_access,
            )
        while len(self.cpu) > self.max_cpu_entries:
            seq_id, record = min(self.cpu.items(), key=lambda item: item[1].last_access)
            del self.cpu[seq_id]
            self.store.save(seq_id, record.key, record.value)
            self.nvme.add(seq_id)

    def tier_counts(self) -> dict[str, int]:
        """Return the number of entries in each storage tier."""
        return {
            "hbm": len(self.hbm),
            "cpu": len(self.cpu),
            "nvme": len(self.nvme),
        }
