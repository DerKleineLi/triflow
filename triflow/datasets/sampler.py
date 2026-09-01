# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

import heapq
import random

import numpy as np
from torch.utils.data import Sampler
from tqdm import tqdm


class FixedSizeBalancedBatchSampler(Sampler):
    """Batch sampler that keeps per-batch voxel count approximately balanced.

    Each batch contains exactly ``batch_size`` samples. Samples are assigned
    to batches by descending voxel count using a min-heap of per-batch totals,
    so the batch with the currently smallest total is always filled first.
    This gives roughly equal per-batch voxel totals — important when sparse
    operations' runtime is dominated by total coordinate count rather than
    sample count — while keeping the number of samples per batch fixed.

    The number of batches is ``sum(duplicate_factors) // batch_size``; any
    trailing samples that don't fit a full batch are dropped.

    Args:
        metadata: Dict keyed by sample name; each value must include
            ``num_occ_fine`` (the number of fine-resolution occupied voxels).
        batch_size: Exact number of samples per batch.
        shuffle: If ``True``, the *order of batches* is shuffled on every
            ``__iter__`` call. The intra-batch assignment is deterministic
            (greedy by descending cost).
        sample_duplicate: Optional callable ``metadata -> int array`` giving
            the number of times each sample should appear in an epoch. If
            ``None``, every sample is used exactly once.
    """

    def __init__(
        self,
        metadata,
        batch_size,
        shuffle=True,
        sample_duplicate=None,
    ):
        self.metadata = metadata

        self.voxel_counts = [meta["num_occ_fine"] for meta in metadata.values()]
        self.batch_size = batch_size
        self.shuffle = shuffle

        if sample_duplicate is not None:
            self.duplicate_factors = sample_duplicate(metadata)
        else:
            self.duplicate_factors = np.ones(len(self.voxel_counts), dtype=int)

        self.num_samples = len(self.voxel_counts)
        self.num_duplicated_samples = np.sum(self.duplicate_factors)
        self.num_batches = self.num_duplicated_samples // batch_size

        # Precompute sorted indices (largest first) - O(N log N)
        self.sorted_indices = sorted(
            range(self.num_samples), key=lambda i: self.voxel_counts[i], reverse=True
        )

    def __iter__(self):
        """Yield batches of sample indices with balanced voxel totals."""
        batches = [[] for _ in range(self.num_batches)]

        # Use a heap to track (cost, batch_index)
        # Initial state: all batches have 0 cost
        batch_heap = [(0, i) for i in range(self.num_batches)]
        heapq.heapify(batch_heap)

        # Greedy assignment: O(N log M)
        # We only take the first (num_batches * batch_size) to ensure exact batch sizes
        for idx in tqdm(
            self.sorted_indices,
            desc="Assigning batches",
        ):
            for _ in range(self.duplicate_factors[idx]):
                if len(batch_heap) == 0:
                    break  # All batches are full

                # Get the batch with the current minimum voxel count
                cost, b_idx = heapq.heappop(batch_heap)

                batches[b_idx].append(idx)

                # Update the cost and push back into heap
                # If the batch is full, we don't push it back so it's not picked again
                if len(batches[b_idx]) < self.batch_size:
                    heapq.heappush(batch_heap, (cost + self.voxel_counts[idx], b_idx))

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        """Return the number of batches produced per epoch."""
        return self.num_batches
