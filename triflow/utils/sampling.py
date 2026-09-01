# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.
#
# Adapted from third_party/TRELLIS: trellis/pipelines/samplers/flow_euler.py
# (microsoft/TRELLIS, MIT). Subclasses FlowEulerSampler and overrides only
# _inference_model; see the class docstring for what changed.

from trellis.pipelines.samplers.flow_euler import FlowEulerSampler as TrellisFlowEulerSampler
import torch


class FlowEulerSampler(TrellisFlowEulerSampler):
    """TriFlow-specific wrapper around TRELLIS's ``FlowEulerSampler``.

    Only overrides ``_inference_model`` to rescale the scalar ``t`` in
    ``[0, 1]`` into the 1000-step range the flow model was trained with,
    and to broadcast it to a per-sample tensor before calling the model.
    Everything else (``sample``, CFG mixins, etc.) is inherited from the
    parent class unchanged.
    """

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        """Call the model with ``t`` rescaled to the training range.

        The upstream sampler passes ``t`` as a Python float in ``[0, 1]``;
        our flow model expects a per-sample tensor in ``[0, 1000]``.
        """
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        return model(x_t, t, cond, **kwargs)
