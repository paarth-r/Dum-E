"""Policy interfaces for closed-loop control and learning.

A :class:`Policy` maps an :class:`~dume.dataset.Observation` to an :class:`~dume.dataset.Action`.
In deployment, the controller calls ``policy.select_action(obs)`` each tick and sends the
resulting joint targets to the arm.

Two concrete policies are provided:

* :class:`ScriptedPolicy` — deterministic, hardware-free. Used in tests, bring-up scripts,
  and any scenario where you want to replay a fixed joint sequence or drive with a lambda.

* :class:`LeRobotDiffusionPolicy` — a future adapter stub. It will wrap lerobot's
  ``DiffusionPolicy`` once real demonstration data exists and the model has been trained.
  All methods raise :exc:`NotImplementedError` until then.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from dume.dataset import Action, Observation


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Policy(Protocol):
    """Minimal interface every policy must satisfy."""

    def select_action(self, observation: Observation) -> Action:
        """Return the action to execute given the current observation."""
        ...

    def load(self, path: str) -> None:
        """Load weights/parameters from ``path``."""
        ...

    def train(self, dataset) -> None:
        """Train (or fine-tune) the policy on ``dataset``.

        ``dataset`` may be a :class:`~dume.dataset.LocalBackend`, a list of
        :class:`~dume.dataset.Episode` objects, or a LeRobot dataset — subclasses
        document what they accept.
        """
        ...


# ---------------------------------------------------------------------------
# ScriptedPolicy — real, hardware-free
# ---------------------------------------------------------------------------


class ScriptedPolicy:
    """Returns a predetermined action each tick — for tests and scripted bring-up.

    Constructed with either a fixed :class:`~dume.dataset.Action` *or* a callable that
    maps ``Observation -> Action``. The callable form lets tests parameterise the action
    on the current observation (e.g. drive joints toward a target).

    ``load`` and ``train`` are no-ops: there are no learnable parameters.

    Parameters
    ----------
    action_or_fn:
        Either a fixed :class:`~dume.dataset.Action` returned every tick, or a callable
        ``(Observation) -> Action`` invoked on each ``select_action`` call.
    """

    def __init__(self, action_or_fn: Action | Callable[[Observation], Action]) -> None:
        if callable(action_or_fn):
            self._fn: Callable[[Observation], Action] = action_or_fn
        else:
            # Wrap the fixed action in a lambda so select_action is uniform
            fixed = action_or_fn
            self._fn = lambda _obs: fixed

    def select_action(self, observation: Observation) -> Action:
        """Return the scripted action, optionally conditioned on ``observation``."""
        return self._fn(observation)

    def load(self, path: str) -> None:
        """No-op — ScriptedPolicy has no learnable parameters."""

    def train(self, dataset) -> None:
        """No-op — ScriptedPolicy has no learnable parameters."""


# ---------------------------------------------------------------------------
# LeRobotDiffusionPolicy — stub adapter
# ---------------------------------------------------------------------------


class LeRobotDiffusionPolicy:
    """Adapter stub for lerobot's DiffusionPolicy.

    Once real SO-101 demonstrations have been recorded with :class:`~dume.dataset.EpisodeRecorder`
    and exported via :func:`~dume.dataset.to_lerobot`, this class will wrap
    ``lerobot.common.policies.diffusion.modeling_diffusion.DiffusionPolicy``. It will:

    * Accept a HuggingFace pretrained checkpoint path in ``load()``.
    * Run the denoising diffusion process inside ``select_action()`` to produce joint
      targets from a recent observation window (horizon configurable via ``config``).
    * Support fine-tuning / training from scratch in ``train()``.

    Until trained weights exist, all methods raise :exc:`NotImplementedError`.

    Parameters
    ----------
    **config:
        Keyword arguments forwarded to the future DiffusionPolicy config
        (e.g. ``n_action_steps``, ``horizon``, ``device``). Stored but not used yet.
    """

    def __init__(self, **config) -> None:
        self._config = config
        self._weights_loaded = False

    def select_action(self, observation: Observation) -> Action:
        """Run the diffusion policy to select an action.

        Raises
        ------
        NotImplementedError
            Always until trained weights are loaded via :meth:`load`.
        """
        if not self._weights_loaded:
            raise NotImplementedError(
                "LeRobotDiffusionPolicy needs trained weights — load() first"
            )
        raise NotImplementedError(
            "LeRobotDiffusionPolicy needs trained weights — load() first"
        )

    def load(self, path: str) -> None:
        """Load DiffusionPolicy weights from ``path`` (HuggingFace repo or local dir).

        Raises
        ------
        NotImplementedError
            Always — wraps lerobot DiffusionPolicy; needs recorded data + hardware.
        """
        raise NotImplementedError(
            "needs recorded data + hardware; wraps lerobot DiffusionPolicy"
        )

    def train(self, dataset) -> None:
        """Train the DiffusionPolicy on ``dataset``.

        Raises
        ------
        NotImplementedError
            Always — wraps lerobot DiffusionPolicy; needs recorded data + hardware.
        """
        raise NotImplementedError(
            "needs recorded data + hardware; wraps lerobot DiffusionPolicy"
        )
