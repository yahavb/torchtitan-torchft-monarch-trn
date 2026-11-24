# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fail-closed Neuron bindings for the pinned TorchTitan attention path."""

from __future__ import annotations

import functools
import importlib
import inspect
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any


_ADAPTER_VERSION = "m1-neuron-attention-v1"
_OWNER_ATTRIBUTE = "_torchft_neuron_m1_neuron_attention_adapter"
_FLEX_ATTRIBUTE = "_torchft_neuron_m1_neuron_flex_callable"
_MASK_ATTRIBUTE = "_torchft_neuron_m1_neuron_mask_callable"
_ORIGINAL_SELECTOR_ATTRIBUTE = "_torchft_neuron_m1_original_backend_selector"

_UPSTREAM_FLEX_OPTIONS = {
    "wrap_inductor_compiled_regions": True,
    "max_autotune": True,
    "coordinate_descent_tuning": True,
    "triton.cudagraphs": False,
}
_NEURON_FLEX_OPTIONS = {
    **_UPSTREAM_FLEX_OPTIONS,
    "wrap_inductor_compiled_regions": False,
}

_INSTALL_LOCK = threading.Lock()

_TORCH_COMPILE_PARAMETERS = (
    ("model", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("fullgraph", inspect.Parameter.KEYWORD_ONLY),
    ("dynamic", inspect.Parameter.KEYWORD_ONLY),
    ("backend", inspect.Parameter.KEYWORD_ONLY),
    ("mode", inspect.Parameter.KEYWORD_ONLY),
    ("options", inspect.Parameter.KEYWORD_ONLY),
    ("name", inspect.Parameter.KEYWORD_ONLY),
    ("disable", inspect.Parameter.KEYWORD_ONLY),
    ("recompile_limit", inspect.Parameter.KEYWORD_ONLY),
    ("isolate_recompiles", inspect.Parameter.KEYWORD_ONLY),
    ("shapes_spec", inspect.Parameter.KEYWORD_ONLY),
)
_FLEX_ATTENTION_PARAMETERS = (
    ("query", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("key", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("value", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("score_mod", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("block_mask", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("scale", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("enable_gqa", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("return_lse", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("kernel_options", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("return_aux", inspect.Parameter.KEYWORD_ONLY),
)
_CREATE_BLOCK_MASK_PARAMETERS = (
    ("mask_mod", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("B", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("H", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("Q_LEN", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("KV_LEN", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("device", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("BLOCK_SIZE", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("_compile", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("separate_full_blocks", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("compute_dq_write_order", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("dq_kv_order", inspect.Parameter.POSITIONAL_OR_KEYWORD),
)


class NeuronAttentionContractError(RuntimeError):
    """The pinned TorchTitan attention contract is absent or was modified."""


@dataclass(frozen=True, slots=True)
class _Targets:
    torch_module: Any
    attention_module: ModuleType
    compile_module: ModuleType
    flex_attention_class: type
    flex_attention: Callable[..., Any]
    create_block_mask: Callable[..., Any]
    selector: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class _Installation:
    targets: _Targets
    compiled_flex_attention: Callable[..., Any]
    compiled_create_block_mask: Callable[..., Any]
    selector: Callable[..., Any]


_installation: _Installation | None = None


def _import_module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise NeuronAttentionContractError(
            f"cannot import pinned attention dependency {name}: {error}"
        ) from error


def _require_callable(owner: Any, name: str) -> Callable[..., Any]:
    value = getattr(owner, name, None)
    if not callable(value):
        raise NeuronAttentionContractError(f"{name} must exist and be callable")
    return value


def _validate_parameters(
    target: Callable[..., Any],
    *,
    label: str,
    expected: tuple[tuple[str, inspect._ParameterKind], ...],
) -> None:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError) as error:
        raise NeuronAttentionContractError(f"cannot inspect {label}") from error
    actual = tuple(
        (parameter.name, parameter.kind) for parameter in signature.parameters.values()
    )
    if actual != expected:
        raise NeuronAttentionContractError(f"{label} signature drifted")


def _validate_selector(selector: Callable[..., Any]) -> None:
    _validate_parameters(
        selector,
        label="_maybe_regional_inductor_backend",
        expected=(
            ("model", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ("backend", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ),
    )


def _resolve_targets() -> _Targets:
    torch_module = _import_module("torch")
    flex_module = _import_module("torch.nn.attention.flex_attention")
    attention_module = _import_module("torchtitan.models.common.attention")
    compile_module = _import_module("torchtitan.distributed.compile")

    torch_compile = _require_callable(torch_module, "compile")
    flex_attention = _require_callable(flex_module, "flex_attention")
    create_block_mask = _require_callable(flex_module, "create_block_mask")
    _validate_parameters(
        torch_compile,
        label="torch.compile",
        expected=_TORCH_COMPILE_PARAMETERS,
    )
    _validate_parameters(
        flex_attention,
        label="flex_attention",
        expected=_FLEX_ATTENTION_PARAMETERS,
    )
    _validate_parameters(
        create_block_mask,
        label="create_block_mask",
        expected=_CREATE_BLOCK_MASK_PARAMETERS,
    )

    flex_attention_class = getattr(attention_module, "FlexAttention", None)
    if not isinstance(flex_attention_class, type):
        raise NeuronAttentionContractError("FlexAttention must exist and be a class")
    _require_callable(flex_attention_class, "_compiled_flex_attn")
    _require_callable(attention_module, "_compiled_create_block_mask")
    _require_callable(attention_module, "create_attention_mask")

    inductor_configs = getattr(flex_attention_class, "inductor_configs", None)
    if not isinstance(inductor_configs, Mapping) or dict(inductor_configs) != (
        _UPSTREAM_FLEX_OPTIONS
    ):
        raise NeuronAttentionContractError(
            "FlexAttention.inductor_configs drifted from the pinned contract"
        )

    selector = _require_callable(
        compile_module,
        "_maybe_regional_inductor_backend",
    )
    _validate_selector(selector)
    regional_enabled = getattr(compile_module, "_regional_inductor_enabled", None)
    if regional_enabled is not False:
        raise NeuronAttentionContractError(
            "regional inductor must be disabled before installing Neuron attention"
        )
    return _Targets(
        torch_module=torch_module,
        attention_module=attention_module,
        compile_module=compile_module,
        flex_attention_class=flex_attention_class,
        flex_attention=flex_attention,
        create_block_mask=create_block_mask,
        selector=selector,
    )


def _verify_installation(installation: _Installation) -> None:
    targets = installation.targets
    if (
        vars(targets.flex_attention_class).get("_compiled_flex_attn")
        is not installation.compiled_flex_attention
        or getattr(targets.attention_module, "_compiled_create_block_mask", None)
        is not installation.compiled_create_block_mask
        or getattr(
            targets.compile_module,
            "_maybe_regional_inductor_backend",
            None,
        )
        is not installation.selector
    ):
        raise NeuronAttentionContractError(
            "the installed Neuron attention bindings were replaced"
        )
    if getattr(installation.selector, _OWNER_ATTRIBUTE, None) != _ADAPTER_VERSION:
        raise NeuronAttentionContractError(
            "the installed backend selector lost its ownership marker"
        )


def _adopt_existing(targets: _Targets) -> _Installation | None:
    marker = getattr(targets.selector, _OWNER_ATTRIBUTE, None)
    if marker is None:
        return None
    if marker != _ADAPTER_VERSION:
        raise NeuronAttentionContractError(
            "an incompatible Neuron attention adapter is already installed"
        )
    installation = _Installation(
        targets=targets,
        compiled_flex_attention=getattr(targets.selector, _FLEX_ATTRIBUTE, None),
        compiled_create_block_mask=getattr(
            targets.selector,
            _MASK_ATTRIBUTE,
            None,
        ),
        selector=targets.selector,
    )
    if not callable(installation.compiled_flex_attention) or not callable(
        installation.compiled_create_block_mask
    ):
        raise NeuronAttentionContractError(
            "the existing Neuron attention adapter is incomplete"
        )
    _verify_installation(installation)
    return installation


def _build_selector(
    targets: _Targets,
    compiled_flex_attention: Callable[..., Any],
    compiled_create_block_mask: Callable[..., Any],
) -> Callable[..., Any]:
    original_selector = targets.selector

    @functools.wraps(original_selector)
    def neuron_aware_selector(model: Any, backend: str) -> Any:
        if backend != "neuron":
            return original_selector(model, backend)
        if (
            vars(targets.flex_attention_class).get("_compiled_flex_attn")
            is not compiled_flex_attention
            or getattr(
                targets.attention_module,
                "_compiled_create_block_mask",
                None,
            )
            is not compiled_create_block_mask
            or getattr(targets.compile_module, "_regional_inductor_enabled", None)
            is not False
        ):
            raise NeuronAttentionContractError(
                "Neuron attention bindings changed before model compilation"
            )
        return backend

    setattr(neuron_aware_selector, _OWNER_ATTRIBUTE, _ADAPTER_VERSION)
    setattr(neuron_aware_selector, _FLEX_ATTRIBUTE, compiled_flex_attention)
    setattr(neuron_aware_selector, _MASK_ATTRIBUTE, compiled_create_block_mask)
    setattr(
        neuron_aware_selector,
        _ORIGINAL_SELECTOR_ATTRIBUTE,
        original_selector,
    )
    return neuron_aware_selector


def _restore_targets(
    targets: _Targets,
    original_flex_attention: Callable[..., Any],
    original_create_block_mask: Callable[..., Any],
    original_selector: Callable[..., Any],
) -> None:
    setattr(
        targets.flex_attention_class,
        "_compiled_flex_attn",
        original_flex_attention,
    )
    setattr(
        targets.attention_module,
        "_compiled_create_block_mask",
        original_create_block_mask,
    )
    setattr(
        targets.compile_module,
        "_maybe_regional_inductor_backend",
        original_selector,
    )


def _publish_installation(
    targets: _Targets,
    compiled_flex_attention: Callable[..., Any],
    compiled_create_block_mask: Callable[..., Any],
    selector: Callable[..., Any],
) -> _Installation:
    original_flex_attention = vars(targets.flex_attention_class)["_compiled_flex_attn"]
    original_create_block_mask = getattr(
        targets.attention_module,
        "_compiled_create_block_mask",
    )
    original_selector = targets.selector
    installation = _Installation(
        targets=targets,
        compiled_flex_attention=compiled_flex_attention,
        compiled_create_block_mask=compiled_create_block_mask,
        selector=selector,
    )
    try:
        setattr(
            targets.flex_attention_class,
            "_compiled_flex_attn",
            compiled_flex_attention,
        )
        setattr(
            targets.attention_module,
            "_compiled_create_block_mask",
            compiled_create_block_mask,
        )
        # The selector is the commit point: the old selector rejects Neuron.
        setattr(
            targets.compile_module,
            "_maybe_regional_inductor_backend",
            selector,
        )
        _verify_installation(installation)
    except Exception as error:
        try:
            _restore_targets(
                targets,
                original_flex_attention,
                original_create_block_mask,
                original_selector,
            )
        except Exception as rollback_error:
            raise NeuronAttentionContractError(
                "Neuron attention publication and rollback both failed"
            ) from rollback_error
        raise NeuronAttentionContractError(
            "failed to publish Neuron attention bindings"
        ) from error

    return installation


def install_neuron_attention() -> None:
    """Install the pinned Neuron attention callables once per rank process."""

    global _installation
    with _INSTALL_LOCK:
        targets = _resolve_targets()
        if _installation is not None:
            if (
                _installation.targets.attention_module is not targets.attention_module
                or _installation.targets.compile_module is not targets.compile_module
                or _installation.targets.flex_attention_class
                is not targets.flex_attention_class
            ):
                raise NeuronAttentionContractError(
                    "TorchTitan modules changed after Neuron attention installation"
                )
            _verify_installation(_installation)
            return

        existing = _adopt_existing(targets)
        if existing is not None:
            _installation = existing
            return

        try:
            compiled_flex_attention = targets.torch_module.compile(
                targets.flex_attention,
                backend="neuron",
                options=dict(_NEURON_FLEX_OPTIONS),
            )
            compiled_create_block_mask = targets.torch_module.compile(
                targets.create_block_mask,
                backend="neuron",
            )
        except Exception as error:
            raise NeuronAttentionContractError(
                "failed to construct Neuron attention callables"
            ) from error
        if not callable(compiled_flex_attention) or not callable(
            compiled_create_block_mask
        ):
            raise NeuronAttentionContractError(
                "torch.compile returned a non-callable attention binding"
            )

        selector = _build_selector(
            targets,
            compiled_flex_attention,
            compiled_create_block_mask,
        )
        _installation = _publish_installation(
            targets,
            compiled_flex_attention,
            compiled_create_block_mask,
            selector,
        )
