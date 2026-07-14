"""Regression tests for the ControlAdapter contract (BUG 1 / BUG 2 / BUG 3 fixes).

These pin the behaviours that the session's three bug fixes established, so a future
refactor that breaks them fails loudly instead of silently corrupting training:

  * forward() returns the tuple (control_signal, style_tokens) with the documented shapes
  * an invalid (zero-filled) modality is masked to EXACTLY zero before gating, so its
    input cannot change the output and its gate logit receives ZERO gradient (BUG 1)
  * style tokens are EXACTLY zero at init via the zero-initialised style_zero proj (BUG 3)
  * there are 5 gate logits / weights — one per spatial modality, style is not gated (BUG 3)
  * the style-append cross-attention hook fails safe (no-op) on an unexpected context
    layout and respects WAN's text_len cap (BUG 3) — skipped if Wan2.2 isn't importable

No pytest dependency (the repo has no test runner): run it directly.

    python tests/test_adapter_contract.py

Exits non-zero if any check fails.
"""

import sys
from pathlib import Path

import torch

# Match tests/test-allcontrols.py's import style: repo root on path, `from src.models...`.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.control_adapter import (
    ControlAdapter,
    MODALITY_ORDER,
    SPATIAL_MODALITIES,
)


# ── small, fast adapter config (real channel dim 256, everything else shrunk) ──────────
CONTROL_DIM = 256          # fixed: equals the encoded channel count the adapter pools over
HIDDEN_DIM = 16
DIT_DIM = 24
STYLE_DIM = 768
TEXT_DIM = 20
NUM_STYLE_TOKENS = 3
B, T, HW = 2, 2, 16        # spatial already 16×16 so adaptive pool is a no-op


def _make_adapter():
    a = ControlAdapter(
        control_dim=CONTROL_DIM,
        hidden_dim=HIDDEN_DIM,
        dit_dim=DIT_DIM,
        style_dim=STYLE_DIM,
        text_dim=TEXT_DIM,
        num_style_tokens=NUM_STYLE_TOKENS,
    )
    a.eval()  # disable dropout so forward() is deterministic
    return a


def _make_controls(requires_grad=False):
    controls = {}
    for m in SPATIAL_MODALITIES:
        controls[f'{m}_encoded'] = torch.randn(
            B, CONTROL_DIM, T, HW, HW, requires_grad=requires_grad
        )
    controls['style_encoded'] = torch.randn(B, STYLE_DIM, requires_grad=requires_grad)
    return controls


def _all_valid(value=1.0):
    return {f'{m}_encoded': torch.full((B,), value) for m in MODALITY_ORDER}


# ── tests ─────────────────────────────────────────────────────────────────────────────
def test_constants():
    assert MODALITY_ORDER == ['depth', 'mask', 'motion', 'pose', 'sketch', 'style'], \
        f"MODALITY_ORDER changed: {MODALITY_ORDER}"
    assert SPATIAL_MODALITIES == ['depth', 'mask', 'motion', 'pose', 'sketch'], \
        f"SPATIAL_MODALITIES changed: {SPATIAL_MODALITIES}"
    assert 'style' not in SPATIAL_MODALITIES


def test_forward_returns_tuple_shapes():
    a = _make_adapter()
    out = a(_make_controls())
    assert isinstance(out, tuple) and len(out) == 2, \
        f"forward must return (control_signal, style_tokens), got {type(out)}"
    control_signal, style_tokens = out
    # control_signal: (B, T·16·16, dit_dim)
    assert control_signal.shape == (B, T * 16 * 16, DIT_DIM), \
        f"control_signal shape {tuple(control_signal.shape)} != {(B, T * 16 * 16, DIT_DIM)}"
    # style_tokens: (B, num_style_tokens, text_dim)
    assert style_tokens.shape == (B, NUM_STYLE_TOKENS, TEXT_DIM), \
        f"style_tokens shape {tuple(style_tokens.shape)} != {(B, NUM_STYLE_TOKENS, TEXT_DIM)}"


def test_style_tokens_zero_at_init():
    a = _make_adapter()
    _, style_tokens = a(_make_controls(), _all_valid())
    assert torch.equal(style_tokens, torch.zeros_like(style_tokens)), \
        "style_tokens must be EXACTLY zero at init (style_zero is zero-initialised)"


def test_invalid_modality_does_not_change_output():
    """An invalid spatial modality is masked to zero before fusion, so its input value
    cannot affect control_signal at all (BUG 1)."""
    a = _make_adapter()
    valid = _all_valid()
    valid['motion_encoded'] = torch.zeros(B)  # mark motion invalid

    controls = _make_controls()
    sig_a, _ = a(controls, valid)

    # Change ONLY the invalid modality's input; output must be bit-identical.
    controls['motion_encoded'] = torch.randn_like(controls['motion_encoded'])
    sig_b, _ = a(controls, valid)

    assert torch.equal(sig_a, sig_b), \
        "changing an invalid modality's input changed control_signal — masking is leaking"


def test_invalid_modality_zero_gate_grad():
    """The gate logit of an invalid modality receives zero gradient; a valid one does not
    (BUG 1: masked before gating => d/dgate of (proj*0*gate) == 0)."""
    a = _make_adapter()
    valid = _all_valid()
    valid['motion_encoded'] = torch.zeros(B)  # motion invalid, everything else valid

    control_signal, _ = a(_make_controls(), valid)
    a.zero_grad()
    control_signal.sum().backward()

    grad = a.modality_gates.grad
    assert grad is not None, "modality_gates received no gradient at all"

    motion_idx = SPATIAL_MODALITIES.index('motion')
    assert grad[motion_idx].abs().item() == 0.0, \
        f"invalid modality 'motion' gate grad should be 0, got {grad[motion_idx].item()}"

    # At least one valid modality should have a non-zero gate gradient (sanity: grads flow).
    valid_grads = [grad[i].abs().item()
                   for i, m in enumerate(SPATIAL_MODALITIES) if m != 'motion']
    assert any(g > 0 for g in valid_grads), \
        "no valid modality received gate gradient — backward path is broken"


def test_five_gate_logits():
    a = _make_adapter()
    assert a.modality_gates.numel() == 5, \
        f"expected 5 gate logits (one per spatial modality), got {a.modality_gates.numel()}"

    weights = a.get_modality_weights()
    logits = a.get_modality_logits()
    assert list(weights.keys()) == SPATIAL_MODALITIES, f"weight keys: {list(weights.keys())}"
    assert list(logits.keys()) == SPATIAL_MODALITIES, f"logit keys: {list(logits.keys())}"
    assert len(weights) == 5 and len(logits) == 5
    assert 'style' not in weights and 'style' not in logits, "style must not be gated"


def test_wrong_spatial_count_raises():
    a = _make_adapter()
    controls = _make_controls()
    del controls['motion_encoded']  # 4 spatial + style → should raise
    raised = False
    try:
        a(controls)
    except ValueError:
        raised = True
    assert raised, "adapter must raise ValueError when spatial modality count != 5"


def test_style_injection_hook_failsafe():
    """The cross-attention style hook (BUG 3) must:
      * no-op when no style tokens are set,
      * append style tokens to a well-formed context list,
      * leave an unexpected-layout context (e.g. CFG-doubled batch) untouched,
      * respect WAN's text_len cap by trimming before appending.

    Skipped if Wan2.2 isn't importable (this dev box may lack the WAN package).
    """
    try:
        from src.models.wan_controllable import ControllableWAN
    except Exception as e:  # noqa: BLE001 - WAN not installed here; coverage runs where it is
        print(f"  SKIP test_style_injection_hook_failsafe (cannot import WAN: {str(e)[:80]})")
        return

    hook = ControllableWAN._style_injection_hook  # unbound; only touches self._style_tokens

    class _Self:
        pass

    class _Mod:
        pass

    n_style, td, L = NUM_STYLE_TOKENS, TEXT_DIM, 7

    # (a) no style tokens → no-op (returns None, hook leaves args/kwargs untouched).
    s = _Self(); s._style_tokens = None
    mod = _Mod(); mod.text_len = 512
    assert hook(s, mod, (), {'context': [torch.randn(L, td)]}) is None

    # (b) well-formed context list (len == batch) → tokens appended.
    s = _Self(); s._style_tokens = torch.randn(1, n_style, td)
    mod = _Mod(); mod.text_len = 512
    context = [torch.randn(L, td)]
    new_args, new_kwargs = hook(s, mod, (), {'context': context})
    assert new_kwargs['context'][0].shape[0] == L + n_style, "style tokens were not appended"

    # (c) unexpected layout (context length != batch, e.g. CFG-doubled) → left untouched.
    s = _Self(); s._style_tokens = torch.randn(1, n_style, td)
    mod = _Mod(); mod.text_len = 512
    context = [torch.randn(L, td), torch.randn(L, td)]  # 2 != batch 1
    _, new_kwargs = hook(s, mod, (), {'context': context})
    assert [c.shape[0] for c in new_kwargs['context']] == [L, L], \
        "hook must not modify an unexpected-layout context"

    # (d) text_len cap → trim trailing padding so total length stays within the cap.
    s = _Self(); s._style_tokens = torch.randn(1, n_style, td)
    mod = _Mod(); mod.text_len = L  # appending n_style would overflow → must trim
    context = [torch.randn(L, td)]
    _, new_kwargs = hook(s, mod, (), {'context': context})
    assert new_kwargs['context'][0].shape[0] == L, \
        f"hook must cap context at text_len={L}, got {new_kwargs['context'][0].shape[0]}"


TESTS = [
    test_constants,
    test_forward_returns_tuple_shapes,
    test_style_tokens_zero_at_init,
    test_invalid_modality_does_not_change_output,
    test_invalid_modality_zero_gate_grad,
    test_five_gate_logits,
    test_wrong_spatial_count_raises,
    test_style_injection_hook_failsafe,
]


def main():
    torch.manual_seed(0)
    failures = 0
    for t in TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001 - report unexpected errors as failures
            failures += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
