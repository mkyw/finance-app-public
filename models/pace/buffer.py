"""Carry-forward buffer: accumulates paycheck surplus, never auto-shrinks.

``BufferState`` is a persistent-style value object (new instance per
update). ``carry_forward()`` adds only positive deltas — shortfalls are
surfaced elsewhere in the UI and do not automatically draw the buffer
down.

Language rules (CLAUDE.md Target Architecture):
  Never: saved, savings account, budget surplus.
  Always: buffer, carried forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date


@dataclass(frozen=True)
class BufferState:
    balance: float
    last_updated: date
    history: list[tuple[date, float]] = field(default_factory=list)


def carry_forward(
    discretionary_available: float,
    total_directed: float,
    current_buffer: BufferState,
) -> BufferState:
    """Return a new ``BufferState`` with positive carry added (shortfalls ignored).

    Args:
        discretionary_available: The paycheck's starting discretionary
            amount.
        total_directed: Total dollars directed/spent during the period.
        current_buffer: Existing buffer state.

    Returns:
        New ``BufferState``. The history entry is the actual delta
        applied to the balance — 0.0 when ``carry_amount`` was negative
        (so the log still reflects a period boundary, but the balance
        didn't change).
    """
    carry_amount = discretionary_available - total_directed
    applied = carry_amount if carry_amount > 0 else 0.0
    today = date.today()
    new_history = [*current_buffer.history, (today, applied)]
    return replace(
        current_buffer,
        balance=current_buffer.balance + applied,
        last_updated=today,
        history=new_history,
    )


def buffer_summary(state: BufferState) -> str:
    """Plain-language summary for the primary dashboard display.

    Tries to say something useful about trend across the history; falls
    back to a steady-state sentence when history is empty.
    """
    balance = int(round(state.balance))

    positive_entries = [delta for _, delta in state.history if delta > 0]
    if positive_entries:
        total_growth = int(round(sum(positive_entries)))
        periods = len(positive_entries)
        if periods == 1:
            return (
                f"Your buffer has grown ${total_growth} this paycheck."
            )
        return (
            f"Your buffer has grown ${total_growth} over "
            f"the last {periods} paychecks."
        )
    return f"Your buffer is steady at ${balance}."
