"""Contract tests for voicecoding.keyboard_listener.KeyboardHotkeyListener.

The listener exposes a single on_hotkey callback fired once per completed
combo press, so the record combo and the submit keycode each get their own
listener wired to the shared ``fired`` recorder. Events are driven through
the pynput-facing ``on_press``/``on_release`` entry points with fake key
objects carrying a ``.vk`` code: the latch logic in ``_process_key_event``
only returns a decision, and the callback invocation lives in ``on_press``.
All hotkey keys pressed fires once, releases reset the latch, repeat events
never refire. Debounce is set to zero so the tests isolate the latch
semantics from wall-clock timing; the debounce window itself is exercised by
the mouse-listener suite.
"""
from __future__ import annotations

import pytest

from conftest import FakeKeyCode

RECORD_COMBO = (162, 160, 32)  # VK_LCONTROL, VK_LSHIFT, VK_SPACE
SUBMIT_KEYCODE = 13  # VK_RETURN


def _make_listeners():
    from voicecoding.keyboard_listener import KeyboardHotkeyListener

    fired: dict[str, list[int]] = {"record": [], "submit": []}
    record_listener = KeyboardHotkeyListener(
        on_hotkey=lambda: fired["record"].append(1),
        keycodes=RECORD_COMBO,
        debounce_s=0.0,
    )
    submit_listener = KeyboardHotkeyListener(
        on_hotkey=lambda: fired["submit"].append(1),
        keycodes=(SUBMIT_KEYCODE,),
        debounce_s=0.0,
    )
    return record_listener, submit_listener, fired


def _press(listener, keycode: int) -> None:
    listener.on_press(FakeKeyCode(vk=keycode))


def _release(listener, keycode: int) -> None:
    listener.on_release(FakeKeyCode(vk=keycode))


def test_empty_keycodes_are_rejected() -> None:
    from voicecoding.keyboard_listener import KeyboardHotkeyListener

    with pytest.raises(ValueError):
        KeyboardHotkeyListener(on_hotkey=lambda: None, keycodes=())


def test_full_combo_fires_once_and_repeats_do_not_refire() -> None:
    record, _submit, fired = _make_listeners()

    _press(record, RECORD_COMBO[0])
    _press(record, RECORD_COMBO[1])
    _press(record, RECORD_COMBO[2])
    assert len(fired["record"]) == 1

    _press(record, RECORD_COMBO[2])  # repeat press of the last key
    assert len(fired["record"]) == 1


def test_release_resets_latch_and_combo_can_refire() -> None:
    record, _submit, fired = _make_listeners()

    _press(record, RECORD_COMBO[0])
    _press(record, RECORD_COMBO[1])
    _press(record, RECORD_COMBO[2])
    assert len(fired["record"]) == 1

    _release(record, RECORD_COMBO[1])  # release one key
    _press(record, RECORD_COMBO[1])  # press it again -> combo complete

    assert len(fired["record"]) == 2


def test_incomplete_combo_never_fires() -> None:
    record, _submit, fired = _make_listeners()

    _press(record, RECORD_COMBO[0])
    _press(record, RECORD_COMBO[1])
    _release(record, RECORD_COMBO[1])  # break the combo
    _press(record, RECORD_COMBO[2])  # third key without the second

    assert fired["record"] == []


def test_unrelated_keycodes_do_not_disturb_latch() -> None:
    record, submit, fired = _make_listeners()

    _press(record, 65)  # some other key
    _release(record, 65)
    _press(record, RECORD_COMBO[0])
    _press(record, RECORD_COMBO[1])
    _press(record, RECORD_COMBO[2])

    assert len(fired["record"]) == 1
    assert submit is not None and fired["submit"] == []


def test_submit_key_fires_submit_once_per_press() -> None:
    _record, submit, fired = _make_listeners()

    _press(submit, SUBMIT_KEYCODE)
    assert len(fired["submit"]) == 1

    _press(submit, SUBMIT_KEYCODE)  # repeat press
    assert len(fired["submit"]) == 1

    _release(submit, SUBMIT_KEYCODE)  # release resets
    _press(submit, SUBMIT_KEYCODE)
    assert len(fired["submit"]) == 2
    assert fired["record"] == []


def test_release_actions_never_fire_callbacks() -> None:
    record, submit, fired = _make_listeners()

    for keycode in RECORD_COMBO:
        _release(record, keycode)
    _release(submit, SUBMIT_KEYCODE)

    assert fired["record"] == []
    assert fired["submit"] == []


def test_key_events_without_virtual_keycode_are_ignored() -> None:
    record, _submit, fired = _make_listeners()

    record.on_press(FakeKeyCode(char="a"))  # no vk: not part of any combo
    record.on_release(FakeKeyCode(char="a"))

    assert fired["record"] == []

