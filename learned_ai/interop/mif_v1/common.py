"""MIF 1.0 wire primitives shared by the independent adapter.

This module is intentionally self-contained.  In particular, it does not
import code from the MIF candidate reference implementation.  The constants
and algorithms below are transcribed from the frozen wire contract itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn


MIF_COMMIT = "7e45d5a3fa970a535ed6a8a8ff5981aba4b9c978"
MIF_SUITE_COMMIT = "3ee7e57c7d4c7208be91f62914f344a587fb0f70"
MIF_SUITE_JCS_SHA256 = (
    "sha256:81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f"
)
MIF_SUITE_RAW_SHA256 = (
    "sha256:088ca33234289b06d9276aa4c430758222aa85d61621dee7bef4bfc6dcc069a4"
)
MIF_RELEASE_MANIFEST_SHA256 = (
    "sha256:b721cb2bd22e404ef2cac1ff570c7ea4d0b4859c97cbaba94a8acce241a00057"
)
MIF_DIFFERENTIAL_LAUNCH_SHA256 = (
    "sha256:560ef369fde248bd96d3468a4336442db1d970ede04f488821509e69925fd48e"
)
MIF_LICENSE_SHA256 = (
    "sha256:c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
)
MIF_ENGLISH_SPEC_SHA256 = (
    "sha256:330e65145ceb26fe582e58b89405d87bd73e8be200b476aef82c0ee27731d995"
)
MIF_CHINESE_SPEC_SHA256 = (
    "sha256:9cc06abb57425e2bc2e26432b6da53abe503e9b5415ea0b4f854f19f68722cc1"
)
MIF_INDEX_SHA256 = (
    "sha256:5acbb714bed77e24eaac72fa5f24d2e54d1e17aaf568a8b60718c840281a6541"
)
MIF_EXECUTABLE_CORPUS_SHA256 = (
    "sha256:350b7ff02772e820a57431e11c4e2f15a874d0779fb6e7afb01e9b16f6992741"
)
MIF_ADAPTER_PROTOCOL_SHA256 = (
    "sha256:253c1d201ea1db625e0c534da445ca4ecaa0b07597dfc7dbf59fbd6adf89874f"
)
MIF_SMOKE_CORPUS_SHA256 = (
    "sha256:a6d292f4d19381172fbc19f89d3ee42145a6d5533d6d81fd719394e25342bb53"
)
MIF_DETERMINISTIC_CORPUS_SHA256 = (
    "sha256:d11317a090300f8a47f77afed647bdbd236dcdb1996c0147a81c874fa39dfd82"
)
PROTOCOL = "MIF-INTEROP/1"
MAX_EXACT_INTEGER = 9_007_199_254_740_991
MAX_INTEROP_REQUEST_BYTES = 16_777_216
MAX_EVENTS = 100_000
MAX_REPETITION_ENTRIES = 100_000

POINTS = (
    "a7",
    "d7",
    "g7",
    "g4",
    "g1",
    "d1",
    "a1",
    "a4",
    "b6",
    "d6",
    "f6",
    "f4",
    "f2",
    "d2",
    "b2",
    "b4",
    "c5",
    "d5",
    "e5",
    "e4",
    "e3",
    "d3",
    "c3",
    "c4",
)
POINT_INDEX = {point: index for index, point in enumerate(POINTS)}

ORTHOGONAL_LINES = (
    ("a7", "d7", "g7"),
    ("g7", "g4", "g1"),
    ("g1", "d1", "a1"),
    ("a1", "a4", "a7"),
    ("b6", "d6", "f6"),
    ("f6", "f4", "f2"),
    ("f2", "d2", "b2"),
    ("b2", "b4", "b6"),
    ("c5", "d5", "e5"),
    ("e5", "e4", "e3"),
    ("e3", "d3", "c3"),
    ("c3", "c4", "c5"),
    ("d7", "d6", "d5"),
    ("g4", "f4", "e4"),
    ("d1", "d2", "d3"),
    ("a4", "b4", "c4"),
)
DIAGONAL_LINES = ORTHOGONAL_LINES + (
    ("a7", "b6", "c5"),
    ("g7", "f6", "e5"),
    ("g1", "f2", "e3"),
    ("a1", "b2", "c3"),
)

ORTHOGONAL_EDGES = (
    ("a7", "d7"),
    ("d7", "g7"),
    ("g7", "g4"),
    ("g4", "g1"),
    ("g1", "d1"),
    ("d1", "a1"),
    ("a1", "a4"),
    ("a4", "a7"),
    ("b6", "d6"),
    ("d6", "f6"),
    ("f6", "f4"),
    ("f4", "f2"),
    ("f2", "d2"),
    ("d2", "b2"),
    ("b2", "b4"),
    ("b4", "b6"),
    ("c5", "d5"),
    ("d5", "e5"),
    ("e5", "e4"),
    ("e4", "e3"),
    ("e3", "d3"),
    ("d3", "c3"),
    ("c3", "c4"),
    ("c4", "c5"),
    ("d7", "d6"),
    ("d6", "d5"),
    ("g4", "f4"),
    ("f4", "e4"),
    ("d1", "d2"),
    ("d2", "d3"),
    ("a4", "b4"),
    ("b4", "c4"),
)
DIAGONAL_EDGES = ORTHOGONAL_EDGES + (
    ("a7", "b6"),
    ("b6", "c5"),
    ("g7", "f6"),
    ("f6", "e5"),
    ("g1", "f2"),
    ("f2", "e3"),
    ("a1", "b2"),
    ("b2", "c3"),
)

TRANSFORM_IDS = (
    "i",
    "r90ccw",
    "r180",
    "r90cw",
    "mirror-v",
    "mirror-h",
    "mirror-main",
    "mirror-anti",
)

IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,62}$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class MifError(Exception):
    """One deterministic MIF diagnostic."""

    category: str
    code: str
    message: str = ""
    instance_path: str | None = None
    event_seq: int | None = None
    expected: Any = None
    actual: Any = None
    include_expected: bool = False
    include_actual: bool = False
    resource_limit: dict[str, int | str] | None = None

    def as_diagnostics(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "category": self.category,
            "code": self.code,
        }
        if self.instance_path is not None:
            error["instancePath"] = self.instance_path
        if self.event_seq is not None:
            error["eventSeq"] = self.event_seq
        if self.include_expected:
            error["expected"] = self.expected
        if self.include_actual:
            error["actual"] = self.actual
        if self.resource_limit is not None:
            error["resourceLimit"] = dict(self.resource_limit)
        if self.message:
            error["message"] = self.message
        return {"format": "MIFDIAG/1.0", "errors": [error]}


def fail(
    category: str,
    code: str,
    message: str = "",
    *,
    instance_path: str | None = None,
    event_seq: int | None = None,
    expected: Any = None,
    actual: Any = None,
    include_expected: bool = False,
    include_actual: bool = False,
    resource_limit: dict[str, int | str] | None = None,
) -> NoReturn:
    raise MifError(
        category,
        code,
        message,
        instance_path,
        event_seq,
        expected,
        actual,
        include_expected,
        include_actual,
        resource_limit,
    )


def enforce_resource_limit(name: str, actual: int, limit: int) -> None:
    """Reject semantic truncation when a published adapter limit is exceeded."""

    if actual > limit:
        fail(
            "resource",
            "resource-limit",
            f"{name} resource limit exceeded",
            resource_limit={"name": name, "limit": limit, "actual": actual},
        )


def _reject_surrogates(value: str) -> None:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        fail("syntax", "invalid-unicode-scalar", "unpaired surrogate is not I-JSON")


def validate_ijson(value: Any, *, path: str = "") -> None:
    """Validate the MIF I-JSON subset recursively."""

    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        _reject_surrogates(value)
        return
    if isinstance(value, int):
        if not 0 <= value <= MAX_EXACT_INTEGER:
            fail(
                "syntax",
                "integer-out-of-range",
                "integer is outside the exact MIF range",
                instance_path=path,
            )
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            fail("syntax", "invalid-number", "non-finite JSON number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_ijson(item, path=f"{path}/{index}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                fail("syntax", "non-string-member", "JSON member name is not text")
            _reject_surrogates(key)
            token = key.replace("~", "~0").replace("/", "~1")
            validate_ijson(item, path=f"{path}/{token}")
        return
    fail("syntax", "non-json-value", f"unsupported JSON value: {type(value).__name__}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(
                "syntax",
                "duplicate-member-after-unescape",
                f"duplicate JSON member {key!r}",
            )
        result[key] = value
    return result


def parse_ijson(raw: bytes) -> Any:
    """Parse UTF-8 I-JSON while preserving duplicate-member detection."""

    if raw.startswith(b"\xef\xbb\xbf"):
        fail("syntax", "utf8-bom", "a UTF-8 BOM is forbidden")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        fail("syntax", "invalid-utf8", str(exc))
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: fail(
                "syntax", "invalid-number", f"invalid number {token}"
            ),
        )
    except MifError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        fail("syntax", "invalid-json", str(exc))
    validate_ijson(value)
    return value


def _utf16_sort_key(value: str) -> bytes:
    return value.encode("utf-16-be")


def _serialize_number(value: float) -> str:
    """Serialize a finite binary64 using RFC 8785/ECMAScript spelling.

    Python's ``repr`` already supplies the shortest round-tripping decimal.
    The remaining work is the ECMAScript choice between fixed and exponent
    notation plus exponent normalization.  MIF semantic fields use exact
    integers; this path exists for permitted annotation values.
    """

    if value == 0:
        return "0"
    negative = value < 0
    text = repr(abs(value)).lower()
    if "e" in text:
        coefficient, exponent_text = text.split("e", 1)
        exponent = int(exponent_text)
        digits = coefficient.replace(".", "")
        decimal_position = coefficient.find(".")
        if decimal_position < 0:
            decimal_position = len(coefficient)
        adjusted = decimal_position + exponent
        if 1e-6 <= abs(value) < 1e21:
            if adjusted <= 0:
                text = "0." + ("0" * -adjusted) + digits
            elif adjusted >= len(digits):
                text = digits + ("0" * (adjusted - len(digits)))
            else:
                text = digits[:adjusted] + "." + digits[adjusted:]
        else:
            coefficient = coefficient.rstrip("0").rstrip(".")
            sign = "+" if exponent >= 0 else "-"
            text = f"{coefficient}e{sign}{abs(exponent)}"
    elif text.endswith(".0"):
        text = text[:-2]
    return f"-{text}" if negative else text


def _jcs_text(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _serialize_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(_jcs_text(item) for item in value) + "]"
    if isinstance(value, Mapping):
        members = []
        for key in sorted(value, key=_utf16_sort_key):
            members.append(
                f"{json.dumps(key, ensure_ascii=False)}:{_jcs_text(value[key])}"
            )
        return "{" + ",".join(members) + "}"
    fail("syntax", "non-json-value", f"unsupported JSON value: {type(value).__name__}")


def jcs_bytes(value: Any) -> bytes:
    validate_ijson(value)
    return _jcs_text(value).encode("utf-8")


def sha256_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(jcs_bytes(value)).hexdigest()


def require_object(
    value: Any,
    *,
    required: set[str] | frozenset[str],
    optional: set[str] | frozenset[str] = frozenset(),
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail("syntax", "object-required", f"{context} must be an object")
    actual = set(value)
    missing = sorted(required - actual)
    extra = sorted(actual - required - optional)
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unknown " + ", ".join(extra))
        fail("syntax", "closed-object-mismatch", f"{context}: {'; '.join(details)}")
    return value


def require_identifier(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None:
        fail("syntax", "invalid-identifier", f"invalid {context}")
    return value


def require_digest(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
        fail("syntax", "invalid-digest", f"invalid {context}")
    return value


def other(player: str) -> str:
    return "b" if player == "w" else "w"


def piece_for(player: str) -> str:
    return player.upper()


def transform_coordinate(coordinate: str, transform: str) -> str:
    if coordinate not in POINT_INDEX:
        fail("syntax", "invalid-coordinate", f"unknown coordinate {coordinate!r}")
    if transform not in TRANSFORM_IDS:
        fail("unsupported", "unsupported-profile", f"unknown transform {transform!r}")
    x = ord(coordinate[0]) - ord("d")
    y = int(coordinate[1]) - 4
    if transform == "i":
        tx, ty = x, y
    elif transform == "r90ccw":
        tx, ty = -y, x
    elif transform == "r180":
        tx, ty = -x, -y
    elif transform == "r90cw":
        tx, ty = y, -x
    elif transform == "mirror-v":
        tx, ty = -x, y
    elif transform == "mirror-h":
        tx, ty = x, -y
    elif transform == "mirror-main":
        tx, ty = y, x
    else:
        tx, ty = -y, -x
    result = f"{chr(ord('d') + tx)}{4 + ty}"
    if result not in POINT_INDEX:
        fail("inconsistent", "invalid-coordinate-transform", result)
    return result


def transform_board(board: str, transform: str, *, slashes: bool = True) -> str:
    compact = board.replace("/", "")
    if len(compact) != 24:
        fail("syntax", "invalid-board", "board must contain 24 points")
    result = ["."] * 24
    for index, value in enumerate(compact):
        destination = transform_coordinate(POINTS[index], transform)
        result[POINT_INDEX[destination]] = value
    joined = "".join(result)
    if not slashes:
        return joined
    return f"{joined[:8]}/{joined[8:16]}/{joined[16:]}"


def transform_mask(mask: int, transform: str) -> int:
    result = 0
    for index, coordinate in enumerate(POINTS):
        if mask & (1 << index):
            result |= 1 << POINT_INDEX[transform_coordinate(coordinate, transform)]
    return result


def topology_lines(topology: str) -> tuple[tuple[str, str, str], ...]:
    if topology == "mill24-orthogonal-v1":
        return ORTHOGONAL_LINES
    if topology == "mill24-diagonal-v1":
        return DIAGONAL_LINES
    fail("unsupported", "unsupported-profile", f"unsupported topology {topology!r}")


def topology_edges(topology: str) -> tuple[tuple[str, str], ...]:
    if topology == "mill24-orthogonal-v1":
        return ORTHOGONAL_EDGES
    if topology == "mill24-diagonal-v1":
        return DIAGONAL_EDGES
    fail("unsupported", "unsupported-profile", f"unsupported topology {topology!r}")


def transform_line_bits(value: int, topology: str, transform: str) -> int:
    lines = topology_lines(topology)
    lookup = {frozenset(line): index for index, line in enumerate(lines)}
    result = 0
    for index, line in enumerate(lines):
        if value & (1 << index):
            mapped = frozenset(transform_coordinate(point, transform) for point in line)
            mapped_index = lookup.get(mapped)
            if mapped_index is None:
                fail("inconsistent", "invalid-line-transform", "line left topology")
            result |= 1 << mapped_index
    return result


def deep_copy_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: deep_copy_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [deep_copy_json(item) for item in value]
    return value
