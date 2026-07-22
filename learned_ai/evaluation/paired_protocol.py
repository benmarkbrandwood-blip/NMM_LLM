"""Frozen fixed-N paired evaluation with recomputable immutable evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch

from game.board import BoardState
from game.game_engine import GameEngine
from game.rules import is_terminal
from learned_ai.delivery.model_bundle import load_bundle_model, verify_model_bundle
from learned_ai.models.scaffolded_encoder import encode_position_with_lookahead
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256


EVALUATION_SPEC_SCHEMA = "nmm.paired-evaluation.v1"
GAME_RECORD_SCHEMA = "nmm.evaluation-game.v1"


class EvaluationError(RuntimeError):
    """Raised when an evaluation contract or record set is invalid."""


@dataclass(frozen=True)
class EvaluationSpec:
    evaluation_id: str
    candidate_bundle: str
    baseline_bundle: str
    start_positions: tuple[str, ...]
    pairs: int
    seed: int
    work_budget: dict[str, int]
    max_ply: int
    rules_version: str
    confidence_z: float
    acceptance_margin: float
    rejection_margin: float
    runtime: dict[str, str]
    spec_identity: str = ""

    _FIELDS: ClassVar[set[str]] = {
        "schema_version", "evaluation_id", "candidate_bundle", "baseline_bundle",
        "start_positions", "start_positions_sha256", "pairs", "seed", "work_budget",
        "max_ply", "adjudication", "rules_version", "confidence", "thresholds",
        "runtime", "spec_identity",
    }

    def __post_init__(self) -> None:
        if not self.evaluation_id or len(self.candidate_bundle) != 64 or len(self.baseline_bundle) != 64:
            raise EvaluationError("evaluation and bundle identities are invalid")
        if not self.start_positions or self.pairs <= 0 or self.max_ply <= 0:
            raise EvaluationError("evaluation size and start positions must be positive")
        for fen in self.start_positions:
            BoardState.from_fen_string(fen)
        if len(set(self.start_positions)) != len(self.start_positions):
            raise EvaluationError("start positions must be unique")
        if self.pairs > len(self.start_positions):
            raise EvaluationError("pair count cannot exceed unique starts")
        if set(self.work_budget) != {"lookahead_rollouts_per_move"}:
            raise EvaluationError("work budget must use fixed lookahead rollouts")
        if self.work_budget["lookahead_rollouts_per_move"] != 0:
            raise EvaluationError("v1 runner supports the frozen zero-rollout policy budget")
        expected = canonical_sha256(self._identity_body())
        if self.spec_identity and self.spec_identity != expected:
            raise EvaluationError("evaluation spec identity mismatch")
        object.__setattr__(self, "spec_identity", expected)

    def _identity_body(self) -> dict[str, Any]:
        return {
            "schema_version": EVALUATION_SPEC_SCHEMA,
            "evaluation_id": self.evaluation_id,
            "candidate_bundle": self.candidate_bundle,
            "baseline_bundle": self.baseline_bundle,
            "start_positions": list(self.start_positions),
            "start_positions_sha256": canonical_sha256(list(self.start_positions)),
            "pairs": self.pairs,
            "seed": self.seed,
            "work_budget": self.work_budget,
            "max_ply": self.max_ply,
            "adjudication": {"max_ply": "draw", "terminal_rules": "game.rules.is_terminal"},
            "rules_version": self.rules_version,
            "confidence": {"method": "normal-interval-on-pair-score-difference", "z": self.confidence_z},
            "thresholds": {"accept_lower_gt": self.acceptance_margin, "reject_upper_lt": -self.rejection_margin},
            "runtime": self.runtime,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_body(), "spec_identity": self.spec_identity}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationSpec":
        if not isinstance(value, dict) or set(value) != cls._FIELDS:
            raise EvaluationError("evaluation spec fields are unknown or incomplete")
        if value["schema_version"] != EVALUATION_SPEC_SCHEMA:
            raise EvaluationError("unsupported evaluation spec schema")
        if value["start_positions_sha256"] != canonical_sha256(value["start_positions"]):
            raise EvaluationError("start-position corpus identity mismatch")
        if value["adjudication"] != {"max_ply": "draw", "terminal_rules": "game.rules.is_terminal"}:
            raise EvaluationError("unsupported adjudication contract")
        return cls(
            evaluation_id=value["evaluation_id"], candidate_bundle=value["candidate_bundle"],
            baseline_bundle=value["baseline_bundle"], start_positions=tuple(value["start_positions"]),
            pairs=value["pairs"], seed=value["seed"], work_budget=value["work_budget"],
            max_ply=value["max_ply"], rules_version=value["rules_version"],
            confidence_z=float(value["confidence"]["z"]),
            acceptance_margin=float(value["thresholds"]["accept_lower_gt"]),
            rejection_margin=float(-value["thresholds"]["reject_upper_lt"]),
            runtime=value["runtime"], spec_identity=value["spec_identity"],
        )


def freeze_evaluation_spec(path: str | Path, spec: EvaluationSpec) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"evaluation spec exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(spec.to_dict()))
    os.replace(temporary, target)


def load_evaluation_spec(path: str | Path) -> EvaluationSpec:
    return EvaluationSpec.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class _BundlePolicy:
    def __init__(self, model: torch.nn.Module, device: str) -> None:
        self.model = model
        self.device = device

    def choose_move(self, board: BoardState) -> dict[str, Any]:
        encoded = encode_position_with_lookahead(
            board, board.turn, lookahead_advisor=None, lookahead_dim=72
        )
        if encoded is None or not encoded.legal_moves:
            return {}
        features = torch.as_tensor(encoded.feat_matrix, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            index = int(torch.argmax(self.model.policy_logits(features)).item())
        return encoded.legal_moves[index]


def _game_id(spec: EvaluationSpec, pair: int, game: int) -> str:
    return "eval-game:" + canonical_sha256(
        {"spec": spec.spec_identity, "pair": pair, "game": game}
    )


def _game_seed(spec: EvaluationSpec, pair: int, game: int) -> int:
    digest = hashlib.sha256(f"{spec.seed}:{pair}:{game}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _expected_game_identity(
    spec: EvaluationSpec,
    pair: int,
    game: int,
) -> dict[str, Any]:
    return {
        "pair": pair,
        "game": game,
        "game_id": _game_id(spec, pair, game),
        "seed": _game_seed(spec, pair, game),
        "start_fen": spec.start_positions[pair % len(spec.start_positions)],
        "candidate_color": "W" if game == 0 else "B",
    }


def _load_partial_prefix(
    spec: EvaluationSpec,
    path: Path,
) -> tuple[int, str | None]:
    """Validate and return the completed length and tail hash of a strict prefix."""
    completed = 0
    previous_hash: str | None = None
    expected_games = spec.pairs * 2
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise EvaluationError(
                    f"malformed game record at line {line_number}: missing newline"
                )
            try:
                wrapper = json.loads(line)
                record = wrapper["record"]
                record_hash = wrapper["record_sha256"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise EvaluationError(
                    f"malformed game record at line {line_number}"
                ) from exc
            if not isinstance(record, dict) or not isinstance(record_hash, str):
                raise EvaluationError(f"malformed game record at line {line_number}")
            if completed >= expected_games:
                raise EvaluationError(
                    f"partial ledger has an unexpected game at line {line_number}"
                )
            if (
                record.get("schema_version") != GAME_RECORD_SCHEMA
                or record.get("spec_identity") != spec.spec_identity
            ):
                raise EvaluationError(f"wrong schema or spec at line {line_number}")
            if (
                canonical_sha256(record) != record_hash
                or record.get("previous_record_sha256") != previous_hash
            ):
                raise EvaluationError(
                    f"record integrity chain failed at line {line_number}"
                )
            pair, game = divmod(completed, 2)
            expected_identity = _expected_game_identity(spec, pair, game)
            if any(record.get(key) != value for key, value in expected_identity.items()):
                raise EvaluationError(
                    f"partial ledger is not the expected ordered prefix at line {line_number}"
                )
            if record.get("complete") is not True or record.get(
                "candidate_score"
            ) not in (0.0, 0.5, 1.0):
                raise EvaluationError(
                    f"incomplete or invalid game at line {line_number}"
                )
            completed += 1
            previous_hash = record_hash
    return completed, previous_hash


def run_paired_evaluation(
    spec_path: str | Path,
    candidate_path: str | Path,
    baseline_path: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run candidate and baseline serially with roles swapped within each pair."""
    spec = load_evaluation_spec(spec_path)
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"evaluation records exist: {target}")
    partial = Path(f"{target}.partial")
    if partial.exists():
        completed_games, previous_hash = _load_partial_prefix(spec, partial)
        open_mode = "a"
    else:
        completed_games = 0
        previous_hash = None
        open_mode = "x"
    candidate_model, candidate_manifest = load_bundle_model(candidate_path, device=device)
    baseline_model, baseline_manifest = load_bundle_model(baseline_path, device=device)
    if candidate_manifest["bundle_identity"] != spec.candidate_bundle or baseline_manifest["bundle_identity"] != spec.baseline_bundle:
        raise EvaluationError("bundle paths do not match the frozen evaluation spec")
    target.parent.mkdir(parents=True, exist_ok=True)
    with partial.open(open_mode, encoding="utf-8", newline="\n") as handle:
        for pair in range(spec.pairs):
            fen = spec.start_positions[pair % len(spec.start_positions)]
            for game in range(2):
                game_index = pair * 2 + game
                if game_index < completed_games:
                    continue
                identity = _expected_game_identity(spec, pair, game)
                candidate_color = identity["candidate_color"]
                policies = {
                    candidate_color: _BundlePolicy(candidate_model, device),
                    "B" if candidate_color == "W" else "W": _BundlePolicy(baseline_model, device),
                }
                engine = GameEngine(human_color=None)
                engine.board = BoardState.from_fen_string(fen)
                terminal_reason = "max_ply"
                for ply in range(spec.max_ply):
                    terminal, winner = is_terminal(engine.board)
                    if terminal:
                        terminal_reason = "rules_terminal"
                        break
                    move = policies[engine.board.turn].choose_move(engine.board)
                    if not move:
                        winner = "B" if engine.board.turn == "W" else "W"
                        terminal_reason = "no_legal_move"
                        break
                    engine.apply_move(move)
                    if engine.finished:
                        winner = engine.winner
                        terminal_reason = engine.draw_reason or "rules_terminal"
                        break
                else:
                    winner = None
                    ply = spec.max_ply - 1
                score = 0.5 if winner is None else (1.0 if winner == candidate_color else 0.0)
                record = {
                    "schema_version": GAME_RECORD_SCHEMA, "spec_identity": spec.spec_identity,
                    **identity, "winner": winner,
                    "candidate_score": score, "ply": ply + 1, "terminal_reason": terminal_reason,
                    "complete": True, "previous_record_sha256": previous_hash,
                }
                record_hash = canonical_sha256(record)
                handle.write(json.dumps({"record": record, "record_sha256": record_hash}, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                previous_hash = record_hash
    result = recompute_evaluation(spec_path, partial)
    if target.exists():
        raise FileExistsError(f"evaluation records exist: {target}")
    os.replace(partial, target)
    return result


def recompute_evaluation(spec_path: str | Path, records_path: str | Path) -> dict[str, Any]:
    """Validate a complete raw ledger and recompute the frozen decision."""
    spec = load_evaluation_spec(spec_path)
    records: dict[str, dict[str, Any]] = {}
    previous_hash: str | None = None
    with Path(records_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                wrapper = json.loads(line)
                record = wrapper["record"]
                record_hash = wrapper["record_sha256"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise EvaluationError(f"malformed game record at line {line_number}") from exc
            if record.get("schema_version") != GAME_RECORD_SCHEMA or record.get("spec_identity") != spec.spec_identity:
                raise EvaluationError(f"wrong schema or spec at line {line_number}")
            if canonical_sha256(record) != record_hash or record.get("previous_record_sha256") != previous_hash:
                raise EvaluationError(f"record integrity chain failed at line {line_number}")
            game_id = record.get("game_id")
            if game_id in records:
                raise EvaluationError(f"duplicate game ID: {game_id}")
            if not record.get("complete") or record.get("candidate_score") not in (0.0, 0.5, 1.0):
                raise EvaluationError(f"incomplete or invalid game: {game_id}")
            records[game_id] = record
            previous_hash = record_hash
    expected = {_game_id(spec, pair, game) for pair in range(spec.pairs) for game in range(2)}
    if set(records) != expected:
        raise EvaluationError(f"game set differs; missing={sorted(expected - set(records))}, unexpected={sorted(set(records) - expected)}")
    pair_differences = []
    wins = draws = losses = 0
    for pair in range(spec.pairs):
        scores = [records[_game_id(spec, pair, game)]["candidate_score"] for game in range(2)]
        pair_differences.append(sum(scores) - 1.0)
        wins += scores.count(1.0)
        draws += scores.count(0.5)
        losses += scores.count(0.0)
    mean = sum(pair_differences) / len(pair_differences)
    if len(pair_differences) > 1:
        variance = sum((item - mean) ** 2 for item in pair_differences) / (len(pair_differences) - 1)
        half_width = spec.confidence_z * math.sqrt(variance / len(pair_differences))
        lower, upper = mean - half_width, mean + half_width
        decision = "accepted" if lower > spec.acceptance_margin else "rejected" if upper < -spec.rejection_margin else "inconclusive"
        interval: list[float | None] = [lower, upper]
    else:
        decision = "inconclusive"
        interval = [None, None]
    result = {
        "schema_version": "nmm.evaluation-result.v1", "spec_identity": spec.spec_identity,
        "records_sha256": hashlib.sha256(Path(records_path).read_bytes()).hexdigest(),
        "games": len(records), "wins": wins, "draws": draws, "losses": losses,
        "pair_score_difference_mean": mean, "confidence_interval": interval,
        "decision": decision,
    }
    result["result_identity"] = canonical_sha256(result)
    return result
