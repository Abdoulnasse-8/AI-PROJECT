from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from player import BasePlayer, Move


WIN_SCORE = 10**9
TIME_BUDGET = 0.92
MAX_DEPTH = 5
BOARD_SIZE_HINT = 16
CONNECT_N = 5


@dataclass
class _StateView:
    grid: List[List[int]]
    rows: int
    cols: int
    current_player: int
    last_move: Optional[Tuple[int, int]] = None


class IntelligentPlayer(BasePlayer):
    def __init__(self, player_id: int, name: str = "IntelligentPlayer") -> None:
        super().__init__(player_id, name)
        self._rng = random.Random(0)
        self._transposition = {}

    def choose_move(self, board):
        start = time.perf_counter()
        view = self._extract_state(board)
        valid_moves = self._get_valid_moves(board, view)
        if not valid_moves:
            return self._fallback_move(0)

        ordered_moves = self._order_moves(view, valid_moves)

        # Immediate winning move.
        for move in ordered_moves:
            if self._would_win(view, move, view.current_player):
                return self._make_move(move)

        # Immediate block.
        opponent = -view.current_player
        for move in ordered_moves:
            if self._would_win(view, move, opponent):
                return self._make_move(move)

        best_move = ordered_moves[0]
        best_score = -math.inf

        for depth in range(1, MAX_DEPTH + 1):
            if time.perf_counter() - start > TIME_BUDGET:
                break

            current_best_move = best_move
            current_best_score = -math.inf
            alpha = -math.inf
            beta = math.inf

            for move in ordered_moves:
                if time.perf_counter() - start > TIME_BUDGET:
                    break
                next_state = self._simulate_move(view, move, view.current_player)
                if next_state is None:
                    continue
                score = -self._alpha_beta(
                    next_state,
                    depth - 1,
                    -beta,
                    -alpha,
                    -view.current_player,
                    start,
                )
                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move
                alpha = max(alpha, score)

            if current_best_score > -math.inf:
                best_move = current_best_move
                best_score = current_best_score
                ordered_moves = self._reorder_with_best_first(ordered_moves, best_move)

        if best_score == -math.inf:
            best_move = self._rng.choice(valid_moves)

        return self._make_move(best_move)

    def _alpha_beta(
        self,
        state: _StateView,
        depth: int,
        alpha: float,
        beta: float,
        player_sign: int,
        start: float,
    ) -> float:
        if time.perf_counter() - start > TIME_BUDGET:
            return self._evaluate(state, player_sign)

        cache_key = self._state_key(state, depth, alpha, beta, player_sign)
        cached = self._transposition.get(cache_key)
        if cached is not None:
            return cached

        if depth <= 0:
            value = self._evaluate(state, player_sign)
            self._transposition[cache_key] = value
            return value

        valid_moves = self._get_valid_moves_from_state(state)
        if not valid_moves:
            return 0.0

        ordered_moves = self._order_moves(state, valid_moves)
        best = -math.inf
        for move in ordered_moves:
            if time.perf_counter() - start > TIME_BUDGET:
                break
            next_state = self._simulate_move(state, move, player_sign)
            if next_state is None:
                continue
            if self._is_winning_state(next_state, player_sign):
                value = WIN_SCORE - (MAX_DEPTH - depth)
                self._transposition[cache_key] = value
                return value
            score = -self._alpha_beta(next_state, depth - 1, -beta, -alpha, -player_sign, start)
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        if best == -math.inf:
            best = self._evaluate(state, player_sign)

        self._transposition[cache_key] = best
        return best

    def _evaluate(self, state: _StateView, player_sign: int) -> float:
        if self._is_winning_state(state, player_sign):
            return WIN_SCORE
        if self._is_winning_state(state, -player_sign):
            return -WIN_SCORE

        grid = state.grid
        rows = state.rows
        cols = state.cols
        score = 0.0

        center = cols // 2
        for r in range(rows):
            for c in range(cols):
                value = grid[r][c]
                if value == 0:
                    continue
                distance = abs(c - center)
                center_weight = max(0, cols - distance)
                if value == player_sign:
                    score += 2.0 * center_weight
                else:
                    score -= 2.0 * center_weight

        score += self._potential_two_way_threats(grid, player_sign)
        score += self._score_lines(grid, player_sign)
        score -= 0.25 * self._count_opponent_threats(grid, player_sign)
        return score

    def _score_lines(self, grid: Sequence[Sequence[int]], player_sign: int) -> float:
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        total = 0.0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for r in range(rows):
            for c in range(cols):
                for dr, dc in directions:
                    cells = []
                    for i in range(CONNECT_N):
                        rr = r + dr * i
                        cc = c + dc * i
                        if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                            break
                        cells.append(grid[rr][cc])
                    if len(cells) != CONNECT_N:
                        continue
                    total += self._evaluate_window(cells, player_sign)

        return total

    def _evaluate_window(self, cells: Sequence[int], player_sign: int) -> float:
        mine = sum(1 for cell in cells if cell == player_sign)
        opp = sum(1 for cell in cells if cell == -player_sign)
        empty = sum(1 for cell in cells if cell == 0)

        if mine > 0 and opp > 0:
            return 0.0
        if mine == 5:
            return WIN_SCORE
        if opp == 5:
            return -WIN_SCORE
        if mine == 4 and empty == 1:
            return 200000.0
        if opp == 4 and empty == 1:
            return -180000.0
        if mine == 3 and empty == 2:
            return 5000.0
        if opp == 3 and empty == 2:
            return -4500.0
        if mine == 2 and empty == 3:
            return 200.0
        if opp == 2 and empty == 3:
            return -180.0
        if mine == 1 and empty == 4:
            return 10.0
        if opp == 1 and empty == 4:
            return -8.0
        return 0.0

    def _count_opponent_threats(self, grid: Sequence[Sequence[int]], player_sign: int) -> int:
        opp = -player_sign
        threats = 0
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(rows):
            for c in range(cols):
                for dr, dc in directions:
                    cells = []
                    for i in range(CONNECT_N):
                        rr = r + dr * i
                        cc = c + dc * i
                        if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                            break
                        cells.append(grid[rr][cc])
                    if len(cells) != CONNECT_N:
                        continue
                    if sum(1 for x in cells if x == opp) == 4 and sum(1 for x in cells if x == 0) == 1:
                        threats += 1
        return threats

    def _potential_two_way_threats(self, grid: Sequence[Sequence[int]], player_sign: int) -> float:
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        if rows == 0 or cols == 0:
            return 0.0

        score = 0.0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != 0:
                    continue
                contrib = 0
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    line = []
                    for offset in range(-4, 5):
                        rr = r + dr * offset
                        cc = c + dc * offset
                        if 0 <= rr < rows and 0 <= cc < cols:
                            line.append(grid[rr][cc])
                    if len(line) < CONNECT_N:
                        continue
                    if self._window_has_double_extension(line, player_sign):
                        contrib += 1
                if contrib >= 2:
                    score += 2000.0
        return score

    def _window_has_double_extension(self, line: Sequence[int], player_sign: int) -> bool:
        found = 0
        for start in range(0, len(line) - CONNECT_N + 1):
            window = line[start : start + CONNECT_N]
            mine = sum(1 for cell in window if cell == player_sign)
            opp = sum(1 for cell in window if cell == -player_sign)
            empty = sum(1 for cell in window if cell == 0)
            if opp == 0 and mine == 3 and empty == 2:
                found += 1
            if found >= 2:
                return True
        return False

    def _order_moves(self, state: _StateView, moves: Iterable[int]) -> List[int]:
        center = state.cols // 2
        scored = []
        for move in moves:
            score = self._move_priority(state, move)
            score -= abs(move - center) * 0.5
            scored.append((score, move))
        scored.sort(reverse=True)
        return [move for _, move in scored] or list(moves)

    def _move_priority(self, state: _StateView, move: int) -> float:
        if self._would_win(state, move, state.current_player):
            return WIN_SCORE
        opponent = -state.current_player
        if self._would_win(state, move, opponent):
            return WIN_SCORE - 1

        next_state = self._simulate_move(state, move, state.current_player)
        if next_state is None:
            return -math.inf

        score = self._evaluate(next_state, state.current_player)
        if move == state.cols // 2:
            score += 40.0
        return score

    def _reorder_with_best_first(self, moves: Sequence[int], best_move: int) -> List[int]:
        ordered = [best_move]
        ordered.extend(move for move in moves if move != best_move)
        return ordered

    def _extract_state(self, board) -> _StateView:
        grid = self._extract_grid(board)
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        current_player = self._extract_current_player(board)
        last_move = self._extract_last_move(board)
        return _StateView(grid=grid, rows=rows, cols=cols, current_player=current_player, last_move=last_move)

    def _extract_grid(self, board) -> List[List[int]]:
        candidates = ["grid", "board", "cells", "state", "matrix", "table"]
        for attr in candidates:
            if hasattr(board, attr):
                value = getattr(board, attr)
                try:
                    grid = self._normalize_grid(value()) if callable(value) else self._normalize_grid(value)
                    if grid:
                        return grid
                except Exception:
                    pass
        if isinstance(board, (list, tuple)):
            return self._normalize_grid(board)
        for method_name in ("to_list", "as_list", "get_grid", "get_board", "copy", "snapshot"):
            if hasattr(board, method_name):
                method = getattr(board, method_name)
                try:
                    value = method()
                    grid = self._normalize_grid(value)
                    if grid:
                        return grid
                except Exception:
                    pass
        if hasattr(board, "__iter__"):
            try:
                grid = self._normalize_grid(list(board))
                if grid:
                    return grid
            except Exception:
                pass
        raise ValueError("Unable to extract board grid from the provided board object.")

    def _normalize_grid(self, value) -> List[List[int]]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list) or not value:
            return []
        if isinstance(value[0], (list, tuple)):
            grid = [list(row) for row in value]
            return [[self._normalize_cell(cell) for cell in row] for row in grid]
        return []

    def _normalize_cell(self, cell) -> int:
        try:
            if cell is None:
                return 0
            if hasattr(cell, "value"):
                cell = cell.value
            return int(cell)
        except Exception:
            text = str(cell).lower()
            if text in {"x", "player1", "p1", "1"}:
                return 1
            if text in {"o", "player2", "p2", "-1"}:
                return -1
            return 0

    def _extract_current_player(self, board) -> int:
        for attr in ("current_player", "turn", "player", "active_player", "whose_turn"):
            if hasattr(board, attr):
                value = getattr(board, attr)
                try:
                    value = value() if callable(value) else value
                    return 1 if int(value) >= 0 else -1
                except Exception:
                    text = str(value).lower()
                    if "2" in text or "o" in text or "-1" in text:
                        return -1
                    return 1
        return 1

    def _extract_last_move(self, board) -> Optional[Tuple[int, int]]:
        for attr in ("last_move", "previous_move", "move", "latest_move"):
            if hasattr(board, attr):
                value = getattr(board, attr)
                try:
                    value = value() if callable(value) else value
                    if isinstance(value, (list, tuple)) and len(value) >= 2:
                        return int(value[0]), int(value[1])
                except Exception:
                    pass
        return None

    def _get_valid_moves(self, board, state: _StateView) -> List[int]:
        for method_name in ("get_valid_moves", "valid_moves", "get_moves", "available_moves", "possible_moves"):
            if hasattr(board, method_name):
                method = getattr(board, method_name)
                try:
                    value = method() if callable(method) else method
                    moves = self._normalize_moves(value)
                    if moves:
                        return moves
                except Exception:
                    pass

        cols = state.cols
        if cols == 0:
            return []
        top_row = state.grid[0]
        return [c for c in range(cols) if top_row[c] == 0]

    def _get_valid_moves_from_state(self, state: _StateView) -> List[int]:
        if state.cols == 0:
            return []
        return [c for c in range(state.cols) if state.grid[0][c] == 0]

    def _normalize_moves(self, value) -> List[int]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple, set)):
            moves = []
            for item in value:
                try:
                    if isinstance(item, Move):
                        moves.append(int(item.column))
                    elif isinstance(item, (list, tuple)) and item:
                        moves.append(int(item[0]))
                    else:
                        moves.append(int(item))
                except Exception:
                    continue
            return sorted(set(moves))
        return []

    def _simulate_move(self, state: _StateView, move: int, player_sign: int) -> Optional[_StateView]:
        grid = [row[:] for row in state.grid]
        row_index = self._drop_row(grid, move)
        if row_index is None:
            return None
        grid[row_index][move] = player_sign
        return _StateView(grid=grid, rows=state.rows, cols=state.cols, current_player=-player_sign, last_move=(row_index, move))

    def _would_win(self, state: _StateView, move: int, player_sign: int) -> bool:
        next_state = self._simulate_move(state, move, player_sign)
        return bool(next_state and self._is_winning_state(next_state, player_sign))

    def _state_key(self, state: _StateView, depth: int, alpha: float, beta: float, player_sign: int):
        return (
            tuple(tuple(row) for row in state.grid),
            depth,
            player_sign,
            round(alpha, 3),
            round(beta, 3),
        )

    def _drop_row(self, grid: Sequence[Sequence[int]], move: int) -> Optional[int]:
        if not grid or move < 0 or move >= len(grid[0]):
            return None
        for row in range(len(grid) - 1, -1, -1):
            if grid[row][move] == 0:
                return row
        return None

    def _is_winning_state(self, state: _StateView, player_sign: int) -> bool:
        grid = state.grid
        rows = state.rows
        cols = state.cols
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != player_sign:
                    continue
                for dr, dc in directions:
                    count = 0
                    rr, cc = r, c
                    while 0 <= rr < rows and 0 <= cc < cols and grid[rr][cc] == player_sign:
                        count += 1
                        if count >= CONNECT_N:
                            return True
                        rr += dr
                        cc += dc
        return False

    def _make_move(self, column: int):
        try:
            return Move(column)
        except Exception:
            return column

    def _fallback_move(self, column: int):
        return self._make_move(column)
