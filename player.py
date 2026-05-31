from __future__ import annotations


class Move:
    def __init__(self, column: int):
        self.column = int(column)

    def __int__(self) -> int:
        return self.column

    def __index__(self) -> int:
        return self.column

    def __repr__(self) -> str:
        return f"Move({self.column})"


class BasePlayer:
    def __init__(self, name: str = "Player") -> None:
        self.name = name

    def choose_move(self, board):
        raise NotImplementedError
