from __future__ import annotations


class Constraint:
    def __init__(self, minimum, maximum):
        self.min = minimum
        self.max = maximum
