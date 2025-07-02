import numpy as np


class CellBlock:
    def __init__(self, type: str, data: np.typing.ArrayLike):
        self.type = type
        self.data = np.asarray(data)


class Mesh:
    def __init__(
        self,
        points: np.typing.ArrayLike,
        cells: list[tuple[str, np.typing.ArrayLike]],
        point_data: dict[str, np.typing.ArrayLike] | None = None,
    ):
        self.points = np.asarray(points)
        self.cells = [CellBlock(type, data) for type, data in cells]
        self.point_data = {} if point_data is None else point_data
