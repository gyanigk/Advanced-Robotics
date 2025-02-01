class Board:
    def __init__(self, size):
        self.size = size
        self.grid = [['_' for _ in range(size)] for _ in range(size)]

    def display(self):
        for i in range(self.size):
            for j in range(self.size):
                if j == self.size - 1:
                    print(self.grid[i][j], end=" ")
                else:
                    print(self.grid[i][j], end="|")
            print()

    def is_valid_move(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size and self.grid[row][col] == '_'

    def make_move(self, row, col, marker):
        self.grid[row][col] = marker

    def is_full(self):
        for row in self.grid:
            for cell in row:
                if cell == '_':  # If any cell is empty
                    return False
        return True  # If no empty cell is found, the grid is full