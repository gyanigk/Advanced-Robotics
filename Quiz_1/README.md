# Tic-Tac-Toe Game

This is a simple command-line Tic-Tac-Toe game written in Python. The game allows two players to play against each other on a customizable grid size (from 3x3 to 10x10). The game saves the results in a file called `game_history.txt`.
------
## Requirements
- Python 3.x
------
## Assignment Requirements met
- Players can select a board size ranging from 3x3 to 10x10.
- The game alternates turns between two players: **Player 1 (X)** and **Player 2 (O)**.
- A player wins if they align three consecutive markers (horizontally, vertically, or diagonally).
- The game ends in a draw if all cells are filled without a winner.
- Game history is saved to a text file for later review.
------
## How to Run
1. Clone the repository
2. Ensure you have Python 3 installed on your machine.
3. The game consists of two Python files:
    - `game.py` (main file for the game logic)
    - `board.py` (contains the `Board` class for managing the game board - submodule required according to the instructions)
4. Run the game on cmd line:
    ```bash
    python game.py
    ```
------
## Gameplay Instructions
1. When the game starts, you'll be prompted to enter the board size (e.g., `3` for a 3x3 board default value taken if nothing entered).
2. The empty board will be displayed.
3. Players take turns entering their moves in the format `row column` (e.g., `0 0` for the top-left corner).
4. The game will validate moves and update the board after each turn.
5. The game announces the winner or a draw when the game ends.
6. You can choose to play another round or exit the game.
------
## Logic to check the winner
A winner is defined by three consecutive matching values (either 'X' or 'O') in rows, columns, or diagonals.
1. Outer loops: The outer loops iterate over each row and column of the grid.
```
- for row in range(size): Loops through each row of the grid.
- for col in range(size - 2): Loops through columns up to size - 2 to allow checking three consecutive cells (since you need three consecutive cells to form a line).
```
2. Check rows:
Inside the row loop, the first if checks if three consecutive values in the same row (starting at col, col + 1, col + 2) are equal and not empty ('_').
If so, it checks if the values are 'X' or 'O' and returns the corresponding player: "Player 1" for 'X' and "Player 2" for 'O'.

3. Check columns:
The second if inside the same loop checks if three consecutive values in the same column (starting at row, row + 1, row + 2) are equal and not empty.
Similar to the row check, if they match, it determines if the value is 'X' or 'O' and returns the corresponding player.

4. Check diagonals:
Another set of loops (for row in range(size - 2) and for col in range(size - 2)) checks diagonals.
```
Main diagonal: The first if checks the diagonal running from top-left to bottom-right (grid[row][col], grid[row + 1][col + 1], grid[row + 2][col + 2]).
Anti-diagonal: The second if checks the diagonal running from top-right to bottom-left (grid[row][col + 2], grid[row + 1][col + 1], grid[row + 2][col]).
```
If any of these diagonals have three consecutive matching values, it returns the winner, similar to the row and column checks.
------
## Example Output
```
PS G:\Advanced-Robotics\Quiz_1> python game.py
Welcome to Tic-Tac-Toe!
Enter the size of board like 3 - means 3x3 or 4 - means 4x4. Make sure value is in the range 3...10. - 3
Lets Play!
_|_|_
_|_|_
_|_|_
Player 1, enter 0 0  i.e row column values to play your move -  0 0
X|_|_ 
_|_|_
_|_|_
Player 2, enter 0 0  i.e row column values to play your move - 1 1
X|_|_ 
_|O|_
_|_|_
Player 1, enter 0 0  i.e row column values to play your move - 2 0
X|_|_ 
_|O|_
X|_|_
Player 2, enter 0 0  i.e row column values to play your move - 2 2
X|_|_ 
_|O|_
X|_|O
Player 1, enter 0 0  i.e row column values to play your move - 1 2
X|_|_ 
_|O|X
X|_|O
Player 2, enter 0 0  i.e row column values to play your move - 0 2
X|_|O 
_|O|X
X|_|O
Player 1, enter 0 0  i.e row column values to play your move - 1 0
X|_|O 
X|O|X
X|_|O
X wins!
Do you want to play another game? (yes/no)
no
['Game 0 : X wins!']
Game history saved to 'game_history.txt'.
```
------
## License
------
