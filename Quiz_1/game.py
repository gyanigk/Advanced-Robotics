from board import Board

class TicTacToe:
    def __init__(self):
        self.history = []
        self.board = None
        self.current_player = None
        self.game_id = 1 # default game starts from 1 according to instructions
        

    def check_winner(self):
        size = self.board.size
        grid = self.board.grid

        # Check rows and columns for 3 consecutive matching values
        for row in range(size):
            for col in range(size - 2):
                # Check three consecutive values in the row
                if grid[row][col] == grid[row][col + 1] == grid[row][col + 2] != '_':
                    if grid[row][col] == 'X':
                        return 'Player 1'
                    elif grid[row][col] == 'O':
                        return 'Player 2'
                
                # Check three consecutive values in the column
                if grid[col][row] == grid[col + 1][row] == grid[col + 2][row] != '_':
                    if grid[col][row] == 'X':
                        return 'Player 1'
                    elif grid[col][row] == 'O':
                        return 'Player 2'

        # Check diagonals for 3 consecutive matching values
        for row in range(size - 2):
            for col in range(size - 2):
                # Check the main diagonal (top-left to bottom-right)
                if grid[row][col] == grid[row + 1][col + 1] == grid[row + 2][col + 2] != '_':
                    if grid[row][col] == 'X':
                        return 'Player 1'
                    elif grid[row][col] == 'O':
                        return 'Player 2'

                # Check the anti-diagonal (top-right to bottom-left)
                if grid[row][col + 2] == grid[row + 1][col + 1] == grid[row + 2][col] != '_':
                    if grid[row][col + 2] == 'X':
                        return 'Player 1'
                    elif grid[row][col + 2] == 'O':
                        return 'Player 2'

        # If no winner is found, return None
        return None

    def play_game(self):
        print("Welcome to Tic-Tac-Toe!")
        while True:
            board_size = self.get_board_size()
            self.board = Board(board_size)
            self.start_game()

            print("Do you want to play another game? (yes/no)")
            if input().strip().lower() != 'yes':
                break

        self.save_history()

    def get_board_size(self):
        while True:
            try:
                size = int(input("Enter the size of board like 3 - means 3x3 or 4 - means 4x4. Make sure value is in the range 3...10. - "))
                if 3 <= size <= 10:
                    return size
                else:
                    print("Invalid input. Enter a number between 3 and 10.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

    def start_game(self):
        print("Lets Play!")
        self.board.display()

        game_over = False
        round = 0
        while not game_over:
            self.current_player = 'Player 1' if round % 2 == 0 else 'Player 2'
            marker = 'X' if self.current_player == 'Player 1' else 'O'

            self.get_player_move(marker)
            self.board.display()

            winner = self.check_winner()
            if winner:
                print(f"{marker} wins!")
                self.history.append(f"Game {self.game_id} : {marker} wins!")
                self.game_id+=1
                game_over = True
            elif self.board.is_full():
                print("It's a draw!")
                self.history.append(f"Game {self.game_id} :Draw!")
                self.game_id+=1
                game_over = True

            round += 1

    def get_player_move(self, marker):
        while True:
            try:
                move = input(f"{self.current_player}, enter 0 0  i.e row column values to play your move - ")
                row, col = map(int, move.split())
                if self.board.is_valid_move(row, col):
                    self.board.make_move(row, col, marker)
                    break
                else:
                    print("Invalid move. The cell is either occupied or out of bounds. Try again.")
            except (ValueError, IndexError):
                print("Invalid input. Enter two integers separated by a space (e.g., '0 0').")

    def save_history(self):
        with open("game_history.txt", "w") as file:
            file.write("\n".join(self.history))
        
        for game in self.history:
            print(game)
        print("Game history saved to 'game_history.txt'.")


if __name__ == "__main__":
    game = TicTacToe()
    game.play_game()