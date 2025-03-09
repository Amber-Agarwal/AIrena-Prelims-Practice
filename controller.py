from grid import Grid
import time

class Controller:
    def __init__(self, size, player1, player2, custom_player_number=None, time_limit=None):
        self._grid = Grid(size)
        self._player1 = player1
        self._player2 = player2
        self._current_player = 1
        self._winner = 0

        # If no time_limit is specified or no player is designated as "custom", do not enforce timing
        self._custom_player_number = custom_player_number
        self._time_limit = time_limit if time_limit is not None else 92.0
        self.custom_time_used_for1 = 0.0  # tracks the custom agent's total time
        self.custom_time_used_for2 = 0.0  # tracks the custom agent's total time

    def update(self):
        start_time = time.time()

        if self._current_player == 1:
            coordinates = self._player1.step()
            self._grid.set_hex(1, coordinates)
            self._current_player = 2
            self._player2.update(coordinates)
        else:
            coordinates = self._player2.step()
            self._grid.set_hex(2, coordinates)
            self._current_player = 1
            self._player1.update(coordinates)

        # Measure time for whichever player is the designated custom player
        step_time = time.time() - start_time
        if self._current_player == 1:
            self.custom_time_used_for2 += step_time
        elif self._current_player == 2:
            self.custom_time_used_for1 += step_time
        # print("Step time:")
        # print(step_time)
        # print("Custom time used:")
        # print(self.custom_time_used_for1, self.custom_time_used_for2)

        # If custom agent exceeds time_limit, stop and declare the other player as winner
        if self._current_player == 1 and self.custom_time_used_for1 > self._time_limit:
            self._winner = 2
            return
        elif self._current_player == 2 and self.custom_time_used_for2 > self._time_limit:
            self._winner = 1
            return

        self._check_win()

    def _check_win(self):
        if self._grid.check_win(1):
            self._winner = 1
        elif self._grid.check_win(2):
            self._winner = 2