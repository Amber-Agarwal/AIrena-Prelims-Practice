import numpy as np
import heapq
from agent import Agent

class CustomPlayer(Agent):
    def __init__(self, size, player_number, adv_number):
        super().__init__(size, player_number, adv_number)
        self.name = "Improved Dijkstra AI"
        self._possible_moves = [[x, y] for x in range(self.size) for y in range(self.size)]

    def step(self):
        best_move = None
        best_score = float('-inf')

        moves = self.free_moves()
        # print(f"Available moves: {moves}")  # Debugging

        for move in moves:
            node = self.copy()
            node.set_hex(self.player_number, move)
            score = self.evaluate_move(node)
            # print(f"Move: {move}, Score: {score}")  # Debugging

            if score > best_score:
                best_score = score
                best_move = move

        if best_move is None:  # Fallback move selection
            # print("Warning: All moves have -inf score, picking any valid move.")
            best_move = moves[0] if moves else None  # Pick the first move if available

        if best_move is None:
            # print("Error: No valid moves available!")  # Debugging output
            return None  # Prevents crash

        self.set_hex(self.player_number, best_move)
        return best_move



    def update(self, move_other_player):
        self.set_hex(self.adv_number, move_other_player)
        if move_other_player in self._possible_moves:
            self._possible_moves.remove(move_other_player)

    def evaluate_move(self, node):
        """Heuristic function combining offense, defense, and flexibility."""
        return (
            (self.shortest_path(node, self.adv_number) - self.shortest_path(node, self.player_number))  # Main goal
            + .7 * self.blocking_potential(node)  # Defense
            + .5 * self.path_flexibility(node)    # Adaptability
        )

    def shortest_path(self, node, player):
        """Uses Dijkstra's algorithm to find the shortest path from the player's pieces to their goal edge."""
        size = node.get_grid_size()
        pq = []
        dist = {tuple([x, 0] if player == 1 else [0, y]): float('inf') for x in range(size) for y in range(size)}

        # Initialize with the player's starting row/column
        for i in range(size):
            start = (i, 0) if player == 1 else (0, i)
            if node.get_hex(start) in [0, player]:
                heapq.heappush(pq, (0, start))
                dist[start] = 0

        while pq:
            cost, (x, y) = heapq.heappop(pq)
            if (player == 1 and y == size - 1) or (player == 2 and x == size - 1):
                return cost  # Reached the goal edge

            for nx, ny in node.neighbors([x, y]):
                if node.get_hex((nx, ny)) in [0, player]:  # Valid move
                    new_cost = cost + 1
                    if new_cost < dist.get((nx, ny), float('inf')):
                        dist[(nx, ny)] = new_cost
                        heapq.heappush(pq, (new_cost, (nx, ny)))

        return float('inf')  # No path found

    def blocking_potential(self, node):
        opp_distance = self.shortest_path(node, self.adv_number)
        opp_moves = node.free_moves()

        blocking_score = sum(1 for move in opp_moves if self.shortest_path(node, self.adv_number) > opp_distance)

        return blocking_score

    def path_flexibility(self, node):
        alternative_paths = []

        for move in node.free_moves():
            temp_node = node.copy()
            temp_node.set_hex(self.player_number, move)
            path_length = self.shortest_path(temp_node, self.player_number)

            alternative_paths.append(path_length)

        # Score based on how many paths are viable
        return -min(alternative_paths) if alternative_paths else 0  # Prefer moves that keep multiple options open
