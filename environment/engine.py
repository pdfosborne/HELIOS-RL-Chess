# Engine used to obtain move scores
import chess.engine
from chess import Board
class Engine:
    """Defines the environment function from the generator engine.
       Expects the following:
        - reset() to reset the env a start position(s)
        - step() to make an action and update the game state
        - legal_moves_generator() to generate the list of legal moves
    """
    def __init__(self) -> None:
        """Initialize Engine"""
        self.board: Board = chess.Board()
        
    def reset(self):
        """Fully reset the environment."""
        self.board.reset()
        obs = self.board.fen()
        return obs

    def step(self, state:any, action:any):
        """Enact an action."""
        # In problems where the agent can choose to reset the env
        if (state=="ENV_RESET")|(action=="ENV_RESET"):
            self.reset()
            
        self.board.push_san(self.board.san(chess.Move.from_uci(action)))
        obs = self.board.fen()
        terminated = self.board.is_game_over()
        # Chess engine does not provide a reward signal by itself
        # - set default per action
        reward = 0
        
        return obs, reward, terminated

    def legal_move_generator(self, obs:any=None):
        """Define legal moves at each position"""
        legal_moves = str(list(self.board.legal_moves)).replace(" Move.from_uci('","").replace("[Move.from_uci('","").replace("')","").replace("]","").split(",")
        legal_moves = legal_moves if (legal_moves != "[]") else [""]
        return legal_moves

