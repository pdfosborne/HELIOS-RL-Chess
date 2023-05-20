from typing import List, Dict
import pandas as pd
import torch
from torch import Tensor
import chess
from chess import Board

# StateAdapter includes static methods for adapters
from adapters.adapter_abstract import StateAdapter
from helios_rl.encoders.sentence_transformer_MiniLM_L6v2 import LanguageEncoder

class PossibleActionsToLanguageAdapter(StateAdapter): 
    _cached_state_idx: Dict[str, int] = dict()

    def __init__(self, size: int = 125):
        self.size = size # Max num of moves for a single board is suggested to be around 110
        self.temp_board: Board = chess.Board()
        self.language_action_history: List[str] = []
        self.last_known_action: str = ''
        self.encoder = LanguageEncoder()

    def adapter(self, board_fen: str, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """Vector of possible actions."""
        possible_actions_to_Lang: List[str] = list()

        if len(episode_action_history)==0:
            self.temp_board.reset()
            self.language_action_history: List[str] = []
            self.last_known_action:str = ''
            if encode:
                state_encoded = self.encoder.encode(['']*self.size)
            else:
                state_encoded = ['']
        else:
            for action in legal_moves:
                # 1 -> 'e2e4' to 'White pawn from e2 to e4'
                # 2 --> 'White pawn from e2 to e4' to 'White pawn moves forward two spaces'
                LANG_action = StateAdapter.uci_to_lang_action(action, board_fen)
                LANG_action_description = StateAdapter.action_to_lang(LANG_action=LANG_action, board_fen=board_fen)
                possible_actions_to_Lang.append(LANG_action_description)
            
            # Encode to Tensor for agents
            # -> fixed length with empty string when few possible actions
            state = ['']*(self.size-len(possible_actions_to_Lang)) + possible_actions_to_Lang[:self.size]
            # Encode each action seperately and stack
            if encode:
                state_encoded = self.encoder.encode(state=state)
            else:
                state_encoded = state

            if (indexed):
                state_indexed = list()
                for sent in state:
                    if (sent not in PossibleActionsToLanguageAdapter._cached_state_idx):
                        PossibleActionsToLanguageAdapter._cached_state_idx[sent] = len(PossibleActionsToLanguageAdapter._cached_state_idx)
                    state_indexed.append(PossibleActionsToLanguageAdapter._cached_state_idx[sent])

                state_encoded = torch.tensor(state_indexed)

        return state_encoded
    
    @staticmethod
    def sample():
        board = chess.Board(fen='rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2')
        legal_moves = ['g1h3', 'g1f3', 'g1e2', 'f1a6', 'f1b5', 'f1c4', 'f1d3', 
                       'f1e2', 'e1e2', 'd1h5', 'd1g4', 'd1f3', 'd1e2', 'b1c3', 
                       'b1a3', 'e4e5', 'h2h3', 'g2g3', 'f2f3', 'd2d3', 'c2c3', 
                       'b2b3', 'a2a3', 'h2h4', 'g2g4', 'f2f4', 'd2d4', 'c2c4', 'b2b4', 'a2a4']
        episode_action_history = ['e2e4', 'c7c5']
        adapter = PossibleActionsToLanguageAdapter()
        state = adapter.adapter(board, legal_moves, episode_action_history, encode=False)
        state_encoded = adapter.adapter(board, legal_moves, episode_action_history, encode=True)
        return state, state_encoded