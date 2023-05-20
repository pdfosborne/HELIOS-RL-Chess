from typing import List, Dict
import pandas as pd
import torch
from torch import Tensor

import chess # Required for us to store board in parallel to actual game for computing prior action piece names
from chess import Board

# StateAdapter includes static methods for adapters
from adapters.adapter_abstract import StateAdapter
from helios_rl.encoders.sentence_transformer_MiniLM_L6v2 import LanguageEncoder

class PriorActionsToLanguageAdapter(StateAdapter):
    _cached_state_idx: Dict[str, int] = dict()

    def __init__(self, size: int = 15):
        self.size = size
        self.temp_board:Board = chess.Board()
        
        self.encoder = LanguageEncoder()

    def adapter(self, board_fen:str = None, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """Map prior actions to Language versions using Logic Rules. 
        Needs to add both the last white and black player's moves into the list."""
        # We play through the episode actions to extract the pieces that were moved
        # -> Reset board on new episode and set encoded state for empty action history        
        if len(episode_action_history)==0:
            self.temp_board.reset()
            self.language_action_history: List[str] = []
            self.last_known_action:str = ''
            if encode:
                state_encoded = self.encoder.encode(['']*self.size)
            else:
                state_encoded = ['']
        else:
            # We currently call the adapter in the env back to back for each player perspective
            # -> for other adapters this is fine but we can't log the same info twice here
            if self.last_known_action == episode_action_history[-1]:
                self.language_action_history = self.language_action_history
            else:
                last_action = episode_action_history[-1]
                board_fen_temp = self.temp_board.fen()
                # Transform action to language description
                # 1 -> 'e2e4' to 'White pawn from e2 to e4'
                # 2 --> 'White pawn from e2 to e4' to 'White pawn moves forward two spaces'
                LANG_action = StateAdapter.uci_to_lang_action(last_action, board_fen_temp)
                LANG_action_description = StateAdapter.action_to_lang(LANG_action, board_fen_temp)
                # Store language descriptions of each action
                self.language_action_history.append(LANG_action_description)
                
                # Update temp board to continue game for next action
                try:
                    self.temp_board.push_san(self.temp_board.san(chess.Move.from_uci(last_action)))
                except:
                    print(" ")
                    print(self.temp_board)
                    print(episode_action_history)
                    print(last_action)
                    self.temp_board.push_san(self.temp_board.san(chess.Move.from_uci(last_action)))
                self.last_known_action = last_action
            
            # Encode to Tensor for agents
            # -> fixed length with empty string when few prior actions
            action_history = ['']*(self.size-len(self.language_action_history)) + self.language_action_history[-self.size:]
            # We need to feed actions individually to encoder to preserve order
            if encode:
                state_encoded = self.encoder.encode(state=action_history)
            else:
                state_encoded = action_history

            if (indexed):
                state_indexed = list()
                for sent in action_history:
                    if (sent not in PriorActionsToLanguageAdapter._cached_state_idx):
                        PriorActionsToLanguageAdapter._cached_state_idx[sent] = len(PriorActionsToLanguageAdapter._cached_state_idx)
                    state_indexed.append(PriorActionsToLanguageAdapter._cached_state_idx[sent])

                state_encoded = torch.tensor(state_indexed)

        return state_encoded
    
    def sample():
        board = chess.Board(fen='rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2')
        legal_moves = ['g1h3', 'g1f3', 'g1e2', 'f1a6', 'f1b5', 'f1c4', 'f1d3', 
                       'f1e2', 'e1e2', 'd1h5', 'd1g4', 'd1f3', 'd1e2', 'b1c3', 
                       'b1a3', 'e4e5', 'h2h3', 'g2g3', 'f2f3', 'd2d3', 'c2c3', 
                       'b2b3', 'a2a3', 'h2h4', 'g2g4', 'f2f4', 'd2d4', 'c2c4', 'b2b4', 'a2a4']
        episode_action_history = ['e2e4', 'c7c5']
        adapter = PriorActionsToLanguageAdapter()
        state = adapter.adapter(board, legal_moves, [], encode=False)
        state = adapter.adapter(board, legal_moves, [episode_action_history[0]], encode=False)
        state = adapter.adapter(board, legal_moves, episode_action_history, encode=False)
        # ---
        adapter = PriorActionsToLanguageAdapter()
        state_encoded = adapter.adapter(board, legal_moves, [], encode=True)
        state_encoded = adapter.adapter(board, legal_moves, [episode_action_history[0]], encode=True)
        state_encoded = adapter.adapter(board, legal_moves, episode_action_history, encode=True)
        return state, state_encoded