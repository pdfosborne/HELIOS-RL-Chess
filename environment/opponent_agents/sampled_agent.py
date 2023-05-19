import random
import json
from typing import List, Iterable, Any, TypeVar, Generic
from helios_rl.agents.agent_abstract import Agent
import torch
from torch import Tensor
T = TypeVar("T")


class SampledAgent(Agent, Generic[T]):
    """This is simply a random decision maker, does not learn."""
    def __init__(self):
        super().__init__()
        #with open('language_info/stats_map.json', 'r') as json_file:
        
        # - MERGE: combine human play data for occurrence count
        # dict[req_board][move_uci] = {'prev_moves_uci': move_seq_prev, 'move_name': move_name,
        #                               'outcome_board': outcome_board,'count': human_play_count,
        #                               'win_perc': win_perc} 
        print("Loading Player Data")
        with open('./language_info/move_stats.json', 'r') as json_file:
            self.player_data_dict = json.load(json_file)
        # dict[board_fen][move_uci] = {'prev_moves_uci': move_list_current,'totalGames': branch["data"]["totalGames"]}
        with open('./language_info/stats_map.json', 'r') as json_file:
            self.player_data_counts_dict = json.load(json_file)
        
        for board_fen in list(self.player_data_dict.keys()):
            for move_uci in list(self.player_data_dict[board_fen].keys()):
                if (board_fen in self.player_data_counts_dict):
                    if (move_uci in self.player_data_counts_dict[board_fen]):
                        self.player_data_dict[board_fen][move_uci]["count"] = self.player_data_counts_dict[board_fen][move_uci]['totalGames']  
                    else:
                        self.player_data_dict[board_fen][move_uci]["count"] = 0
                else:
                    self.player_data_dict[board_fen][move_uci]["count"] = 0
                
    def policy(self, state:Iterable[Any] = None, legal_actions:List[T] = []) -> T:
        if state in self.player_data_dict:
            # Create list of actions/count
            # - need to do this to get total
            total = 0
            action_lst = []
            count_lst = []
            for action in self.player_data_dict[state]:
                count = self.player_data_dict[state][action]['count']
                total += count

                action_lst.append(action)
                count_lst.append(count)
                
            # Sample over count list
            rng = random.randint(0, total)
            idx = 0
            cum_count = 0
            for count in count_lst:
                cum_count+=count
                if cum_count >= rng:
                    break
                else:
                    idx+=1
            if cum_count>0:
                action = action_lst[idx]
            else:
                action = str(random.choice(legal_actions))
        else:
            action = str(random.choice(legal_actions))
        
        return action
    
    def learn(self, state: Tensor, next_state: Tensor, r_p: float, action_code: str) -> float:
        # Do nothing.
        return None