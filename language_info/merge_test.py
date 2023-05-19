import json
# TODO: New method for filtering random policy with 'reasonable' moves
# TODO: Ideally we have occurance count so we can remove very unlikely choices
from move_names_parser import MoveDataExtract as PlayerDataA

# Opening JSON file
with open('./language_info/stats_map.json', 'r') as json_file:
    PlayerDataB = json.load(json_file)

# --- Data A -------------------------------------------------------------------------------------------------
# dict[req_board][move_uci] = {'prev_moves_uci': move_seq_prev, 'move_name': move_name,
#                                       'outcome_board': outcome_board,'count': human_play_count,
#                                       'win_perc': win_perc} 
player_data_dict = PlayerDataA().player_data()
print(player_data_dict[list(player_data_dict.keys())[0]])
print("---.---")
# --- Data B -------------------------------------------------------------------------------------------------
# Dict[board_fen][move_uci] = {'prev_moves_uci': move_list_current,'totalGames': branch["data"]["totalGames"]} 
player_data_counts_dict = PlayerDataB()#.tree_to_dict()
print(player_data_counts_dict[list(player_data_counts_dict.keys())[0]])
print("---.---")
# --- Merge -------------------------------------------------------------------------------------------------

for board_fen in list(player_data_dict.keys()):
    for move_uci in list(player_data_dict[board_fen].keys()):
        try:
            player_data_dict[board_fen][move_uci]["count"] = player_data_counts_dict[board_fen][move_uci]['totalGames']
        except:
            print(move_uci, "->", player_data_dict[board_fen][move_uci])
            print("---")
# ------------------------------------------------------------------------------------------------------------
print(player_data_dict[list(player_data_dict.keys())[0]])
