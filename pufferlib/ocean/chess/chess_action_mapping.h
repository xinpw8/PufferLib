#ifndef CHESS_ACTION_MAPPING_H
#define CHESS_ACTION_MAPPING_H

#define TOTAL_CHESS_ACTIONS 1968

// Pre-computed mapping from action ID to UCI string
extern const char* ACTION_ID_TO_UCI[TOTAL_CHESS_ACTIONS];

// Function to convert UCI string to action ID
int uci_to_action_id(const char* uci_str);

#endif // CHESS_ACTION_MAPPING_H
