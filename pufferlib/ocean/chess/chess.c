#define CHESS_DEBUG_BUILD
#include "chess.h"
#include <time.h>

void debug_save_state(Chess* env) {
    if (!env->debug_mode) return;
    if (env->debug_history_idx < env->debug_history_count - 1 && env->debug_history_count > 0) {
        env->debug_history_count = env->debug_history_idx + 1;
    }
    int idx = env->debug_history_count % DEBUG_HISTORY_SIZE;
    memcpy(env->debug_obs_history[idx], env->observations, OBS_SIZE * 2);
    memcpy(&env->debug_pos_history[idx], &env->pos, sizeof(Position));
    env->debug_pick_phase_history[idx][0] = env->pick_phase[0];
    env->debug_pick_phase_history[idx][1] = env->pick_phase[1];
    env->debug_selected_sq_history[idx][0] = env->selected_square[0];
    env->debug_selected_sq_history[idx][1] = env->selected_square[1];
    env->debug_actions_history[idx][0] = env->actions[0];
    env->debug_actions_history[idx][1] = env->selfplay ? env->actions[1] : -1;
    env->debug_last_move_history[idx] = env->last_move;
    env->debug_chess_moves_history[idx] = env->chess_moves;
    env->debug_rewards_history[idx] = env->rewards[0];
    env->debug_history_count++;
    env->debug_history_idx = env->debug_history_count - 1;
}

static Font debug_piece_font = {0};
static int debug_use_unicode = 0;

static void debug_load_piece_font(int font_size) {
    const char* candidates[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Apple Symbols.ttf",
        "C:\\Windows\\Fonts\\seguisym.ttf",
        "C:\\Windows\\Fonts\\arial.ttf"
    };
    int codepoints[12];
    for (int i = 0; i < 12; i++) {
        codepoints[i] = 0x2654 + i;
    }
    for (int i = 0; i < (int)(sizeof(candidates) / sizeof(candidates[0])); i++) {
        if (FileExists(candidates[i])) {
            debug_piece_font = LoadFontEx(candidates[i], font_size, codepoints, 12);
            if (debug_piece_font.glyphCount > 0) {
                int test_glyph = GetGlyphIndex(debug_piece_font, 0x2659);
                if (test_glyph > 0) {
                    debug_use_unicode = 1;
                    TraceLog(LOG_INFO, "Loaded Unicode chess font: %s", candidates[i]);
                    return;
                }
                UnloadFont(debug_piece_font);
            }
        }
    }
    debug_use_unicode = 0;
    TraceLog(LOG_INFO, "No Unicode chess font found, using letters");
}

static const char* debug_get_piece_char(Piece p) {
    if (p < 0 || p > 14) return "";
    return debug_use_unicode ? PIECE_UNICODE[p] : PIECE_CHARS[p];
}

static void debug_draw_piece_text(Piece p, int x, int y, int font_size, Color color) {
    if (p == NO_PIECE) return;
    Color outline = (color_of(p) == CHESS_WHITE) ? (Color){0, 0, 0, 220} : (Color){255, 255, 255, 180};
    if (debug_use_unicode && debug_piece_font.glyphCount > 0) {
        const char* str = PIECE_FILLED[p];
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx != 0 || dy != 0) {
                    DrawTextEx(debug_piece_font, str, (Vector2){x + dx, y + dy}, font_size, 0, outline);
                }
            }
        }
        DrawTextEx(debug_piece_font, str, (Vector2){x, y}, font_size, 0, color);
    } else {
        const char* str = PIECE_CHARS[p];
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx != 0 || dy != 0) {
                    DrawText(str, x + dx, y + dy, font_size, outline);
                }
            }
        }
        DrawText(str, x, y, font_size, color);
    }
}

static void debug_unload_piece_font(void) {
    if (debug_use_unicode && debug_piece_font.glyphCount > 0) {
        UnloadFont(debug_piece_font);
        debug_piece_font = (Font){0};
        debug_use_unicode = 0;
    }
}

static void debug_draw_obs_plane_with_pieces(uint8_t* plane_data, Position* pos, int x, int y, int cell_size, 
                                              const char* label, Color highlight_color, int flip_for_black,
                                              int plane_type, int is_my_piece) {
    DrawRectangle(x - 2, y - 18, cell_size * 8 + 4, cell_size * 8 + 22, (Color){30, 30, 30, 255});
    DrawText(label, x, y - 16, 10, WHITE);
    for (int sq = 0; sq < 64; sq++) {
        int file = sq % 8;
        int rank = sq / 8;
        int display_file = flip_for_black ? (7 - file) : file;
        int display_rank = 7 - rank;
        int draw_x = x + display_file * cell_size;
        int draw_y = y + display_rank * cell_size;
        Color base_color = ((file + rank) % 2 == 0) ? (Color){60, 60, 60, 255} : (Color){40, 40, 40, 255};
        DrawRectangle(draw_x, draw_y, cell_size, cell_size, base_color);
        if (plane_data[sq]) {
            DrawRectangle(draw_x + 1, draw_y + 1, cell_size - 2, cell_size - 2, highlight_color);
            int actual_sq = flip_for_black ? (sq ^ 56) : sq;
            Piece pc = pos->board[actual_sq];
            if (pc != NO_PIECE && cell_size >= 8) {
                Color text_color = color_of(pc) == CHESS_WHITE ? WHITE : YELLOW;
                int font_size = cell_size > 10 ? 8 : 6;
                debug_draw_piece_text(pc, draw_x + 1, draw_y, font_size, text_color);
            }
        }
    }
    DrawRectangleLines(x, y, cell_size * 8, cell_size * 8, GRAY);
}

static void debug_draw_promo_plane(uint8_t* plane_data, int x, int y, int cell_size, const char* label,
                                    int* promo_counts) {
    DrawRectangle(x - 2, y - 18, cell_size * 8 + 4 + 150, cell_size * 4 + 22, (Color){30, 30, 30, 255});
    DrawText(label, x, y - 16, 10, WHITE);
    const char* promo_labels[] = {"Q", "R", "B", "N"};
    const char* promo_names[] = {"Queen", "Rook", "Bishop", "Knight"};
    int counts[4] = {0, 0, 0, 0};
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 8; col++) {
            int idx = row * 8 + col;
            if (plane_data[idx]) counts[row]++;
        }
    }
    if (promo_counts) {
        for (int i = 0; i < 4; i++) promo_counts[i] = counts[i];
    }
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 8; col++) {
            int draw_x = x + col * cell_size;
            int draw_y = y + row * cell_size;
            Color base_color = ((row + col) % 2 == 0) ? (Color){60, 60, 60, 255} : (Color){40, 40, 40, 255};
            DrawRectangle(draw_x, draw_y, cell_size, cell_size, base_color);
            int idx = row * 8 + col;
            if (plane_data[idx]) {
                DrawRectangle(draw_x + 1, draw_y + 1, cell_size - 2, cell_size - 2, PURPLE);
                char file_label[2] = {'a' + col, '\0'};
                DrawText(file_label, draw_x + 1, draw_y, 6, WHITE);
            }
        }
        DrawText(promo_labels[row], x - 12, y + row * cell_size + 2, 10, LIGHTGRAY);
        char count_text[32];
        snprintf(count_text, sizeof(count_text), "%s: %d file(s)", promo_names[row], counts[row]);
        Color count_color = counts[row] > 0 ? PURPLE : GRAY;
        DrawText(count_text, x + cell_size * 8 + 8, y + row * cell_size + 2, 10, count_color);
    }
    DrawRectangleLines(x, y, cell_size * 8, cell_size * 4, GRAY);
}

static int debug_draw_main_board(Chess* env, Position* pos, int x, int y, int cell_size, 
                                   int view_player, int active_player, Move last_move, uint8_t* valid_pieces,
                                   uint8_t* valid_dests, int is_dest_phase, Square selected_sq,
                                   MoveList* valid_destinations, int* hovered_sq_out) {
    Vector2 mouse = GetMousePosition();
    int hovered_sq = -1;
    for (int sq = 0; sq < 64; sq++) {
        int file = sq % 8;
        int rank = sq / 8;
        int display_file = view_player ? (7 - file) : file;
        int display_rank = view_player ? rank : (7 - rank);
        int draw_x = x + display_file * cell_size;
        int draw_y = y + display_rank * cell_size;
        if (mouse.x >= draw_x && mouse.x < draw_x + cell_size &&
            mouse.y >= draw_y && mouse.y < draw_y + cell_size) {
            hovered_sq = sq;
        }
        Color sq_color = ((file + rank) % 2 == 0) ? (Color){181, 136, 99, 255} : (Color){240, 217, 181, 255};
        if (last_move != MOVE_NONE) {
            if (sq == (int)from_sq(last_move) || sq == (int)to_sq(last_move)) {
                sq_color = (Color){186, 202, 68, 255};
            }
        }
        DrawRectangle(draw_x, draw_y, cell_size, cell_size, sq_color);
        int obs_sq = active_player ? (sq ^ 56) : sq;
        if (!is_dest_phase && valid_pieces && valid_pieces[obs_sq]) {
            DrawRectangle(draw_x, draw_y, cell_size, cell_size, (Color){0, 200, 0, 80});
            DrawRectangleLines(draw_x + 2, draw_y + 2, cell_size - 4, cell_size - 4, (Color){0, 255, 0, 200});
        }
        if (is_dest_phase && valid_dests && valid_dests[obs_sq]) {
            DrawRectangle(draw_x, draw_y, cell_size, cell_size, (Color){0, 200, 255, 80});
            DrawRectangleLines(draw_x + 2, draw_y + 2, cell_size - 4, cell_size - 4, (Color){0, 255, 255, 200});
        }
        if (is_dest_phase && selected_sq != SQ_NONE && sq == (int)selected_sq) {
            DrawRectangleLines(draw_x + 1, draw_y + 1, cell_size - 2, cell_size - 2, YELLOW);
            DrawRectangleLines(draw_x + 2, draw_y + 2, cell_size - 4, cell_size - 4, YELLOW);
        }
        Piece pc = pos->board[sq];
        if (pc != NO_PIECE) {
            Color pc_color = color_of(pc) == CHESS_WHITE
                ? (Color){255, 255, 255, 255}
                : (Color){30, 30, 30, 255};
            int text_x = draw_x + cell_size / 4;
            int text_y = draw_y + cell_size / 8;
            debug_draw_piece_text(pc, text_x, text_y, cell_size / 2, pc_color);
        }
    }
    int hover_obs_sq = active_player ? (hovered_sq ^ 56) : hovered_sq;
    if (!is_dest_phase && hovered_sq >= 0 && valid_pieces && valid_pieces[hover_obs_sq]) {
        Piece hp = pos->board[hovered_sq];
        if (hp != NO_PIECE) {
            MoveList ml;
            ml.count = 0;
            generate_legal(pos, &ml, NULL, NULL);
            for (int m = 0; m < ml.count; m++) {
                if (from_sq(ml.moves[m].move) == (Square)hovered_sq) {
                    int dest_sq = to_sq(ml.moves[m].move);
                    int dest_file = dest_sq % 8;
                    int dest_rank = dest_sq / 8;
                    int dest_display_file = view_player ? (7 - dest_file) : dest_file;
                    int dest_display_rank = view_player ? dest_rank : (7 - dest_rank);
                    int dest_x = x + dest_display_file * cell_size;
                    int dest_y = y + dest_display_rank * cell_size;
                    DrawCircle(dest_x + cell_size/2, dest_y + cell_size/2, cell_size/6, (Color){255, 100, 100, 180});
                }
            }
            int hover_file = hovered_sq % 8;
            int hover_rank = hovered_sq / 8;
            int hover_display_file = view_player ? (7 - hover_file) : hover_file;
            int hover_display_rank = view_player ? hover_rank : (7 - hover_rank);
            int hover_x = x + hover_display_file * cell_size;
            int hover_y = y + hover_display_rank * cell_size;
            DrawRectangleLines(hover_x, hover_y, cell_size, cell_size, RED);
        }
    }
    DrawRectangleLines(x, y, cell_size * 8, cell_size * 8, WHITE);
    for (int f = 0; f < 8; f++) {
        int df = view_player ? (7 - f) : f;
        char label[2] = {'a' + f, '\0'};
        DrawText(label, x + df * cell_size + cell_size/2 - 4, y + cell_size * 8 + 2, 12, WHITE);
    }
    for (int r = 0; r < 8; r++) {
        int dr = view_player ? r : (7 - r);
        char label[2] = {'1' + r, '\0'};
        DrawText(label, x - 12, y + dr * cell_size + cell_size/2 - 6, 12, WHITE);
    }
    if (hovered_sq_out) *hovered_sq_out = hovered_sq;
    return hovered_sq;
}

void c_render_debug(Chess* env) {
    const int mini_cell = 10;
    const int main_cell = 52;
    const int main_board_size = main_cell * 8;
    const int padding = 12;
    const int window_width = main_board_size + 50 + (mini_cell * 8 + padding) * 6 + 250;
    const int window_height = 850;
    if (env->client == NULL) {
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        InitWindow(window_width, window_height, "PufferLib Chess - DEBUG VIEW");
        SetTargetFPS(60);
        debug_load_piece_font(main_cell / 2);
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->cell_size = main_cell;
        env->debug_paused = 1;
        env->debug_view_player = env->learner_color;
        env->debug_selected_plane = -1;
    }
    int needs_step = 0;
    do {
        int hist_idx = env->debug_history_idx;
        if (hist_idx < 0) hist_idx = 0;
        if (hist_idx >= env->debug_history_count && env->debug_history_count > 0) {
            hist_idx = env->debug_history_count - 1;
        }
        if (env->debug_history_count == 0) {
            populate_observations(env);
            debug_save_state(env);
            hist_idx = 0;
        }
        uint8_t* obs = env->debug_obs_history[hist_idx % DEBUG_HISTORY_SIZE];
        Position* pos = &env->debug_pos_history[hist_idx % DEBUG_HISTORY_SIZE];
        if (pos->board[0] == 0 && pos->board[1] == 0) {
            pos = &env->pos;
            obs = env->observations;
        }
        int view_player = env->debug_view_player;
        int active_player = pos->sideToMove;
        int active_buffer_idx = (env->learner_color == CHESS_WHITE) ? active_player : (1 - active_player);
        int view_buffer_idx = (env->learner_color == CHESS_WHITE) ? view_player : (1 - view_player);
        int active_buffer_offset = active_buffer_idx * OBS_SIZE;
        int view_buffer_offset = view_buffer_idx * OBS_SIZE;
        int pick_phase = env->debug_pick_phase_history[hist_idx % DEBUG_HISTORY_SIZE][active_player];
        int is_dest_phase = (pick_phase == 1);
        uint8_t* valid_pieces = obs + active_buffer_offset + O_VALID_PIECES;
        uint8_t* valid_dests = obs + active_buffer_offset + O_VALID_DESTS;
        uint8_t* valid_promos = obs + active_buffer_offset + O_VALID_PROMOS;
        Square selected_sq = env->debug_selected_sq_history[hist_idx % DEBUG_HISTORY_SIZE][active_player];
        BeginDrawing();
        ClearBackground((Color){20, 20, 28, 255});
        int main_x = 20;
        int main_y = 45;
        const char* view_side = view_player ? "BLACK" : "WHITE";
        const char* active_side = active_player ? "BLACK" : "WHITE";
        char board_title[128];
        snprintf(board_title, sizeof(board_title), "BOARD (orient: %s | turn: %s) - V to flip", 
                 view_side, active_side);
        DrawText(board_title, main_x, main_y - 28, 13, WHITE);
        int hovered_sq = -1;
        debug_draw_main_board(env, pos, main_x, main_y, main_cell, view_player, active_player,
                              env->debug_last_move_history[hist_idx % DEBUG_HISTORY_SIZE],
                              valid_pieces, valid_dests, is_dest_phase, selected_sq,
                              NULL, &hovered_sq);
        int hover_info_y = main_y + main_board_size + 20;
        if (hovered_sq >= 0) {
            char hover_text[128];
            int file = hovered_sq % 8;
            int rank = hovered_sq / 8;
            Piece pc = pos->board[hovered_sq];
            const char* piece_name = "";
            (void)piece_name;
            if (pc != NO_PIECE) {
                const char* names[] = {"", "Pawn", "Knight", "Bishop", "Rook", "Queen", "King"};
                int pt = type_of_p(pc);
                const char* color_name = color_of(pc) == CHESS_WHITE ? "White" : "Black";
                snprintf(hover_text, sizeof(hover_text), "Hover: %c%d - %s %s", 
                         'a' + file, rank + 1, color_name, names[pt]);
            } else {
                snprintf(hover_text, sizeof(hover_text), "Hover: %c%d - Empty", 'a' + file, rank + 1);
            }
            DrawText(hover_text, main_x, hover_info_y, 12, SKYBLUE);
            if (!is_dest_phase && valid_pieces[hovered_sq] && pc != NO_PIECE) {
                MoveList ml;
                ml.count = 0;
                generate_legal(pos, &ml, NULL, NULL);
                int move_count = 0;
                for (int m = 0; m < ml.count; m++) {
                    if (from_sq(ml.moves[m].move) == (Square)hovered_sq) move_count++;
                }
                char moves_hint[64];
                snprintf(moves_hint, sizeof(moves_hint), "  -> %d legal destination(s)", move_count);
                DrawText(moves_hint, main_x, hover_info_y + 14, 11, (Color){0, 255, 0, 200});
            }
        } else {
            DrawText("Hover over board for square info", main_x, hover_info_y, 11, GRAY);
        }
        int obs_start_x = main_x + main_board_size + 50;
        int obs_y = 20;
        int col_width = mini_cell * 8 + padding;
        const char* my_piece_labels[] = {"My Pawns", "My Knights", "My Bishops", "My Rooks", "My Queens", "My King"};
        Color piece_colors[] = {GREEN, BLUE, ORANGE, RED, PURPLE, YELLOW};
        for (int i = 0; i < 6; i++) {
            debug_draw_obs_plane_with_pieces(obs + view_buffer_offset + O_BOARD + i * 64, pos,
                                obs_start_x + i * col_width, obs_y, mini_cell,
                                my_piece_labels[i], piece_colors[i], view_player, i, 1);
        }
        obs_y += mini_cell * 8 + 30;
        const char* opp_piece_labels[] = {"Opp Pawns", "Opp Knights", "Opp Bishops", "Opp Rooks", "Opp Queens", "Opp King"};
        for (int i = 0; i < 6; i++) {
            debug_draw_obs_plane_with_pieces(obs + view_buffer_offset + O_BOARD + (6 + i) * 64, pos,
                                obs_start_x + i * col_width, obs_y, mini_cell,
                                opp_piece_labels[i], (Color){255, 100, 100, 200}, view_player, i, 0);
        }
        obs_y += mini_cell * 8 + 30;
        debug_draw_obs_plane_with_pieces(obs + active_buffer_offset + O_VALID_PIECES, pos,
                            obs_start_x, obs_y, mini_cell,
                            "Valid Pieces", (Color){0, 255, 0, 200}, view_player, -1, 1);
        debug_draw_obs_plane_with_pieces(obs + active_buffer_offset + O_VALID_DESTS, pos,
                            obs_start_x + col_width, obs_y, mini_cell,
                            "Valid Dests", (Color){0, 200, 255, 200}, view_player, -1, 0);
        debug_draw_obs_plane_with_pieces(obs + active_buffer_offset + O_SELECTED_PIECE, pos,
                            obs_start_x + col_width * 2, obs_y, mini_cell,
                            "Selected Piece", YELLOW, view_player, -1, 1);
        /*debug_draw_obs_plane_with_pieces(obs + active_buffer_offset + O_SELF_CHECK_PLANE, pos,
                            obs_start_x + col_width * 3, obs_y, mini_cell,
                            "In Check (self)", (Color){255, 50, 50, 200}, view_player, -2, 1);
        debug_draw_obs_plane_with_pieces(obs + active_buffer_offset + O_OPP_CHECK_PLANE, pos,
                            obs_start_x + col_width * 4, obs_y, mini_cell,
                            "Checking (opp)", (Color){255, 150, 50, 200}, view_player, -2, 0);
          */                  
        obs_y += mini_cell * 8 + 30;
        int promo_counts[4];
        debug_draw_promo_plane(obs + active_buffer_offset + O_VALID_PROMOS, 
                              obs_start_x, obs_y, mini_cell, "Valid Promos (Q/R/B/N x file)", promo_counts);
        int info_x = window_width - 240;
        int info_y = 20;
        DrawText("=== DEBUG INFO ===", info_x, info_y, 14, WHITE);
        info_y += 25;
        char hist_text[96];
        int is_latest = (hist_idx == env->debug_history_count - 1);
        snprintf(hist_text, sizeof(hist_text), "History: %d / %d %s", 
                 hist_idx + 1, env->debug_history_count,
                 is_latest ? "[LATEST]" : "[HISTORICAL]");
        Color hist_color = is_latest ? GREEN : YELLOW;
        DrawText(hist_text, info_x, info_y, 12, hist_color);
        info_y += 16;
        if (!is_latest) {
            DrawText("(SPACE to jump to latest)", info_x, info_y, 10, YELLOW);
        } else {
            DrawText("(SPACE to step forward)", info_x, info_y, 10, GRAY);
        }
        info_y += 18;
        DrawRectangle(info_x - 5, info_y - 2, 230, 105, (Color){35, 35, 45, 255});
        const char* learner_color_str = (env->learner_color == CHESS_WHITE) ? "WHITE" : "BLACK";
        char learner_line[64];
        snprintf(learner_line, sizeof(learner_line), "Learner: %s", learner_color_str);
        Color learner_display_color = (env->learner_color == CHESS_WHITE) ? 
            (Color){200, 200, 255, 255} : (Color){255, 200, 200, 255};
        DrawText(learner_line, info_x, info_y, 14, learner_display_color);
        info_y += 20;
        
        int current_side_to_move = pos->sideToMove;
        const char* stm_color = (current_side_to_move == CHESS_WHITE) ? "WHITE" : "BLACK";
        int is_learner_turn = (current_side_to_move == env->learner_color);
        int player_idx_for_turn = current_side_to_move;
        int pick_phase_for_turn = env->debug_pick_phase_history[hist_idx % DEBUG_HISTORY_SIZE][player_idx_for_turn];
        char status_line[128];
        if (pick_phase_for_turn == 0) {
            snprintf(status_line, sizeof(status_line), "%s: SELECT PIECE", stm_color);
        } else {
            snprintf(status_line, sizeof(status_line), "%s: SELECT DEST", stm_color);
        }
        Color status_color = is_learner_turn ? GREEN : ORANGE;
        DrawText(status_line, info_x, info_y, 18, status_color);
        info_y += 22;
        char phase_line[32];
        snprintf(phase_line, sizeof(phase_line), "Phase %d", pick_phase_for_turn);
        DrawText(phase_line, info_x, info_y, 16, LIGHTGRAY);
        info_y += 20;
        char role_line[64];
        snprintf(role_line, sizeof(role_line), "(%s)", is_learner_turn ? "LEARNER" : "OPPONENT");
        DrawText(role_line, info_x, info_y, 12, status_color);
        info_y += 18;
        int chess_moves = env->debug_chess_moves_history[hist_idx % DEBUG_HISTORY_SIZE];
        char moves_line[64];
        snprintf(moves_line, sizeof(moves_line), "Full moves: %d", chess_moves);
        DrawText(moves_line, info_x, info_y, 11, WHITE);
        info_y += 16;
        Square sel_sq = env->debug_selected_sq_history[hist_idx % DEBUG_HISTORY_SIZE][player_idx_for_turn];
        if (pick_phase_for_turn == 1 && sel_sq != SQ_NONE) {
            char sel_line[64];
            int sel_file = sel_sq % 8;
            int sel_rank = sel_sq / 8;
            Piece sel_pc = pos->board[sel_sq];
            const char* pc_char = (sel_pc != NO_PIECE) ? debug_get_piece_char(sel_pc) : "?";
            snprintf(sel_line, sizeof(sel_line), "Selected: %c%d (%s)", 'a' + sel_file, sel_rank + 1, pc_char);
            DrawText(sel_line, info_x, info_y, 11, YELLOW);
        } else {
            DrawText("Selected: none", info_x, info_y, 11, GRAY);
        }
        info_y += 22;
        const char* view_color = view_player ? "BLACK" : "WHITE";
        int view_is_learner = (view_player == env->learner_color);
        char view_text[64];
        snprintf(view_text, sizeof(view_text), "Viewing: %s (%s)", view_color, view_is_learner ? "learner" : "opponent");
        DrawText(view_text, info_x, info_y, 11, view_is_learner ? (Color){100, 255, 100, 255} : (Color){255, 180, 100, 255});
        info_y += 22;
        DrawRectangle(info_x - 5, info_y - 2, 230, 52, (Color){35, 35, 45, 255});
        DrawText("Can Castle?", info_x, info_y, 12, WHITE);
        info_y += 16;
        uint8_t* castle_onehot = obs + view_buffer_offset + O_CASTLE;
        int castle_idx = 0;
        for (int i = 0; i < 16; i++) if (castle_onehot[i]) { castle_idx = i; break; }
        int w_kingside = (castle_idx & WHITE_OO);
        int w_queenside = (castle_idx & WHITE_OOO);
        int b_kingside = (castle_idx & BLACK_OO);
        int b_queenside = (castle_idx & BLACK_OOO);
        char castle_w[64], castle_b[64];
        snprintf(castle_w, sizeof(castle_w), "  WHITE: K=%s Q=%s", 
                 w_kingside ? "YES" : "no", w_queenside ? "YES" : "no");
        snprintf(castle_b, sizeof(castle_b), "  BLACK: K=%s Q=%s", 
                 b_kingside ? "YES" : "no", b_queenside ? "YES" : "no");
        DrawText(castle_w, info_x, info_y, 10, w_kingside || w_queenside ? GREEN : GRAY);
        info_y += 12;
        DrawText(castle_b, info_x, info_y, 10, b_kingside || b_queenside ? GREEN : GRAY);
        info_y += 20;
        uint8_t* ep_onehot = obs + view_buffer_offset + O_EP;
        int ep_sq = -1;
        for (int i = 0; i < 65; i++) if (ep_onehot[i]) { ep_sq = i; break; }
        char ep_text[32];
        if (ep_sq >= 0 && ep_sq < 64) {
            snprintf(ep_text, sizeof(ep_text), "EP Square: %c%d", 'a' + (ep_sq % 8), 1 + (ep_sq / 8));
        } else {
            snprintf(ep_text, sizeof(ep_text), "EP Square: none");
        }
        DrawText(ep_text, info_x, info_y, 11, ep_sq < 64 ? SKYBLUE : GRAY);
        info_y += 18;
        int rule50 = obs[view_buffer_offset + O_RULE50];
        int rep = obs[view_buffer_offset + O_REPETITION];
        char rule_text[64];
        snprintf(rule_text, sizeof(rule_text), "Rule50: %d/100", (rule50 * 100) / 255);
        DrawText(rule_text, info_x, info_y, 11, rule50 > 200 ? ORANGE : LIGHTGRAY);
        info_y += 14;
        snprintf(rule_text, sizeof(rule_text), "Repetition: %s", rep == 0 ? "3x DRAW!" : (rep == 128 ? "2x" : "1x"));
        DrawText(rule_text, info_x, info_y, 11, (rep < 255) ? (rep == 0 ? RED : YELLOW) : LIGHTGRAY);
        info_y += 20;
        char moves_text[32];
        snprintf(moves_text, sizeof(moves_text), "Chess Moves: %d", env->debug_chess_moves_history[hist_idx % DEBUG_HISTORY_SIZE]);
        DrawText(moves_text, info_x, info_y, 12, WHITE);
        info_y += 22;
        DrawRectangle(info_x - 5, info_y - 2, 230, 85, (Color){35, 35, 45, 255});
        DrawText("Last Actions:", info_x, info_y, 12, WHITE);
        info_y += 16;
        int action0 = env->debug_actions_history[hist_idx % DEBUG_HISTORY_SIZE][0];
        int action1 = env->debug_actions_history[hist_idx % DEBUG_HISTORY_SIZE][1];
        int was_learner_turn = (pos->sideToMove == env->learner_color);
        char a0_desc[128];
        if (action0 == PASS_ACTION) {
            snprintf(a0_desc, sizeof(a0_desc), "Learner: PASS");
        } else if (action0 < 64) {
            int a0_file = action0 % 8;
            int a0_rank = action0 / 8;
            Piece a0_pc = pos->board[action0];
            const char* pc_name = (a0_pc != NO_PIECE) ? debug_get_piece_char(a0_pc) : "?";
            snprintf(a0_desc, sizeof(a0_desc), "Learner: %d = %c%d (%s)", 
                     action0, 'a' + a0_file, a0_rank + 1, pc_name);
        } else if (action0 < 96) {
            int promo_type = (action0 - 64) / 8;
            int promo_file = (action0 - 64) % 8;
            const char* promo_names[] = {"Queen", "Rook", "Bishop", "Knight"};
            snprintf(a0_desc, sizeof(a0_desc), "Learner: %d = Promo %s @%c-file", 
                     action0, promo_names[promo_type], 'a' + promo_file);
        } else {
            snprintf(a0_desc, sizeof(a0_desc), "Learner: %d (invalid)", action0);
        }
        Color a0_color = was_learner_turn ? (Color){100, 255, 100, 255} : (Color){120, 120, 120, 255};
        DrawText(a0_desc, info_x, info_y, 10, a0_color);
        info_y += 14;
        const char* a0_turn = was_learner_turn ? "ACTIVE" : "not their turn";
        char a0_context[64];
        snprintf(a0_context, sizeof(a0_context), "    (%s)", a0_turn);
        DrawText(a0_context, info_x, info_y, 9, was_learner_turn ? GREEN : DARKGRAY);
        info_y += 14;
        if (env->selfplay) {
            int was_opp_turn = !was_learner_turn;
            char a1_desc[128];
            if (action1 == PASS_ACTION) {
                snprintf(a1_desc, sizeof(a1_desc), "Opponent: PASS");
            } else if (action1 >= 0 && action1 < 64) {
                int a1_file = action1 % 8;
                int a1_rank = action1 / 8;
                Piece a1_pc = pos->board[action1];
                const char* pc_name = (a1_pc != NO_PIECE) ? debug_get_piece_char(a1_pc) : "?";
                snprintf(a1_desc, sizeof(a1_desc), "Opponent: %d = %c%d (%s)", 
                         action1, 'a' + a1_file, a1_rank + 1, pc_name);
            } else if (action1 >= 64 && action1 < 96) {
                int promo_type = (action1 - 64) / 8;
                int promo_file = (action1 - 64) % 8;
                const char* promo_names[] = {"Queen", "Rook", "Bishop", "Knight"};
                snprintf(a1_desc, sizeof(a1_desc), "Opponent: %d = Promo %s @%c-file", 
                         action1, promo_names[promo_type], 'a' + promo_file);
            } else {
                snprintf(a1_desc, sizeof(a1_desc), "Opponent: %d (none)", action1);
            }
            Color a1_color = was_opp_turn ? (Color){255, 200, 100, 255} : (Color){120, 120, 120, 255};
            DrawText(a1_desc, info_x, info_y, 10, a1_color);
            info_y += 14;
            const char* a1_turn = was_opp_turn ? "ACTIVE" : "not their turn";
            char a1_context[64];
            snprintf(a1_context, sizeof(a1_context), "    (%s)", a1_turn);
            DrawText(a1_context, info_x, info_y, 9, was_opp_turn ? ORANGE : DARKGRAY);
            info_y += 14;
        }
        Move last_move = env->debug_last_move_history[hist_idx % DEBUG_HISTORY_SIZE];
        if (last_move != MOVE_NONE) {
            char move_str[32];
            Square from = from_sq(last_move);
            Square to = to_sq(last_move);
            snprintf(move_str, sizeof(move_str), "Last move: %c%d-%c%d",
                     'a' + (from % 8), 1 + (from / 8),
                     'a' + (to % 8), 1 + (to / 8));
            DrawText(move_str, info_x, info_y, 11, (Color){186, 202, 68, 255});
            info_y += 16;
        }
        info_y += 6;
        DrawRectangle(info_x - 5, info_y - 2, 230, 62, (Color){35, 35, 45, 255});
        DrawText("Reward (Training):", info_x, info_y, 12, WHITE);
        info_y += 16;
        float step_reward = env->debug_rewards_history[hist_idx % DEBUG_HISTORY_SIZE];
        char reward_str[64];
        snprintf(reward_str, sizeof(reward_str), "  Total: %.6f", step_reward);
        Color reward_color;
        if (step_reward > 0.0001f) {
            reward_color = GREEN;
        } else if (step_reward < -0.0001f) {
            reward_color = RED;
        } else {
            reward_color = LIGHTGRAY;
        }
        DrawText(reward_str, info_x, info_y, 12, reward_color);
        info_y += 14;
        char reward_config[96];
        snprintf(reward_config, sizeof(reward_config), "  mat:%.4f pos:%.5f", 
                 env->reward_material, env->reward_position);
        DrawText(reward_config, info_x, info_y, 9, DARKGRAY);
        info_y += 10;
        snprintf(reward_config, sizeof(reward_config), "  rep:%.4f", env->reward_repetition);
        DrawText(reward_config, info_x, info_y, 9, DARKGRAY);
        info_y += 16;
        int ctrl_y = window_height - 130;
        DrawRectangle(info_x - 5, ctrl_y - 5, 230, 105, (Color){30, 30, 40, 255});
        DrawText("=== CONTROLS ===", info_x, ctrl_y, 12, WHITE);
        ctrl_y += 18;
        DrawText("[<-/->] Navigate history", info_x, ctrl_y, 10, LIGHTGRAY); ctrl_y += 13;
        DrawText("[SPACE] Step to next state", info_x, ctrl_y, 10, LIGHTGRAY); ctrl_y += 13;
        DrawText("[V] Toggle view (W/B)", info_x, ctrl_y, 10, LIGHTGRAY); ctrl_y += 13;
        DrawText("[R] Reset history", info_x, ctrl_y, 10, LIGHTGRAY); ctrl_y += 13;
        DrawText("[ESC] Exit", info_x, ctrl_y, 10, LIGHTGRAY); ctrl_y += 18;
        if (env->debug_paused) {
            DrawRectangle(info_x, window_height - 32, 90, 26, MAROON);
            DrawText("PAUSED", info_x + 15, window_height - 27, 16, WHITE);
        } else {
            DrawRectangle(info_x, window_height - 32, 90, 26, DARKGREEN);
            DrawText("RUNNING", info_x + 10, window_height - 27, 14, WHITE);
        }
        int mask_y = obs_y + mini_cell * 4 + 45;
        int pass_valid = obs[active_buffer_offset + O_PASS_VALID] > 0;
        int valid_sq_count = 0, valid_promo_count = 0;
        if (!pass_valid) {
            for (int i = 0; i < 64; i++) {
                if (is_dest_phase ? valid_dests[i] : valid_pieces[i]) valid_sq_count++;
            }
            for (int i = 0; i < 32; i++) {
                if (valid_promos[i]) valid_promo_count++;
            }
        }
        char mask_title[128];
        if (pass_valid) {
            snprintf(mask_title, sizeof(mask_title), "ACTION MASK: PASS only (not your turn)");
        } else {
            snprintf(mask_title, sizeof(mask_title), "ACTION MASK: %d squares + %d promos = %d valid", 
                     valid_sq_count, valid_promo_count, valid_sq_count + valid_promo_count);
        }
        DrawText(mask_title, obs_start_x, mask_y, 12, pass_valid ? YELLOW : SKYBLUE);
        mask_y += 18;
        for (int i = 0; i < 97; i++) {
            int x_pos = obs_start_x + i * 4;
            int is_valid = 0;
            if (i < 64) {
                is_valid = pass_valid ? 0 : (is_dest_phase ? valid_dests[i] : valid_pieces[i]);
            } else if (i < 96) {
                is_valid = pass_valid ? 0 : valid_promos[i - 64];
            } else {
                is_valid = pass_valid;
            }
            Color bar_color = is_valid ? GREEN : (Color){40, 40, 40, 255};
            if (i == 96 && is_valid) bar_color = YELLOW;
            DrawRectangle(x_pos, mask_y, 3, 20, bar_color);
            if (i == 63) {
                DrawLine(x_pos + 5, mask_y - 5, x_pos + 5, mask_y + 25, WHITE);
            }
            if (i == 95) {
                DrawLine(x_pos + 5, mask_y - 5, x_pos + 5, mask_y + 25, WHITE);
            }
        }
        const char* sq_label = is_dest_phase ? "Destinations (0-63)" : "Pieces (0-63)";
        DrawText(sq_label, obs_start_x, mask_y + 24, 10, is_dest_phase ? SKYBLUE : GREEN);
        DrawText("Promos (64-95)", obs_start_x + 255, mask_y + 24, 10, PURPLE);
        DrawText("PASS", obs_start_x + 380, mask_y + 24, 10, YELLOW);
        EndDrawing();
        if (!env->debug_paused) {
            break;
        }
        if (WindowShouldClose()) {
            debug_unload_piece_font();
            CloseWindow();
            exit(0);
        }
        int key = GetKeyPressed();
        while (key != 0) {
            if (key == KEY_ESCAPE) {
                debug_unload_piece_font();
                CloseWindow();
                exit(0);
            }
            if (key == KEY_LEFT) {
                if (env->debug_history_idx > 0) {
                    env->debug_history_idx--;
                }
            }
            if (key == KEY_RIGHT) {
                if (env->debug_history_idx < env->debug_history_count - 1) {
                    env->debug_history_idx++;
                } else {
                    needs_step = 1;
                    break;
                }
            }
            if (key == KEY_V) {
                env->debug_view_player = !env->debug_view_player;
            }
            if (key == KEY_R) {
                c_reset(env);
                env->debug_history_count = 0;
                env->debug_history_idx = 0;
                break;
            }
            if (key == KEY_SPACE || key == KEY_S) {
                if (env->debug_history_idx < env->debug_history_count - 1) {
                    env->debug_history_idx = env->debug_history_count - 1;
                } else {
                    needs_step = 1;
                    break;
                }
            }
            key = GetKeyPressed();
        }
        if (IsKeyPressedRepeat(KEY_LEFT)) {
            if (env->debug_history_idx > 0) {
                env->debug_history_idx--;
            }
        }
        if (IsKeyPressedRepeat(KEY_RIGHT)) {
            if (env->debug_history_idx < env->debug_history_count - 1) {
                env->debug_history_idx++;
            }
        }
        if (needs_step) break;
    } while (env->debug_paused);
    env->debug_paused = 1;
}

static int get_valid_action(Chess* env, int player_idx) {
    int buffer_idx;
    if (env->learner_color == CHESS_WHITE) {
        buffer_idx = player_idx;
    } else {
        buffer_idx = 1 - player_idx;
    }
    uint8_t* player_obs = env->observations + (buffer_idx * OBS_SIZE);
    uint8_t* valid_pieces = player_obs + O_VALID_PIECES;
    uint8_t* valid_dests = player_obs + O_VALID_DESTS;
    uint8_t* valid_promos = player_obs + O_VALID_PROMOS;
    uint8_t pass_valid = player_obs[O_PASS_VALID];
    if (pass_valid) {
        return 96;
    }
    int pick_phase = env->pick_phase[player_idx];
    if (pick_phase == 0) {
        int valid_squares[64];
        int num_valid = 0;
        for (int sq = 0; sq < 64; sq++) {
            if (valid_pieces[sq]) {
                valid_squares[num_valid++] = sq;
            }
        }
        if (num_valid > 0) {
            return valid_squares[rand() % num_valid];
        }
    } else {
        int valid_promos_list[32];
        int num_promos = 0;
        for (int i = 0; i < 32; i++) {
            if (valid_promos[i]) {
                valid_promos_list[num_promos++] = 64 + i;
            }
        }
        int valid_dest_squares[64];
        int num_dests = 0;
        for (int sq = 0; sq < 64; sq++) {
            if (valid_dests[sq]) {
                valid_dest_squares[num_dests++] = sq;
            }
        }
        int total_valid = num_promos + num_dests;
        if (total_valid > 0) {
            int choice = rand() % total_valid;
            if (choice < num_promos) {
                return valid_promos_list[choice];
            } else {
                return valid_dest_squares[choice - num_promos];
            }
        }
    }
    return rand() % 64;
}

void pos_to_fen(Position* pos, char* fen_out) {
    char* ptr = fen_out;
    const char piece_chars[] = " PNBRQK  pnbrqk";
    
    for (int rank = 7; rank >= 0; rank--) {
        int empty = 0;
        for (int file = 0; file < 8; file++) {
            Square sq = rank * 8 + file;
            Piece p = pos->board[sq];
            if (p == NO_PIECE) {
                empty++;
            } else {
                if (empty > 0) {
                    *ptr++ = '0' + empty;
                    empty = 0;
                }
                *ptr++ = piece_chars[p];
            }
        }
        if (empty > 0) *ptr++ = '0' + empty;
        if (rank > 0) *ptr++ = '/';
    }
    
    *ptr++ = ' ';
    *ptr++ = (pos->sideToMove == CHESS_WHITE) ? 'w' : 'b';
    *ptr++ = ' ';
    
    int has_castling = 0;
    if (pos->castlingRights & 1) { *ptr++ = 'K'; has_castling = 1; }
    if (pos->castlingRights & 2) { *ptr++ = 'Q'; has_castling = 1; }
    if (pos->castlingRights & 4) { *ptr++ = 'k'; has_castling = 1; }
    if (pos->castlingRights & 8) { *ptr++ = 'q'; has_castling = 1; }
    if (!has_castling) *ptr++ = '-';
    
    *ptr++ = ' ';
    if (pos->epSquare != SQ_NONE) {
        *ptr++ = 'a' + (pos->epSquare % 8);
        *ptr++ = '1' + (pos->epSquare / 8);
    } else {
        *ptr++ = '-';
    }
    
    *ptr++ = ' ';
    *ptr++ = '0';
    *ptr++ = ' ';
    *ptr++ = '1';
    *ptr = '\0';
}

float read_config_float(const char* section, const char* key, float default_val) {
    FILE* f = fopen("../../../config/ocean/chess.ini", "r");
    if (!f) return default_val;
    
    char line[256];
    int in_section = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '[') {
            in_section = (strstr(line, section) != NULL);
        } else if (in_section && strstr(line, key)) {
            char* eq = strchr(line, '=');
            if (eq) {
                float val = atof(eq + 1);
                fclose(f);
                return val;
            }
        }
    }
    fclose(f);
    return default_val;
}

void interactive() {
    float reward_material = read_config_float("[env]", "reward_material", 0.1f);
    float reward_position = read_config_float("[env]", "reward_position", 0.01f);
    float reward_repetition = read_config_float("[env]", "reward_repetition", 0.002f);
    
    printf("Loaded config: material=%.6f position=%.6f repetition=%.6f\n", 
           reward_material, reward_position, reward_repetition);
    
    Chess env = {
        .max_moves = 50000,
        .reward_draw = 0.0f,
        .reward_invalid_piece = 0.0f,
        .reward_invalid_move = 0.0f,
        .reward_valid_piece = 0.0f,
        .reward_valid_move = 0.0f,
        .reward_material = reward_material,
        .reward_position = reward_position,
        .reward_castling = 0.0f,
        .reward_repetition = reward_repetition,
        .render_fps = 30,
        .selfplay = 1,
        .human_play = 0,
        .debug_mode = 1,
        .enable_50_move_rule = 1,
        .enable_threefold_repetition = 1,
        .learner_color = CHESS_WHITE,
        .client = NULL,
        .fen_curriculum = NULL,
        .num_fens = 0,
        .random_fen = 0,
        .log_pgn = 0,
        .log_pgn_choice_made = 1,
    };
    env.observations = (uint8_t*)calloc(OBS_SIZE * 2, sizeof(uint8_t));
    env.actions = (int*)calloc(2, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    init_bitboards();
    strcpy(env.starting_fen, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    env.debug_paused = 1;
    env.debug_history_idx = 0;
    env.debug_history_count = 0;
    env.debug_view_player = env.learner_color;
    env.debug_selected_plane = -1;
    c_reset(&env);
    env.debug_history_count = 0;
    env.debug_history_idx = 0;
    
    int game_number = 0;
    printf("\n========================================\n");
    printf("GAME %d - Learner plays as: %s\n", ++game_number, 
           env.learner_color == CHESS_WHITE ? "WHITE" : "BLACK");
    printf("========================================\n");
    
    while (1) {
        c_render_debug(&env);
        if (WindowShouldClose()) break;
        int current_player = env.pos.sideToMove;
        if (current_player == env.learner_color) {
            env.actions[0] = get_valid_action(&env, current_player);
            env.actions[1] = 96;
        } else {
            env.actions[0] = 96;
            env.actions[1] = get_valid_action(&env, current_player);
        }
        
        const char* player_name = (current_player == CHESS_WHITE) ? "White" : "Black";
        int action = (current_player == env.learner_color) ? env.actions[0] : env.actions[1];
        int phase_before = env.pick_phase[current_player];
        
        char fen_before[128];
        pos_to_fen(&env.pos, fen_before);
        
        printf("\n=== Move %d | %s to move | Phase %d ===\n", env.chess_moves + 1, player_name, phase_before);
        printf("FEN: %s\n", fen_before);
        if (action < 64) {
            int file = action % 8;
            int rank = action / 8;
            printf("Action: %d = %c%d\n", action, 'a' + file, rank + 1);
        } else if (action < 96) {
            int promo_type = (action - 64) / 8;
            int promo_file = (action - 64) % 8;
            const char* promo_names[] = {"Queen", "Rook", "Bishop", "Knight"};
            printf("Action: %d = Promo %s @%c-file\n", action, promo_names[promo_type], 'a' + promo_file);
        } else {
            printf("Action: PASS\n");
        }
        
        float reward_before = env.rewards[0];
        int16_t mat_before = env.pos.materialScore;
        int16_t pst_before = env.pos.psqtScore;
        
        c_step(&env);
        
        float reward_after = env.rewards[0];
        int16_t mat_after = env.pos.materialScore;
        int16_t pst_after = env.pos.psqtScore;
        
          int mat_delta = mat_after - mat_before;
        int pst_delta = pst_after - pst_before;
        printf("Material: %d -> %d (delta: %d)\n", mat_before, mat_after, mat_delta);
        printf("Position: %d -> %d (delta: %d)\n", pst_before, pst_after, pst_delta);
        
        int phase_after = env.pick_phase[current_player];
        printf("Phase: %d → %d", phase_before, phase_after);
        if (phase_before == 0 && phase_after == 1) {
            printf(" (piece selected: %c%d)", 
                   'a' + (env.selected_square[current_player] % 8),
                   1 + (env.selected_square[current_player] / 8));
        } else if (phase_before == 1 && phase_after == 0) {
            printf(" (move completed)");
        }
        printf("\n");
        
        if (mat_delta != 0 || pst_delta != 0) {
            float raw_mat = (float)mat_delta / 900.0f * env.reward_material;
            float raw_pos = (float)pst_delta / 1000.0f * env.reward_position;
            
            if (env.learner_color == CHESS_BLACK) {
                raw_mat = -raw_mat;
                raw_pos = -raw_pos;
            }
            
            
            if (mat_delta != 0) {
                raw_pos = 0.0f;
            }
            
            float mat_contribution = 0.0f;
            float pos_contribution = 0.0f;
            
            if (current_player == env.learner_color) {
                if (env.reward_material != 0.0f) {
                    if (raw_mat > 0) {
                        if (env.last_see_value >= 0) {
                            mat_contribution = raw_mat;
                        } else {
                            mat_contribution = 0.0f;
                        }
                    } else {
                        mat_contribution = raw_mat;
                    }
                }
                pos_contribution = raw_pos;
            } else {
                mat_contribution = raw_mat;
                pos_contribution = raw_pos;
            }
            
            printf("  Material delta: %d cp → mat_reward: %.6f\n", mat_delta, mat_contribution);
            printf("  Position delta: %d cp → pos_reward: %.6f\n", pst_delta, pos_contribution);
            printf("  SEE value: %d\n", env.last_see_value);
            printf("  Mover: %s, Learner: %s\n", 
                   current_player == CHESS_WHITE ? "WHITE" : "BLACK",
                   env.learner_color == CHESS_WHITE ? "WHITE" : "BLACK");
            
            char fen_after[128];
            pos_to_fen(&env.pos, fen_after);
            printf("  FEN after reward: %s\n", fen_after);
        }
        
        if (current_player == env.learner_color && env.last_see_value < 0 && env.reward_material != 0.0f) {
            float hanging_penalty = (float)env.last_see_value / 900.0f * env.reward_material;
            printf("SEE=%d → penalty: %.6f\n", env.last_see_value, hanging_penalty);
        }
        printf("Total Reward: %.6f\n", reward_after);
        
        if (env.last_move != MOVE_NONE) {
            Square from = from_sq(env.last_move);
            Square to = to_sq(env.last_move);
            printf("Chess move executed: %c%d-%c%d\n", 
                   'a' + (from % 8), 1 + (from / 8),
                   'a' + (to % 8), 1 + (to / 8));
            
            if (env.reward_repetition != 0.0f && current_player == env.learner_color && env.undo_stack_ptr >= 4) {
                uint8_t plies = env.undo_stack[env.undo_stack_ptr - 1].pliesFromNull;
                if (plies >= 4) {
                    Key current_key = env.pos.key;
                    for (int i = 4; i <= plies; i += 2) {
                        int idx = env.undo_stack_ptr - i;
                        if (idx >= 0 && env.undo_stack[idx].key == current_key) {
                            printf("Repetition penalty: %.6f\n", env.reward_repetition);
                            break;
                        }
                    }
                }
            }
        }
        
        if (env.terminals[0]) {
            printf("\n=== GAME OVER: %s ===\n", env.last_result);
            printf("Final score - White: %.1f, Black: %.1f\n", env.white_score, env.black_score);
            printf("\n========================================\n");
            printf("GAME %d - Learner plays as: %s\n", ++game_number, 
                   env.learner_color == CHESS_WHITE ? "WHITE" : "BLACK");
            printf("========================================\n");
        }
        
        debug_save_state(&env);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

void performance_test() {
    long test_time = 10;
    Chess env = {
        .max_moves = 500,
        .reward_draw = 0.0f,
        .reward_invalid_piece = -0.1f,
        .reward_invalid_move = -0.1f,
        .reward_valid_piece = 0.0f,
        .reward_valid_move = 0.0f,
        .reward_material = 0.0f,
        .reward_position = 0.0f,
        .reward_castling = 0.0f,
        .reward_repetition = 0.0f,
        .render_fps = 0,
        .selfplay = 1,
        .human_play = 0,
        .debug_mode = 0,
        .enable_50_move_rule = 1,
        .enable_threefold_repetition = 1,
        .learner_color = CHESS_WHITE,
        .client = NULL,
        .fen_curriculum = NULL,
        .num_fens = 0,
        .random_fen = 0,
        .log_pgn = 0,
        .log_pgn_choice_made = 1,
    };
    env.observations = (uint8_t*)calloc(OBS_SIZE * 2, sizeof(uint8_t));
    env.actions = (int*)calloc(2, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    init_bitboards();
    strcpy(env.starting_fen, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    c_reset(&env);
    populate_observations(&env);
    long start = time(NULL);
    long i = 0;
    while (time(NULL) - start < test_time) {
        int current_player = env.pos.sideToMove;
        if (current_player == env.learner_color) {
            env.actions[0] = get_valid_action(&env, current_player);
            env.actions[1] = 96;
        } else {
            env.actions[0] = 96;
            env.actions[1] = get_valid_action(&env, current_player);
        }
        c_step(&env);
        i++;
    }
    long end = time(NULL);
    long elapsed = end - start;
    if (elapsed > 0) {
        printf("Chess SPS: %ld\n", i / elapsed);
    } else {
        printf("Chess steps: %ld (test too short)\n", i);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
}

int main(int argc, char* argv[]) {
    srand((unsigned int)time(NULL));
    if (argc > 1 && strcmp(argv[1], "perf") == 0) {
        performance_test();
    } else if (argc > 1 && strcmp(argv[1], "auto") == 0) {
        float reward_material = read_config_float("[env]", "reward_material", 0.1f);
        float reward_position = read_config_float("[env]", "reward_position", 0.01f);
        float reward_repetition = read_config_float("[env]", "reward_repetition", 0.002f);
        
        printf("Loaded config: material=%.6f position=%.6f repetition=%.6f\n", 
               reward_material, reward_position, reward_repetition);
        
        Chess env = {
            .max_moves = 100,
            .reward_draw = 0.0f,
            .reward_invalid_piece = 0.0f,
            .reward_invalid_move = 0.0f,
            .reward_valid_piece = 0.0f,
            .reward_valid_move = 0.0f,
            .reward_material = reward_material,
            .reward_position = reward_position,
            .reward_castling = 0.0f,
            .reward_repetition = reward_repetition,
            .selfplay = 1,
            .human_play = 0,
            .debug_mode = 0,
            .enable_50_move_rule = 1,
            .enable_threefold_repetition = 1,
            .learner_color = CHESS_WHITE,
            .client = NULL,
            .fen_curriculum = NULL,
            .num_fens = 0,
            .random_fen = 0,
            .log_pgn = 0,
            .log_pgn_choice_made = 1,
        };
        env.observations = (uint8_t*)calloc(OBS_SIZE * 2, sizeof(uint8_t));
        env.actions = (int*)calloc(2, sizeof(int));
        env.rewards = (float*)calloc(1, sizeof(float));
        env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
        init_bitboards();
        strcpy(env.starting_fen, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        
        c_reset(&env);
        
        printf("\n========================================\n");
        printf("AUTO-PLAY MODE - Learner: %s\n", env.learner_color == CHESS_WHITE ? "WHITE" : "BLACK");
        printf("========================================\n");
        
        int move_count = 0;
        while (!env.terminals[0] && move_count < 50) {
            int current_player = env.pos.sideToMove;
            const char* player_name = (current_player == CHESS_WHITE) ? "White" : "Black";
            int action = (current_player == env.learner_color) ? env.actions[0] : env.actions[1];
            
            if (current_player == env.learner_color) {
                env.actions[0] = get_valid_action(&env, current_player);
                env.actions[1] = 96;
                action = env.actions[0];
            } else {
                env.actions[0] = 96;
                env.actions[1] = get_valid_action(&env, current_player);
                action = env.actions[1];
            }
            
            printf("\n=== Move %d | %s to move ===\n", env.chess_moves + 1, player_name);
            
            float reward_before = env.rewards[0];
            int16_t mat_before = env.pos.materialScore;
            int16_t pst_before = env.pos.psqtScore;
            
            c_step(&env);
            
            float reward_after = env.rewards[0];
            int16_t mat_after = env.pos.materialScore;
            int16_t pst_after = env.pos.psqtScore;
            
            int mat_delta = mat_after - mat_before;
            int pst_delta = pst_after - pst_before;
            
            if (mat_delta != 0 || pst_delta != 0) {
                float raw_mat = (float)mat_delta / 900.0f * env.reward_material;
                float raw_pos = (float)pst_delta / 1000.0f * env.reward_position;
                
                if (env.learner_color == CHESS_BLACK) {
                    raw_mat = -raw_mat;
                    raw_pos = -raw_pos;
                }
                
                if (mat_delta != 0) {
                    raw_pos = 0.0f;
                }
                
                float mat_contribution = 0.0f;
                float pos_contribution = 0.0f;
                
                if (current_player == env.learner_color) {
                    if (env.reward_material != 0.0f) {
                        if (raw_mat > 0) {
                            if (env.last_see_value >= 0) {
                                mat_contribution = raw_mat;
                            } else {
                                mat_contribution = 0.0f;
                            }
                        } else {
                            mat_contribution = raw_mat;
                        }
                    }
                    pos_contribution = raw_pos;
                } else {
                    mat_contribution = raw_mat;
                    pos_contribution = raw_pos;
                }
                
                printf("  Material delta: %d cp → mat_reward: %.6f\n", mat_delta, mat_contribution);
                printf("  Position delta: %d cp → pos_reward: %.6f\n", pst_delta, pos_contribution);
                printf("  SEE value: %d\n", env.last_see_value);
            }
            
            if (env.last_move != MOVE_NONE) {
                Square from = from_sq(env.last_move);
                Square to = to_sq(env.last_move);
                printf("Chess move: %c%d-%c%d\n", 
                       'a' + (from % 8), 1 + (from / 8),
                       'a' + (to % 8), 1 + (to / 8));
                
                if (env.reward_repetition != 0.0f && current_player == env.learner_color && env.undo_stack_ptr >= 4) {
                    uint8_t plies = env.undo_stack[env.undo_stack_ptr - 1].pliesFromNull;
                    if (plies >= 4) {
                        Key current_key = env.pos.key;
                        for (int i = 4; i <= plies; i += 2) {
                            int idx = env.undo_stack_ptr - i;
                            if (idx >= 0 && env.undo_stack[idx].key == current_key) {
                                printf("Repetition penalty: %.6f\n", env.reward_repetition);
                                break;
                            }
                        }
                    }
                }
            }
            
            printf("Total Reward: %.6f\n", reward_after);
            move_count++;
        }
        
        if (env.terminals[0]) {
            printf("\n=== GAME OVER: %s ===\n", env.last_result);
        }
        
        free(env.observations);
        free(env.actions);
        free(env.rewards);
        free(env.terminals);
    } else {
        interactive();
    }
    return 0;
}
