// render.h - Pokemon Showdown-style battle UI renderer using Raylib
// Included at the end of poke_battle.h after all game types are defined.

#ifndef POKE_BATTLE_RENDER_H
#define POKE_BATTLE_RENDER_H

#include "raylib.h"
#include <stdio.h>
#include <string.h>

// ============================================================================
// Window & Layout Constants
// ============================================================================

#define WIN_W           960
#define WIN_H           640
#define BATTLE_W        660
#define BATTLE_H        360
#define PANEL_Y         370
#define PANEL_H         270
#define LOG_X           660
#define LOG_W           300
#define LOG_MAX_LINES   20
#define LOG_LINE_LEN    80

// ============================================================================
// Showdown-Accurate Colors
// ============================================================================

#define CLR_HP_GREEN    (Color){0x00, 0xBB, 0x51, 0xFF}
#define CLR_HP_YELLOW   (Color){0xF5, 0xD5, 0x38, 0xFF}
#define CLR_HP_RED      (Color){0xEE, 0x49, 0x28, 0xFF}
#define CLR_HP_BG       (Color){0x33, 0x33, 0x33, 0xFF}

#define CLR_BRN         (Color){0xEE, 0x55, 0x33, 0xFF}
#define CLR_PSN         (Color){0xA4, 0x00, 0x9A, 0xFF}
#define CLR_SLP         (Color){0xAA, 0x77, 0xAA, 0xFF}
#define CLR_PAR         (Color){0x9A, 0xA4, 0x00, 0xFF}
#define CLR_FRZ         (Color){0x00, 0x9A, 0xA4, 0xFF}
#define CLR_TOX         (Color){0xA4, 0x00, 0x9A, 0xFF}

#define CLR_PANEL_BG    (Color){0x33, 0x3E, 0x49, 0xFF}
#define CLR_LOG_BG      (Color){0xF3, 0xF0, 0xE7, 0xFF}
#define CLR_SHADOW      (Color){0x00, 0x00, 0x00, 0x40}

// Battle field
#define CLR_FIELD_TOP   (Color){0x82, 0xAA, 0x64, 0xFF}
#define CLR_FIELD_BOT   (Color){0x6B, 0x8E, 0x50, 0xFF}
#define CLR_EARTH       (Color){0xC4, 0xA8, 0x6B, 0xFF}

// Type colors (Showdown CSS)
static const Color TYPE_COLORS[NUM_TYPES] = {
    [TYPE_NORMAL]   = (Color){0xA8, 0xA7, 0x7A, 0xFF},
    [TYPE_FIRE]     = (Color){0xEE, 0x81, 0x30, 0xFF},
    [TYPE_WATER]    = (Color){0x63, 0x90, 0xF0, 0xFF},
    [TYPE_ELECTRIC] = (Color){0xF7, 0xD0, 0x2C, 0xFF},
    [TYPE_GRASS]    = (Color){0x7A, 0xC7, 0x4C, 0xFF},
    [TYPE_ICE]      = (Color){0x96, 0xD9, 0xD6, 0xFF},
    [TYPE_FIGHTING] = (Color){0xC2, 0x2E, 0x28, 0xFF},
    [TYPE_POISON]   = (Color){0xA3, 0x3E, 0xA1, 0xFF},
    [TYPE_GROUND]   = (Color){0xE2, 0xBF, 0x65, 0xFF},
    [TYPE_FLYING]   = (Color){0xA9, 0x8F, 0xF3, 0xFF},
    [TYPE_PSYCHIC]  = (Color){0xF9, 0x55, 0x87, 0xFF},
    [TYPE_BUG]      = (Color){0xA6, 0xB9, 0x1A, 0xFF},
    [TYPE_ROCK]     = (Color){0xB6, 0xA1, 0x36, 0xFF},
    [TYPE_GHOST]    = (Color){0x73, 0x57, 0x97, 0xFF},
    [TYPE_DRAGON]   = (Color){0x6F, 0x35, 0xFC, 0xFF},
};

static const char* TYPE_NAMES[NUM_TYPES + 1] = {
    "Normal", "Fire", "Water", "Electric", "Grass",
    "Ice", "Fighting", "Poison", "Ground", "Flying",
    "Psychic", "Bug", "Rock", "Ghost", "Dragon",
    "---",
};

// ============================================================================
// Species Name Mapping (matches SPECIES_DATA order, 1-indexed)
// ============================================================================

static const char* RENDER_SPECIES_NAMES[NUM_SPECIES + 1] = {
    NULL,
    "tauros", "chansey", "snorlax", "alakazam", "exeggutor",
    "starmie", "gengar", "jynx", "zapdos", "rhydon",
    "cloyster", "golem", "lapras", "slowbro", "jolteon",
    "persian", "hypno", "articuno", "dragonite", "machamp",
};

// ============================================================================
// Client Struct
// ============================================================================

struct Client {
    Texture2D front_sprites[NUM_SPECIES + 1];
    Texture2D back_sprites[NUM_SPECIES + 1];
    int sprites_loaded;

    // Battle log ring buffer
    char log_lines[LOG_MAX_LINES][LOG_LINE_LEN];
    Color log_colors[LOG_MAX_LINES];
    int log_head;
    int log_count;

    // State tracking for log generation
    int prev_turn;
    int prev_p1_hp[NUM_POKEMON];
    int prev_p2_hp[NUM_POKEMON];
    int prev_p1_status[NUM_POKEMON];
    int prev_p2_status[NUM_POKEMON];
    int prev_p1_active;
    int prev_p2_active;
    int prev_p1_alive[NUM_POKEMON];
    int prev_p2_alive[NUM_POKEMON];

    // Action tracking for move log
    int prev_p1_action;
    int prev_p2_action;

    // UI state
    int show_switch_panel;
    int hover_action;

    // Game result overlay
    int show_result;
    char result_text[64];

    // Player controller labels (set by demo before each render)
    char p1_label[32];
    char p2_label[32];
};

// ============================================================================
// Log Helpers
// ============================================================================

static void log_add(Client* client, const char* text, Color color) {
    int idx = (client->log_head + client->log_count) % LOG_MAX_LINES;
    if (client->log_count >= LOG_MAX_LINES) {
        client->log_head = (client->log_head + 1) % LOG_MAX_LINES;
    } else {
        client->log_count++;
    }
    strncpy(client->log_lines[idx], text, LOG_LINE_LEN - 1);
    client->log_lines[idx][LOG_LINE_LEN - 1] = '\0';
    client->log_colors[idx] = color;
}

static const char* species_name(int id) {
    if (id <= SPECIES_NONE || id > NUM_SPECIES) return "???";
    return SPECIES_DATA[id].name;
}

static const char* move_name(int id) {
    if (id <= MOVE_NONE || id > NUM_MOVES) return "Struggle";
    return MOVE_DATA[id].name;
}

static const char* status_verb(int status) {
    switch (status) {
        case STATUS_BURN: return "burned";
        case STATUS_POISON: return "poisoned";
        case STATUS_TOXIC: return "badly poisoned";
        case STATUS_PARALYSIS: return "paralyzed";
        case STATUS_SLEEP: return "put to sleep";
        case STATUS_FREEZE: return "frozen solid";
        default: return "afflicted";
    }
}

static void update_battle_log(Client* client, PokeBattle* env) {
    // Event-driven log: read from event buffer written during c_step
    if (env->event_count == 0) return;

    // Turn header
    if (env->battle.turn > client->prev_turn) {
        char buf[LOG_LINE_LEN];
        snprintf(buf, LOG_LINE_LEN, "--- Turn %d ---", env->battle.turn);
        log_add(client, buf, (Color){0x88, 0x99, 0xAA, 0xFF});
    }

    char buf[LOG_LINE_LEN];
    for (int i = 0; i < env->event_count; i++) {
        BattleEvent* e = &env->events[i];
        const char* pn = (e->data1 == 0) ? "P1" : "P2";

        switch (e->type) {
        case EVT_MOVE_USED:
            snprintf(buf, LOG_LINE_LEN, "%s %s used %s!",
                     pn, species_name(e->data3), move_name(e->data2));
            log_add(client, buf, (e->data1 == 1) ?
                    (Color){0xCC, 0xCC, 0xCC, 0xFF} : WHITE);
            break;
        case EVT_MOVE_MISSED:
            log_add(client, "It missed!", (Color){0xAA, 0xAA, 0x66, 0xFF});
            break;
        case EVT_IMMUNE: {
            // data1=attacker, data3=defender species; defender is the other player
            const char* def_pn = (e->data1 == 0) ? "P2" : "P1";
            snprintf(buf, LOG_LINE_LEN, "It doesn't affect %s %s...",
                     def_pn, species_name(e->data3));
            log_add(client, buf, (Color){0xAA, 0xAA, 0x66, 0xFF});
            break;
        }
        case EVT_CRITICAL:
            log_add(client, "A critical hit!", (Color){0xFF, 0xCC, 0x00, 0xFF});
            break;
        case EVT_SUPER_EFFECTIVE:
            log_add(client, "It's super effective!", (Color){0xFF, 0x88, 0x33, 0xFF});
            break;
        case EVT_NOT_EFFECTIVE:
            log_add(client, "It's not very effective...", (Color){0x88, 0xAA, 0x88, 0xFF});
            break;
        case EVT_DAMAGE: {
            int max_hp = 0;
            Player* p = &env->battle.players[e->data1];
            for (int j = 0; j < NUM_POKEMON; j++) {
                if (p->team[j].species == e->data3) {
                    max_hp = p->team[j].max_hp;
                    break;
                }
            }
            int pct = max_hp > 0 ? e->data2 * 100 / max_hp : 0;
            snprintf(buf, LOG_LINE_LEN, "%s %s -%d HP (%d%%)",
                     pn, species_name(e->data3), e->data2, pct);
            log_add(client, buf, (Color){0xCC, 0x33, 0x33, 0xFF});
            break;
        }
        case EVT_HEAL:
            snprintf(buf, LOG_LINE_LEN, "%s %s +%d HP",
                     pn, species_name(e->data3), e->data2);
            log_add(client, buf, (Color){0x33, 0xCC, 0x33, 0xFF});
            break;
        case EVT_FAINT:
            snprintf(buf, LOG_LINE_LEN, "%s %s fainted!",
                     pn, species_name(e->data3));
            log_add(client, buf, (Color){0xFF, 0x33, 0x33, 0xFF});
            break;
        case EVT_SWITCH:
            snprintf(buf, LOG_LINE_LEN, "%s sent out %s!", pn, species_name(e->data3));
            log_add(client, buf, (e->data1 == 1) ?
                    (Color){0xFF, 0xAA, 0x66, 0xFF} : (Color){0x66, 0xCC, 0xFF, 0xFF});
            break;
        case EVT_STATUS:
            snprintf(buf, LOG_LINE_LEN, "%s %s was %s!",
                     pn, species_name(e->data3), status_verb(e->data2));
            log_add(client, buf, (Color){0xCC, 0xCC, 0x33, 0xFF});
            break;
        case EVT_SLEEP:
            snprintf(buf, LOG_LINE_LEN, "%s %s is fast asleep. (asleep for %d turn%s)",
                     pn, species_name(e->data3), e->data2,
                     e->data2 == 1 ? "" : "s");
            log_add(client, buf, (Color){0x99, 0x99, 0xCC, 0xFF});
            break;
        case EVT_WAKE_UP:
            snprintf(buf, LOG_LINE_LEN, "%s %s woke up!",
                     pn, species_name(e->data3));
            log_add(client, buf, WHITE);
            break;
        case EVT_FROZEN:
            snprintf(buf, LOG_LINE_LEN, "%s %s is frozen solid!",
                     pn, species_name(e->data3));
            log_add(client, buf, (Color){0x66, 0xCC, 0xFF, 0xFF});
            break;
        case EVT_PARALYZED:
            snprintf(buf, LOG_LINE_LEN, "%s %s is fully paralyzed!",
                     pn, species_name(e->data3));
            log_add(client, buf, (Color){0xCC, 0xCC, 0x33, 0xFF});
            break;
        case EVT_CONFUSED_HIT:
            snprintf(buf, LOG_LINE_LEN, "%s %s hurt itself in confusion!",
                     pn, species_name(e->data3));
            log_add(client, buf, (Color){0xCC, 0x88, 0x33, 0xFF});
            break;
        case EVT_RECHARGING:
            snprintf(buf, LOG_LINE_LEN, "%s %s must recharge!",
                     pn, species_name(e->data3));
            log_add(client, buf, (Color){0x99, 0x99, 0x99, 0xFF});
            break;
        case EVT_SUBSTITUTE:
            snprintf(buf, LOG_LINE_LEN, "%s's substitute took %d damage!",
                     pn, e->data2);
            log_add(client, buf, (Color){0xCC, 0x99, 0x33, 0xFF});
            break;
        case EVT_SUB_BROKE:
            snprintf(buf, LOG_LINE_LEN, "%s's substitute broke!", pn);
            log_add(client, buf, (Color){0xCC, 0x66, 0x33, 0xFF});
            break;
        case EVT_STAT_CHANGE: {
            int stat_id = (e->data2 >> 4) & 0xF;
            int direction = e->data2 & 0xF;
            const char* stat_names[] = {"Attack", "Defense", "Special", "Speed", "accuracy", "evasion"};
            const char* stat_name = (stat_id < 6) ? stat_names[stat_id] : "stat";
            snprintf(buf, LOG_LINE_LEN, "%s %s's %s %s!",
                     pn, species_name(e->data3), stat_name,
                     direction ? "sharply rose" : "fell");
            log_add(client, buf, direction ?
                    (Color){0x33, 0xAA, 0xFF, 0xFF} : (Color){0xFF, 0x88, 0x33, 0xFF});
            break;
        }
        default:
            break;
        }
    }

    // Clear events after consuming
    env->event_count = 0;
    client->prev_turn = env->battle.turn;
}

// ============================================================================
// make_client / close_client
// ============================================================================

static Client* make_client(PokeBattle* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    if (!client) return NULL;

    InitWindow(WIN_W, WIN_H, "PokeBattle - Gen 1 OU");
    SetTargetFPS(60);

    // Load sprites
    client->sprites_loaded = 0;
    for (int s = 1; s <= NUM_SPECIES; s++) {
        char path[256];
        const char* name = RENDER_SPECIES_NAMES[s];
        if (!name) continue;

        snprintf(path, 256, "pufferlib/resources/poke_battle/sprites/gen1/%s.png", name);
        Image img = LoadImage(path);
        if (img.data) {
            client->front_sprites[s] = LoadTextureFromImage(img);
            UnloadImage(img);
            client->sprites_loaded++;
        }

        snprintf(path, 256, "pufferlib/resources/poke_battle/sprites/gen1-back/%s.png", name);
        img = LoadImage(path);
        if (img.data) {
            client->back_sprites[s] = LoadTextureFromImage(img);
            UnloadImage(img);
        }
    }

    // Init log
    client->log_head = 0;
    client->log_count = 0;
    client->prev_turn = -1;
    client->hover_action = -1;
    client->show_switch_panel = 0;
    client->show_result = 0;

    // Snapshot initial state
    Battle* b = &env->battle;
    for (int i = 0; i < NUM_POKEMON; i++) {
        client->prev_p1_hp[i] = b->players[0].team[i].hp;
        client->prev_p2_hp[i] = b->players[1].team[i].hp;
        client->prev_p1_status[i] = (int)b->players[0].team[i].status;
        client->prev_p2_status[i] = (int)b->players[1].team[i].status;
        client->prev_p1_alive[i] = b->players[0].team[i].is_alive;
        client->prev_p2_alive[i] = b->players[1].team[i].is_alive;
    }
    client->prev_p1_active = b->players[0].active_idx;
    client->prev_p2_active = b->players[1].active_idx;
    client->prev_turn = b->turn;

    log_add(client, "Battle started!", WHITE);

    return client;
}

static void close_client(Client* client) {
    if (!client) return;
    for (int s = 1; s <= NUM_SPECIES; s++) {
        if (client->front_sprites[s].id > 0)
            UnloadTexture(client->front_sprites[s]);
        if (client->back_sprites[s].id > 0)
            UnloadTexture(client->back_sprites[s]);
    }
    CloseWindow();
    free(client);
}

// ============================================================================
// Drawing Helpers
// ============================================================================

static Color hp_color(int hp, int max_hp) {
    if (max_hp <= 0) return CLR_HP_RED;
    float ratio = (float)hp / (float)max_hp;
    if (ratio > 0.5f) return CLR_HP_GREEN;
    if (ratio > 0.2f) return CLR_HP_YELLOW;
    return CLR_HP_RED;
}

static void draw_hp_bar(int x, int y, int width, int height, int hp, int max_hp, int show_numbers) {
    // Background
    DrawRectangle(x, y, width, height, CLR_HP_BG);

    // Fill
    if (max_hp > 0 && hp > 0) {
        int fill_w = width * hp / max_hp;
        if (fill_w < 1) fill_w = 1;
        DrawRectangle(x, y, fill_w, height, hp_color(hp, max_hp));
    }

    // Border
    DrawRectangleLines(x, y, width, height, DARKGRAY);

    // Optional numbers
    if (show_numbers) {
        char buf[32];
        snprintf(buf, 32, "%d/%d", hp, max_hp);
        DrawText(buf, x + width + 6, y - 1, 14, WHITE);
    }
}

static Color status_color(StatusCondition s) {
    switch (s) {
        case STATUS_BURN: return CLR_BRN;
        case STATUS_POISON: return CLR_PSN;
        case STATUS_TOXIC: return CLR_TOX;
        case STATUS_PARALYSIS: return CLR_PAR;
        case STATUS_SLEEP: return CLR_SLP;
        case STATUS_FREEZE: return CLR_FRZ;
        default: return BLANK;
    }
}

static const char* status_abbr(StatusCondition s) {
    switch (s) {
        case STATUS_BURN: return "BRN";
        case STATUS_POISON: return "PSN";
        case STATUS_TOXIC: return "TOX";
        case STATUS_PARALYSIS: return "PAR";
        case STATUS_SLEEP: return "SLP";
        case STATUS_FREEZE: return "FRZ";
        default: return NULL;
    }
}

static void draw_status_badge(int x, int y, StatusCondition status) {
    const char* abbr = status_abbr(status);
    if (!abbr) return;
    Color col = status_color(status);
    DrawRectangleRounded((Rectangle){(float)x, (float)y, 36, 16}, 0.4f, 4, col);
    DrawText(abbr, x + 4, y + 2, 12, WHITE);
}

static void draw_stat_stages(int x, int y, Player* p) {
    // Draw non-zero stat stages as compact colored labels
    const char* labels[] = {"Atk", "Def", "Spc", "Spe"};
    int stages[] = {p->atk_stage, p->def_stage, p->spc_stage, p->spe_stage};
    int cx = x;
    for (int i = 0; i < 4; i++) {
        if (stages[i] == 0) continue;
        char buf[12];
        snprintf(buf, 12, "%+d %s", stages[i], labels[i]);
        Color col = (stages[i] > 0)
            ? (Color){0x33, 0xBB, 0xFF, 0xFF}   // blue for boosts
            : (Color){0xFF, 0x66, 0x33, 0xFF};   // orange for drops
        DrawText(buf, cx, y, 11, col);
        cx += MeasureText(buf, 11) + 8;
    }
}

static void draw_team_icons(int x, int y, Player* p) {
    for (int i = 0; i < NUM_POKEMON; i++) {
        int cx = x + i * 18;
        Color col;
        if (p->team[i].species == SPECIES_NONE) {
            col = (Color){0x55, 0x55, 0x55, 0xFF};
        } else if (!p->team[i].is_alive) {
            col = (Color){0x88, 0x33, 0x33, 0xFF};
        } else if (i == p->active_idx) {
            col = (Color){0xFF, 0xCC, 0x00, 0xFF};
        } else {
            col = (Color){0x33, 0xCC, 0x33, 0xFF};
        }
        DrawCircle(cx + 7, y + 7, 7, col);
        DrawCircleLines(cx + 7, y + 7, 7, DARKGRAY);
    }
}

static Rectangle draw_move_button(int x, int y, int w, int h, Pokemon* poke, int move_idx,
                                   int is_hovered, int is_valid) {
    Rectangle rect = {(float)x, (float)y, (float)w, (float)h};
    MoveSlot* slot = &poke->moves[move_idx];
    const MoveData* mdata = &MOVE_DATA[slot->id];

    Color bg = TYPE_COLORS[mdata->type];
    if (!is_valid) {
        // Gray out
        bg.r = bg.r / 2;
        bg.g = bg.g / 2;
        bg.b = bg.b / 2;
    }

    DrawRectangleRounded(rect, 0.2f, 4, bg);
    if (is_hovered && is_valid) {
        DrawRectangleRoundedLinesEx(rect, 0.2f, 4, 3, WHITE);
    } else {
        DrawRectangleRoundedLinesEx(rect, 0.2f, 4, 1, (Color){0, 0, 0, 0x60});
    }

    // Move name
    DrawText(mdata->name, x + 8, y + 6, 16, WHITE);

    // Type label (bottom-left)
    DrawText(TYPE_NAMES[mdata->type], x + 8, y + h - 18, 12, (Color){0xFF, 0xFF, 0xFF, 0xCC});

    // PP (bottom-right)
    char pp_buf[20];
    snprintf(pp_buf, 20, "PP %d/%d", slot->pp, slot->max_pp);
    int pp_w = MeasureText(pp_buf, 12);
    DrawText(pp_buf, x + w - pp_w - 8, y + h - 18, 12, (Color){0xFF, 0xFF, 0xFF, 0xCC});

    return rect;
}

static Rectangle draw_switch_slot(int x, int y, int w, int h, Pokemon* poke, int slot_idx,
                                   int is_active, int is_hovered, int is_valid) {
    Rectangle rect = {(float)x, (float)y, (float)w, (float)h};

    Color bg;
    if (is_active) {
        bg = (Color){0x44, 0x66, 0x88, 0xFF};
    } else if (!poke->is_alive) {
        bg = (Color){0x55, 0x33, 0x33, 0xFF};
    } else if (!is_valid) {
        bg = (Color){0x44, 0x44, 0x44, 0xFF};
    } else {
        bg = (Color){0x3A, 0x4A, 0x5A, 0xFF};
    }

    DrawRectangleRounded(rect, 0.15f, 4, bg);
    if (is_hovered && is_valid && !is_active) {
        DrawRectangleRoundedLinesEx(rect, 0.15f, 4, 2, WHITE);
    }

    // Species name
    const char* name = SPECIES_DATA[poke->species].name;
    DrawText(name, x + 6, y + 4, 14, WHITE);

    // Mini HP bar
    int bar_x = x + 80;
    int bar_w = (w > 250) ? 80 : 50;
    draw_hp_bar(bar_x, y + 6, bar_w, 10, poke->hp, poke->max_hp, 0);

    // HP numbers
    char hp_buf[20];
    snprintf(hp_buf, 20, "%d/%d", poke->hp, poke->max_hp);
    DrawText(hp_buf, bar_x + bar_w + 6, y + 4, 12, (Color){0xCC, 0xCC, 0xCC, 0xFF});

    // Status badge
    if (poke->status != STATUS_NONE) {
        draw_status_badge(x + w - 50, y + 4, poke->status);
    }

    // Tags
    if (is_active) {
        DrawText("In", x + w - 18, y + 6, 12, (Color){0xFF, 0xCC, 0x00, 0xFF});
    } else if (!poke->is_alive) {
        DrawText("KO", x + w - 20, y + 6, 12, (Color){0xCC, 0x66, 0x66, 0xFF});
    }

    return rect;
}

static void draw_battle_log(Client* client, int x, int y, int w, int h) {
    // Background
    DrawRectangle(x, y, w, h, CLR_LOG_BG);
    DrawLine(x, y, x, y + h, DARKGRAY);

    // Header
    DrawRectangle(x, y, w, 28, (Color){0xDD, 0xD8, 0xCC, 0xFF});
    DrawText("Battle Log", x + 10, y + 6, 16, DARKGRAY);

    // Lines
    int line_y = y + 34;
    int max_visible = (h - 40) / 16;
    int start = 0;
    if (client->log_count > max_visible) {
        start = client->log_count - max_visible;
    }

    for (int i = start; i < client->log_count; i++) {
        int idx = (client->log_head + i) % LOG_MAX_LINES;
        Color col = client->log_colors[idx];
        // For log bg contrast, darken some colors
        if (col.r == 0xFF && col.g == 0xFF && col.b == 0xFF) {
            col = DARKGRAY; // White text on cream bg is hard to read
        }
        DrawText(client->log_lines[idx], x + 8, line_y, 13, col);
        line_y += 16;
        if (line_y > y + h - 8) break;
    }
}

// ============================================================================
// Battle Field Drawing
// ============================================================================

static void draw_battle_field(Client* client, PokeBattle* env) {
    Battle* b = &env->battle;
    Player* p1 = &b->players[0];
    Player* p2 = &b->players[1];
    Pokemon* p1_active = &p1->team[p1->active_idx];
    Pokemon* p2_active = &p2->team[p2->active_idx];

    // Green battle background
    DrawRectangleGradientV(0, 0, BATTLE_W, BATTLE_H / 2, CLR_FIELD_TOP, CLR_FIELD_BOT);
    DrawRectangleGradientV(0, BATTLE_H / 2, BATTLE_W, BATTLE_H / 2, CLR_FIELD_BOT, CLR_EARTH);

    // --- Opponent side (top-right area) ---

    // Shadow under opponent
    DrawEllipse(470, 195, 50, 12, CLR_SHADOW);

    // Opponent front sprite
    SpeciesID opp_species = p2_active->species;
    if (opp_species > 0 && opp_species <= NUM_SPECIES && client->front_sprites[opp_species].id > 0) {
        Texture2D tex = client->front_sprites[opp_species];
        float scale = 2.5f;
        DrawTextureEx(tex, (Vector2){470 - tex.width * scale / 2, 195 - tex.height * scale},
                      0, scale, WHITE);
    } else {
        // Fallback: draw a colored circle with name
        DrawCircle(470, 140, 40, (Color){0xCC, 0x44, 0x44, 0xFF});
        const char* name = SPECIES_DATA[opp_species].name;
        int tw = MeasureText(name, 14);
        DrawText(name, 470 - tw / 2, 133, 14, WHITE);
    }

    // Opponent stat bar (top-left) — fixed height to prevent team icon jitter
    {
        DrawRectangleRounded((Rectangle){20, 30, 240, 72}, 0.15f, 4, (Color){0x20, 0x30, 0x20, 0xDD});

        const char* name = SPECIES_DATA[opp_species].name;
        DrawText(name, 30, 36, 18, WHITE);
        int name_w = MeasureText(name, 18);
        DrawText("Lv100", 30 + name_w + 8, 40, 12, (Color){0xBB, 0xBB, 0xBB, 0xFF});

        // HP bar (no numbers for opponent, Showdown style)
        draw_hp_bar(30, 58, 180, 12, p2_active->hp, p2_active->max_hp, 0);

        // Status badge
        if (p2_active->status != STATUS_NONE) {
            draw_status_badge(215, 56, p2_active->status);
        }

        // Stat stage indicators
        if (p2->atk_stage || p2->def_stage || p2->spc_stage || p2->spe_stage) {
            draw_stat_stages(30, 74, p2);
        }
    }

    // Opponent team icons (fixed y)
    draw_team_icons(30, 106, p2);

    // --- Player side (bottom-left area) ---

    // Shadow under player
    DrawEllipse(190, 310, 50, 12, CLR_SHADOW);

    // Player back sprite
    SpeciesID own_species = p1_active->species;
    if (own_species > 0 && own_species <= NUM_SPECIES && client->back_sprites[own_species].id > 0) {
        Texture2D tex = client->back_sprites[own_species];
        float scale = 2.5f;
        DrawTextureEx(tex, (Vector2){190 - tex.width * scale / 2, 310 - tex.height * scale},
                      0, scale, WHITE);
    } else {
        // Fallback
        DrawCircle(190, 260, 40, (Color){0x44, 0x44, 0xCC, 0xFF});
        const char* name = SPECIES_DATA[own_species].name;
        int tw = MeasureText(name, 14);
        DrawText(name, 190 - tw / 2, 253, 14, WHITE);
    }

    // Player stat bar (bottom-right) — fixed height to prevent team icon overflow
    {
        DrawRectangleRounded((Rectangle){370, 260, 270, 80}, 0.15f, 4, (Color){0x20, 0x30, 0x20, 0xDD});

        const char* name = SPECIES_DATA[own_species].name;
        DrawText(name, 380, 266, 18, WHITE);
        int name_w = MeasureText(name, 18);
        DrawText("Lv100", 380 + name_w + 8, 270, 12, (Color){0xBB, 0xBB, 0xBB, 0xFF});

        // HP bar with numbers (Showdown shows numbers for your own)
        draw_hp_bar(380, 290, 160, 14, p1_active->hp, p1_active->max_hp, 1);

        // Status badge
        if (p1_active->status != STATUS_NONE) {
            draw_status_badge(380, 310, p1_active->status);
        }

        // Stat stage indicators
        if (p1->atk_stage || p1->def_stage || p1->spc_stage || p1->spe_stage) {
            draw_stat_stages(380, 324, p1);
        }
    }

    // Player team icons (fixed y, must end before BATTLE_H=360)
    draw_team_icons(380, 344, p1);

    // Player controller labels
    if (client->p2_label[0]) {
        int tw = MeasureText(client->p2_label, 14);
        DrawRectangleRounded((Rectangle){20, 10, (float)(tw + 16), 18},
                             0.3f, 4, (Color){0xCC, 0x33, 0x33, 0xCC});
        DrawText(client->p2_label, 28, 12, 14, WHITE);
    }
    if (client->p1_label[0]) {
        int tw = MeasureText(client->p1_label, 14);
        DrawRectangleRounded((Rectangle){(float)(640 - tw - 16), 244, (float)(tw + 16), 18},
                             0.3f, 4, (Color){0x33, 0x66, 0xCC, 0xCC});
        DrawText(client->p1_label, 640 - tw - 8, 246, 14, WHITE);
    }

    // Turn counter (top center)
    {
        char buf[32];
        snprintf(buf, 32, "Turn %d", b->turn);
        int tw = MeasureText(buf, 16);
        DrawRectangleRounded((Rectangle){(float)(BATTLE_W / 2 - tw / 2 - 10), 6, (float)(tw + 20), 24},
                             0.3f, 4, (Color){0, 0, 0, 0x88});
        DrawText(buf, BATTLE_W / 2 - tw / 2, 10, 16, WHITE);
    }

    // Divider line
    DrawLine(0, BATTLE_H, BATTLE_W, BATTLE_H, DARKGRAY);
}

// ============================================================================
// Control Panel Drawing & Mouse Interaction
// ============================================================================

static void draw_control_panel(Client* client, PokeBattle* env) {
    Battle* b = &env->battle;
    Player* p1 = &b->players[0];
    Pokemon* p1_active = &p1->team[p1->active_idx];

    // Panel background
    DrawRectangle(0, PANEL_Y, BATTLE_W, WIN_H - PANEL_Y, CLR_PANEL_BG);

    // Get action mask
    int mask[NUM_ACTIONS];
    get_action_mask(p1, b->mode, 0, mask);

    // Check if it's a forced switch situation
    int forced_switch = (b->mode == 1 || b->mode == 3);

    // Mouse position
    Vector2 mouse = GetMousePosition();
    client->hover_action = -1;

    // Result overlay in panel
    if (client->show_result) {
        DrawText(client->result_text, 20, PANEL_Y + 8, 18, (Color){0xFF, 0xCC, 0x00, 0xFF});
        DrawText("Click anywhere to continue...", 20, PANEL_Y + 32, 14, (Color){0xBB, 0xBB, 0xBB, 0xFF});
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            client->show_result = 0;
            env->mouse_action = -2; // Signal restart
        }
        return;
    }

    // Header text
    if (forced_switch) {
        DrawText("Choose a replacement Pokemon!", 20, PANEL_Y + 8, 16, (Color){0xFF, 0xCC, 0x00, 0xFF});
    } else {
        char header[64];
        snprintf(header, 64, "What will %s do?", SPECIES_DATA[p1_active->species].name);
        DrawText(header, 20, PANEL_Y + 8, 16, WHITE);
    }

    // --- Party panel (right side, always visible) ---
    {
        int slot_w = 220;
        int slot_h = 34;
        int party_x = 420;
        int party_y = PANEL_Y + 34;
        int gap_y = 4;

        for (int i = 0; i < NUM_POKEMON; i++) {
            int sy = party_y + i * (slot_h + gap_y);

            int is_active = (i == p1->active_idx);
            int is_valid = mask[4 + i];
            int hovered = CheckCollisionPointRec(mouse,
                (Rectangle){(float)party_x, (float)sy, (float)slot_w, (float)slot_h});

            if (hovered && is_valid && !is_active) {
                client->hover_action = 4 + i;
            }

            draw_switch_slot(party_x, sy, slot_w, slot_h, &p1->team[i], i,
                             is_active, hovered, is_valid);
        }
    }

    // --- Moves panel (left side, hidden during forced switch) ---
    if (!forced_switch) {
        int btn_w = 190;
        int btn_h = 55;
        int start_x = 20;
        int start_y = PANEL_Y + 34;
        int gap = 10;

        if (p1->is_recharging) {
            DrawText("Recharging... (must wait)", start_x, start_y + 20, 16,
                     (Color){0xCC, 0xCC, 0x33, 0xFF});
            client->hover_action = 0;
            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                env->mouse_action = 0;
            }
        } else {
            for (int i = 0; i < NUM_MOVE_SLOTS; i++) {
                int col = i % 2;
                int row = i / 2;
                int bx = start_x + col * (btn_w + gap);
                int by = start_y + row * (btn_h + gap);

                int hovered = CheckCollisionPointRec(mouse,
                    (Rectangle){(float)bx, (float)by, (float)btn_w, (float)btn_h});
                int valid = mask[i];

                if (hovered && valid) {
                    client->hover_action = i;
                }

                draw_move_button(bx, by, btn_w, btn_h, p1_active, i, hovered, valid);
            }

            // Struggle fallback
            int any_move = 0;
            for (int i = 0; i < NUM_MOVE_SLOTS; i++) {
                if (mask[i]) any_move = 1;
            }
            if (!any_move) {
                DrawRectangleRounded(
                    (Rectangle){(float)start_x, (float)start_y, (float)btn_w, (float)btn_h},
                    0.2f, 4, (Color){0x88, 0x44, 0x44, 0xFF});
                DrawText("Struggle", start_x + 8, start_y + 18, 16, WHITE);
                int hovered = CheckCollisionPointRec(mouse,
                    (Rectangle){(float)start_x, (float)start_y, (float)btn_w, (float)btn_h});
                if (hovered) {
                    client->hover_action = 0;
                    DrawRectangleRoundedLinesEx(
                        (Rectangle){(float)start_x, (float)start_y, (float)btn_w, (float)btn_h},
                        0.2f, 4, 3, WHITE);
                }
            }
        }
    }

    // Handle click
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && client->hover_action >= 0) {
        int action = client->hover_action;
        if (action >= 0 && action < NUM_ACTIONS && mask[action]) {
            env->mouse_action = action;
        }
    }
}

// ============================================================================
// Result Overlay
// ============================================================================

static void draw_result_overlay(Client* client) {
    if (!client->show_result) return;

    // Semi-transparent overlay on battle area
    DrawRectangle(0, 0, BATTLE_W, BATTLE_H, (Color){0, 0, 0, 0xAA});

    int tw = MeasureText(client->result_text, 36);
    DrawText(client->result_text, BATTLE_W / 2 - tw / 2, BATTLE_H / 2 - 18, 36, (Color){0xFF, 0xCC, 0x00, 0xFF});

    const char* sub = "Click to continue...";
    int sw = MeasureText(sub, 16);
    DrawText(sub, BATTLE_W / 2 - sw / 2, BATTLE_H / 2 + 30, 16, (Color){0xBB, 0xBB, 0xBB, 0xFF});
}

// Reset client state for a new game
static void reset_client_state(Client* client, PokeBattle* env) {
    if (!client) return;
    client->show_result = 0;
    client->show_switch_panel = 0;
    client->hover_action = -1;
    client->log_head = 0;
    client->log_count = 0;
    client->prev_turn = -1;

    Battle* b = &env->battle;
    for (int i = 0; i < NUM_POKEMON; i++) {
        client->prev_p1_hp[i] = b->players[0].team[i].hp;
        client->prev_p2_hp[i] = b->players[1].team[i].hp;
        client->prev_p1_status[i] = (int)b->players[0].team[i].status;
        client->prev_p2_status[i] = (int)b->players[1].team[i].status;
        client->prev_p1_alive[i] = b->players[0].team[i].is_alive;
        client->prev_p2_alive[i] = b->players[1].team[i].is_alive;
    }
    client->prev_p1_active = b->players[0].active_idx;
    client->prev_p2_active = b->players[1].active_idx;

    log_add(client, "Battle started!", WHITE);
}

// ============================================================================
// Main Render Function
// ============================================================================

void c_render(PokeBattle* env) {
    // Lazy init client
    if (!env->client) {
        env->client = make_client(env);
        if (!env->client) return;
    }

    Client* client = env->client;

    // Detect new game (reset happened) — turn went backward
    if (client->prev_turn > 0 && env->battle.turn == 0) {
        reset_client_state(client, env);
    }

    // Auto-detect game end and set result overlay
    if (env->terminals && env->terminals[0] && !client->show_result) {
        client->show_result = 1;
        if (env->last_result > 0) {
            snprintf(client->result_text, 64, "You won!");
        } else if (env->last_result < 0) {
            snprintf(client->result_text, 64, "You lost!");
        } else {
            snprintf(client->result_text, 64, "Draw!");
        }
        log_add(client, client->result_text, (Color){0xFF, 0xCC, 0x00, 0xFF});
    }

    // Update log from state diffs
    update_battle_log(client, env);

    BeginDrawing();
    ClearBackground(CLR_PANEL_BG);

    // Battle field
    draw_battle_field(client, env);

    // Result overlay
    draw_result_overlay(client);

    // Control panel
    draw_control_panel(client, env);

    // Log panel
    draw_battle_log(client, LOG_X, 0, LOG_W, WIN_H);

    EndDrawing();
}

#endif // POKE_BATTLE_RENDER_H
