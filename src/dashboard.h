#pragma once

#include <stdio.h>
#include <string.h>
#include <unistd.h>

static double puf_log_get_or(Dict* dict, const char* key, double fallback) {
    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0) {
            return dict->items[i].value;
        }
    }
    return fallback;
}

static int puf_dashboard_tty = 0;

#define PUF_DASH_WIDTH 80

static const char* puf_cyan(void) {
    return puf_dashboard_tty ? "\033[36m" : "";
}

static const char* puf_bcyan(void) {
    return puf_dashboard_tty ? "\033[96m" : "";
}

static const char* puf_white(void) {
    return puf_dashboard_tty ? "\033[37m" : "";
}

static const char* puf_bwhite(void) {
    return puf_dashboard_tty ? "\033[97m" : "";
}

static const char* puf_ansi_reset(void) {
    return puf_dashboard_tty ? "\033[0m" : "";
}

static void puf_dashboard_eol(void) {
    if (puf_dashboard_tty) {
        printf("\033[K");
    }
    putchar('\n');
}

static void puf_abbrev(char* out, size_t out_len, double val) {
    const char* suffix[] = {"", "K", "M", "B", "T"};
    int i = 0;
    while (val >= 1000.0 && i < 4) {
        val /= 1000.0;
        i++;
    }
    snprintf(out, out_len, "%.1f%s", val, suffix[i]);
}

static void puf_duration(char* out, size_t out_len, double seconds) {
    if (seconds < 0) {
        seconds = 0;
    }
    if (seconds < 1.0) {
        snprintf(out, out_len, "%.0fms", seconds * 1000.0);
        return;
    }

    long s = (long)seconds;
    snprintf(out, out_len, "%ldd %ldh %ldm %lds",
        s / 86400, (s / 3600) % 24, (s / 60) % 60, s % 60);
}

static void puf_perf_value(char* time_out, size_t time_len, char* pct_out, size_t pct_len,
        double part, double total) {
    int pct = total > 0 ? (int)(100.0 * part / total) : 0;
    puf_duration(time_out, time_len, part);
    snprintf(pct_out, pct_len, "%d%%", pct);
}

static void puf_strip_prefix(char* out, size_t out_len, const char* key, const char* prefix) {
    size_t n = strlen(prefix);
    if (strncmp(key, prefix, n) == 0) {
        snprintf(out, out_len, "%s", key + n);
    } else {
        snprintf(out, out_len, "%s", key);
    }
}

static int puf_loss_value(Dict* log, const char* key, char* out, size_t out_len) {
    for (int i = 0; i < log->size; i++) {
        if (strcmp(log->items[i].key, key) == 0) {
            snprintf(out, out_len, "%.3f", log->items[i].value);
            return 1;
        }
    }
    if (out_len > 0) {
        out[0] = 0;
    }
    return 0;
}

static void puf_panel_header(const char* eval_t, const char* eval_pct) {
    printf("%s│", puf_bcyan());
    printf("%s %-9s %13s%s    %s%-12s%s %s%6s %4s%s    %s%-10s %7s%s    ",
        puf_cyan(), "Summary", "Value", puf_ansi_reset(),
        puf_bcyan(), "Evaluate", puf_ansi_reset(), puf_bwhite(), eval_t, eval_pct, puf_ansi_reset(),
        puf_cyan(), "Losses", "Value", puf_ansi_reset());
    printf("%s│%s", puf_bcyan(), puf_ansi_reset());
    puf_dashboard_eol();
}

static void puf_panel_row(const char* s_name, const char* s_val,
        const char* p_name, const char* p_time, const char* p_pct,
        const char* l_name, const char* l_val, int emph_perf) {
    const char* perf_color = emph_perf ? puf_bcyan() : puf_bwhite();
    printf("%s│", puf_bcyan());
    printf("%s %s%-9s%s %s%13s%s    %s%-12s%s %s%6s %4s%s    %s%-10s %7s%s    ",
        puf_ansi_reset(),
        puf_white(), s_name, puf_ansi_reset(), puf_bwhite(), s_val, puf_ansi_reset(),
        perf_color, p_name, puf_ansi_reset(), puf_bwhite(), p_time, p_pct, puf_ansi_reset(),
        puf_bwhite(), l_name, l_val, puf_ansi_reset());
    printf("%s│%s", puf_bcyan(), puf_ansi_reset());
    puf_dashboard_eol();
}

static void puf_user_header(void) {
    printf("%s│", puf_bcyan());
    printf("%s %-23s %9s%s   %s%-23s %9s%s        ",
        puf_cyan(), "User Stats", "Value", puf_ansi_reset(),
        puf_cyan(), "User Stats", "Value", puf_ansi_reset());
    printf("%s│%s", puf_bcyan(), puf_ansi_reset());
    puf_dashboard_eol();
}

static void puf_user_row(const char* left_key, double left_val,
        const char* right_key, double right_val, int has_right) {
    printf("%s│", puf_bcyan());
    if (has_right) {
        printf("%s %s%-23s %9.3f%s   %s%-23s %9.3f%s        ",
            puf_ansi_reset(),
            puf_bwhite(), left_key, left_val, puf_ansi_reset(),
            puf_bwhite(), right_key, right_val, puf_ansi_reset());
    } else {
        printf("%s %s%-23s %9.3f%s   %-23s %9s        ",
            puf_ansi_reset(),
            puf_bwhite(), left_key, left_val, puf_ansi_reset(), "", "");
    }
    printf("%s│%s", puf_bcyan(), puf_ansi_reset());
    puf_dashboard_eol();
}

static void puf_dashboard_blank(void) {
    printf("%s│%*s│%s", puf_bcyan(), PUF_DASH_WIDTH - 2, "", puf_ansi_reset());
    puf_dashboard_eol();
}

static void puf_dashboard_rule(const char* left, const char* right) {
    printf("%s%s", puf_bcyan(), left);
    for (int i = 0; i < PUF_DASH_WIDTH - 2; i++) {
        printf("─");
    }
    printf("%s%s", right, puf_ansi_reset());
    puf_dashboard_eol();
}

static void puf_dashboard_print(Config* cfg, PuffeRL* p, Dict* log, int epoch) {
    puf_dashboard_tty = isatty(STDOUT_FILENO);
    if (puf_dashboard_tty) {
        printf("\033[?2026h\033[H");
    }

    const char* env_name = puf_config_str(cfg, "base", "env_name");
    double steps = puf_log_get_or(log, "agent_steps", (double)p->global_step);
    double sps = puf_log_get_or(log, "SPS", 0);
    double target_steps = puf_config_get(cfg, "train", "total_timesteps");
    double remaining_sec = sps > 0 ? (target_steps - steps) / sps : 0;
    double rollout = puf_log_get_or(log, "perf/rollout", 0);
    double train_time = puf_log_get_or(log, "perf/train", 0);
    double perf_total = rollout + train_time;

    char params[32];
    char steps_s[32];
    char sps_s[32];
    char uptime[64];
    char remaining[64];
    puf_abbrev(params, sizeof(params), (double)numel(p->master_weights.shape));
    puf_abbrev(steps_s, sizeof(steps_s), steps);
    puf_abbrev(sps_s, sizeof(sps_s), sps);
    puf_duration(uptime, sizeof(uptime), puf_log_get_or(log, "uptime", 0));
    puf_duration(remaining, sizeof(remaining), remaining_sec);

    puf_dashboard_rule("╭", "╮");
    printf("%s│", puf_bcyan());
    printf("%s %sPufferLib %s4.0%s        %s🐡%s        %sGPU:%s %2.0f%%    %sVRAM:%s %.1f/%.0fG    %sRAM:%s %.1fG        ",
        puf_ansi_reset(),
        puf_bcyan(), puf_bwhite(), puf_ansi_reset(),
        puf_bcyan(), puf_ansi_reset(),
        puf_cyan(), puf_bwhite(),
        puf_log_get_or(log, "util/gpu_percent", 0),
        puf_cyan(), puf_bwhite(),
        puf_log_get_or(log, "util/vram_used_gb", 0),
        puf_log_get_or(log, "util/vram_total_gb", 0),
        puf_cyan(), puf_bwhite(),
        puf_log_get_or(log, "util/cpu_mem_gb", 0));
    printf("%s│%s", puf_bcyan(), puf_ansi_reset());
    puf_dashboard_eol();
    puf_dashboard_blank();

    char epoch_s[32];
    char eval_t[64];
    char eval_pct[16];
    char gpu_t[64];
    char gpu_pct[16];
    char env_t[64];
    char env_pct[16];
    char train_t[64];
    char train_pct[16];
    char misc_t[64];
    char misc_pct[16];
    char forward_t[64];
    char forward_pct[16];
    char loss_policy[32];
    char loss_value[32];
    char loss_entropy[32];
    char loss_total[32];
    char loss_old_kl[32];
    char loss_kl[32];
    char loss_clipfrac[32];
    snprintf(epoch_s, sizeof(epoch_s), "%d", epoch);
    puf_perf_value(eval_t, sizeof(eval_t), eval_pct, sizeof(eval_pct), rollout, perf_total);
    puf_perf_value(gpu_t, sizeof(gpu_t), gpu_pct, sizeof(gpu_pct),
        puf_log_get_or(log, "perf/eval_gpu", 0), perf_total);
    puf_perf_value(env_t, sizeof(env_t), env_pct, sizeof(env_pct),
        puf_log_get_or(log, "perf/eval_env", 0), perf_total);
    puf_perf_value(train_t, sizeof(train_t), train_pct, sizeof(train_pct), train_time, perf_total);
    puf_perf_value(misc_t, sizeof(misc_t), misc_pct, sizeof(misc_pct),
        puf_log_get_or(log, "perf/train_misc", 0), perf_total);
    puf_perf_value(forward_t, sizeof(forward_t), forward_pct, sizeof(forward_pct),
        puf_log_get_or(log, "perf/train_forward", 0), perf_total);
    puf_loss_value(log, "loss/policy", loss_policy, sizeof(loss_policy));
    puf_loss_value(log, "loss/value", loss_value, sizeof(loss_value));
    puf_loss_value(log, "loss/entropy", loss_entropy, sizeof(loss_entropy));
    puf_loss_value(log, "loss/total", loss_total, sizeof(loss_total));
    puf_loss_value(log, "loss/old_kl", loss_old_kl, sizeof(loss_old_kl));
    puf_loss_value(log, "loss/kl", loss_kl, sizeof(loss_kl));
    puf_loss_value(log, "loss/clipfrac", loss_clipfrac, sizeof(loss_clipfrac));

    puf_panel_header(eval_t, eval_pct);
    puf_panel_row("Env", env_name, "  GPU", gpu_t, gpu_pct, "policy", loss_policy, 0);
    puf_panel_row("Params", params, "  Env", env_t, env_pct, "value", loss_value, 0);
    puf_panel_row("Steps", steps_s, "Train", train_t, train_pct, "entropy", loss_entropy, 1);
    puf_panel_row("SPS", sps_s, "  Misc", misc_t, misc_pct, "total", loss_total, 0);
    puf_panel_row("Epoch", epoch_s, "  Forward", forward_t, forward_pct, "old_kl", loss_old_kl, 0);
    puf_panel_row("Uptime", uptime, "", "", "", "kl", loss_kl, 0);
    puf_panel_row("To go", remaining, "", "", "", "clipfrac", loss_clipfrac, 0);
    puf_dashboard_blank();

    puf_user_header();
    char pending_key[128];
    double pending_val = 0;
    int pending = 0;
    int n = 0;
    for (int i = 0; i < log->size && n < 30; i++) {
        const char* key = log->items[i].key;
        if (strncmp(key, "env/", 4) != 0 || strcmp(key, "env/n") == 0) {
            continue;
        }

        char short_key[128];
        puf_strip_prefix(short_key, sizeof(short_key), key, "env/");
        if (!pending) {
            snprintf(pending_key, sizeof(pending_key), "%s", short_key);
            pending_val = log->items[i].value;
            pending = 1;
        } else {
            puf_user_row(pending_key, pending_val, short_key, log->items[i].value, 1);
            pending = 0;
        }
        n++;
    }
    if (pending) {
        puf_user_row(pending_key, pending_val, "", 0, 0);
    }
    puf_dashboard_rule("╰", "╯");
    if (puf_dashboard_tty) {
        printf("\033[J\033[?2026l");
    }
    fflush(stdout);
}
