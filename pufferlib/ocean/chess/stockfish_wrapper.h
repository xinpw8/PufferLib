#pragma once

#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <stdexcept>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/select.h>
#include <chrono>
#include <cerrno>

// ────────────────────────────────────────────────────────────────
// Debug‑print macro: compile with ‑DSTOCKFISH_VERBOSE to enable.
// ────────────────────────────────────────────────────────────────
#ifndef STOCKFISH_VERBOSE
#   define SF_LOG(...)        /* silence */
#else
#   define SF_LOG(...)        fprintf(stderr, __VA_ARGS__)
#endif

class Stockfish {
    //----------------------------------------------------------------
    // Helpers and state
    //----------------------------------------------------------------
    void send(const std::string &line) {
        fputs(line.c_str(), in_fp);
        fflush(in_fp);
        SF_LOG("[Stockfish ⇐] %s", line.c_str());
    }

    FILE *in_fp  = nullptr;   // write commands
    FILE *out_fp = nullptr;   // read responses
    pid_t pid    = -1;        // child PID
    char  buf[256]{};
    int   search_time_ms = 10;
    std::string cmd_str{"stockfish"};
    int   elo_level = 1320;

    static FILE *fdopen_checked(int fd, const char *mode) {
        FILE *f = fdopen(fd, mode);
        if (!f) close(fd);
        return f;
    }

    bool start_engine() {
        const char *cmd = cmd_str.c_str();
        SF_LOG("[Stockfish] Attempting to start engine with command: %s, ELO: %d, search_ms: %d\n",
               cmd, elo_level, search_time_ms);

        int in_pipe[2];   // parent writes to child
        int out_pipe[2];  // parent reads from child
        if (pipe(in_pipe) == -1 || pipe(out_pipe) == -1) {
            perror("pipe");
            return false;
        }

        pid = fork();
        if (pid == -1) {
            perror("fork");
            return false;
        }

        if (pid == 0) {
            // Child
            dup2(in_pipe[0], STDIN_FILENO);
            dup2(out_pipe[1], STDOUT_FILENO);
            dup2(out_pipe[1], STDERR_FILENO);
            close(in_pipe[0]); close(in_pipe[1]);
            close(out_pipe[0]); close(out_pipe[1]);
            execlp(cmd, cmd, (char *)nullptr);
            perror("execlp");
            _exit(1);
        }

        // Parent
        close(in_pipe[0]);
        close(out_pipe[1]);

        in_fp  = fdopen_checked(in_pipe[1], "w");
        out_fp = fdopen_checked(out_pipe[0], "r");
        if (!in_fp || !out_fp) {
            fprintf(stderr, "[Stockfish] pipe setup failed\n");
            return false;
        }
        setvbuf(in_fp,  NULL, _IOLBF, 0);
        setvbuf(out_fp, NULL, _IOLBF, 0);

        SF_LOG("[Stockfish] Pipes established, sending UCI handshake…\n");
        send("uci\n");
        while (fgets(buf, sizeof(buf), out_fp)) if (!strncmp(buf, "uciok", 5)) break;

        send("setoption name EvalFile value nn-1c0000000000.nnue\n");
        send("isready\n");
        while (fgets(buf, sizeof(buf), out_fp)) if (!strncmp(buf, "readyok", 7)) break;
        SF_LOG("[Stockfish] UCI handshake complete\n");

        send("setoption name UCI_LimitStrength value true\n");
        send(("setoption name UCI_Elo value " + std::to_string(elo_level) + "\n").c_str());
        send("setoption name Skill Level value 0\n");
        send("setoption name Threads value 1\n");
        send("setoption name Hash value 16\n");
        send("isready\n");
        while (fgets(buf, sizeof(buf), out_fp)) if (!strncmp(buf, "readyok", 7)) break;
        SF_LOG("[Stockfish] Engine ready (ELO=%d)\n", elo_level);
        return true;
    }

    //----------------------------------------------------------------
    // No‑copy / move‑enabled semantics
    //----------------------------------------------------------------
    Stockfish(const Stockfish&)            = delete;
    Stockfish& operator=(const Stockfish&) = delete;

    // Move constructor: transfer ownership of pipes/PID
    Stockfish(Stockfish&& other) noexcept {
        in_fp  = other.in_fp;   other.in_fp  = nullptr;
        out_fp = other.out_fp;  other.out_fp = nullptr;
        pid    = other.pid;     other.pid    = -1;
        search_time_ms = other.search_time_ms;
        cmd_str        = std::move(other.cmd_str);
        elo_level      = other.elo_level;
    }
    Stockfish& operator=(Stockfish&&) = delete;

public:
    //----------------------------------------------------------------
    // Construction & destruction
    //----------------------------------------------------------------
    explicit Stockfish(const char *cmd = "stockfish", int elo = 1320, int search_ms = 10)
        : search_time_ms(search_ms), cmd_str(cmd), elo_level(elo) {
        start_engine();
    }

    ~Stockfish() {
        if (!ok()) return;
        fputs("quit\n", in_fp);
        fflush(in_fp);
        fclose(in_fp);
        fclose(out_fp);
        waitpid(pid, nullptr, 0);
    }

    //----------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------
    std::string bestmove(const std::string &fen, int ms = 50) {
        if (!ok()) return "0000";
        send("stop\n");
        send(("position fen " + fen + "\n").c_str());
        send(("go movetime " + std::to_string(ms) + "\n").c_str());
        while (fgets(buf, sizeof(buf), out_fp))
            if (!strncmp(buf, "bestmove", 8))
                return std::string(buf + 9, strcspn(buf + 9, " \n"));
        return "0000";
    }

    std::pair<std::string,int> bestmove_with_score(const std::string &fen, int ms = -1) {
        if (ms == -1) ms = search_time_ms;
        int last_cp = 0;
        if (!ok()) { restart_engine(); if (!ok()) return {"0000",0}; }
        send(("position fen " + fen + "\n").c_str());
        send(("go movetime " + std::to_string(ms) + "\n").c_str());
        while (fgets(buf, sizeof(buf), out_fp)) {
            if (!strncmp(buf, "info", 4)) {
                char *p = strstr(buf, "score");
                if (p) {
                    int cp;
                    if (sscanf(p, "score cp %d", &cp) == 1) last_cp = cp;
                    else if (sscanf(p, "score mate %d", &cp) == 1)
                        last_cp = cp > 0 ? 32000 - cp*100 : -32000 - cp*100;
                }
            } else if (!strncmp(buf, "bestmove", 8)) {
                std::string mv(buf + 9, strcspn(buf + 9, " \n"));
                return {mv, last_cp};
            }
        }
        restart_engine();
        return {"0000", last_cp};
    }

    bool ok() const {
        return in_fp && out_fp && pid > 0 && (kill(pid,0) == 0 || errno != ESRCH);
    }

    void restart_engine() {
        SF_LOG("[Stockfish] Restarting crashed engine (pid=%d)\n", pid);
        if (in_fp) fclose(in_fp);
        if (out_fp) fclose(out_fp);
        if (pid > 0) { kill(pid, SIGTERM); waitpid(pid,nullptr,0); }
        in_fp = out_fp = nullptr; pid = -1;
        start_engine();
    }
};
