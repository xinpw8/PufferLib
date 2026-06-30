#pragma once

#include <filesystem>
#include <string>

#include <pybind11/pybind11.h>

#include "config.h"

namespace puf_cfg {
namespace py = pybind11;
namespace fs = std::filesystem;

static py::object puf_parse_py_value(const char* raw) {
    if (puf_config_streq_ci(raw, "none")) {
        return py::none();
    }
    if (puf_config_streq_ci(raw, "true")) {
        return py::bool_(true);
    }
    if (puf_config_streq_ci(raw, "false")) {
        return py::bool_(false);
    }

    char buf[256];
    size_t j = 0;
    int has_float = 0;
    for (size_t i = 0; raw[i] && j + 1 < sizeof(buf); i++) {
        if (raw[i] == '_' || isspace((unsigned char)raw[i])) {
            continue;
        }
        if (raw[i] == '.' || raw[i] == 'e' || raw[i] == 'E') {
            has_float = 1;
        }
        buf[j++] = raw[i];
    }
    buf[j] = 0;

    if (buf[0]) {
        char* end = 0;
        if (has_float) {
            double v = strtod(buf, &end);
            if (end && !*end) {
                return py::float_(v);
            }
        } else {
            long long v = strtoll(buf, &end, 10);
            if (end && !*end) {
                return py::int_(v);
            }
        }
    }

    return py::str(raw);
}

static py::dict puf_config_to_pydict(PufConfigFile* f) {
    py::dict out;
    for (int i = 0; i < f->len; i++) {
        PufConfig* cfg = &f->sections[i];
        py::dict section;
        for (int j = 0; j < cfg->len; j++) {
            section[py::str(cfg->items[j].key)] = puf_parse_py_value(cfg->items[j].val);
        }

        if (strcmp(cfg->name, "base") == 0) {
            for (auto item : section) {
                out[item.first] = item.second;
            }
        } else {
            py::dict cur = out;
            std::string name = cfg->name;
            size_t start = 0;
            for (;;) {
                size_t dot = name.find('.', start);
                std::string part = name.substr(start, dot - start);
                if (dot == std::string::npos) {
                    cur[py::str(part)] = section;
                    break;
                }

                py::str key(part);
                if (!cur.contains(key)) {
                    cur[key] = py::dict();
                }
                cur = cur[key].cast<py::dict>();
                start = dot + 1;
            }
        }
    }
    return out;
}

static int puf_apply_cli_arg(PufConfigFile* f, const char* arg, const char* value, int idx) {
    char* tmp = puf_config_strdup(arg);
    char* s = tmp;
    while (*s == '-') {
        s++;
    }

    char* eq = strchr(s, '=');
    if (eq) {
        *eq = 0;
        value = eq + 1;
    }

    char* dot = strchr(s, '.');
    const char* section = "base";
    char* key = s;
    if (dot) {
        *dot = 0;
        section = s;
        key = dot + 1;
    }

    for (char* p = key; *p; p++) {
        if (*p == '-') {
            *p = '_';
        }
    }

    if (!*key) {
        fprintf(stderr, "argv:%d: empty key\n", idx);
        free(tmp);
        return 0;
    }

    puf_config_put(puf_config_get_section(f, section), key, value ? value : "true");
    free(tmp);
    return 1;
}

static int puf_apply_cli(PufConfigFile* f, py::list argv) {
    for (py::ssize_t i = 0; i < py::len(argv); i++) {
        std::string arg = argv[i].cast<std::string>();
        if (arg.size() == 0) {
            continue;
        }

        if (arg[0] != '-' && arg.find('=') == std::string::npos) {
            continue;
        }

        const char* value = 0;
        std::string value_storage;
        if (arg.find('=') == std::string::npos) {
            if (i + 1 < py::len(argv)) {
                std::string next = argv[i + 1].cast<std::string>();
                if (next.size() && next[0] != '-') {
                    value_storage = next;
                    value = value_storage.c_str();
                    i++;
                }
            }
        }

        if (!puf_apply_cli_arg(f, arg.c_str(), value, (int)i)) {
            return 0;
        }
    }
    return 1;
}

static bool puf_config_matches_env(const fs::path& path, const char* env_name) {
    PufConfigFile tmp = {0};
    bool ok = puf_config_load_file(&tmp, path.string().c_str(), false);
    PufConfig* base = puf_config_section(&tmp, "base");
    const char* names = base ? puf_config_get(base, "env_name") : 0;
    bool match = false;
    if (ok && names) {
        char* copy = puf_config_strdup(names);
        for (char* tok = strtok(copy, " \t\r\n"); tok; tok = strtok(0, " \t\r\n")) {
            if (strcmp(tok, env_name) == 0) {
                match = true;
                break;
            }
        }
        free(copy);
    }
    puf_config_free(&tmp);
    return match;
}

static void puf_add_compat_defaults(PufConfigFile* f) {
    PufConfig* base = puf_config_get_section(f, "base");
    struct DefaultKV {
        const char* key;
        const char* val;
    };
    DefaultKV defaults[] = {
        {"load_model_path", "None"},
        {"load_enemy_model_path", "None"},
        {"num_games", "4096"},
        {"enemy_hidden_size", "None"},
        {"enemy_num_layers", "None"},
        {"load_id", "None"},
        {"render_mode", "auto"},
        {"wandb", "false"},
        {"wandb_project", "puffer4"},
        {"wandb_group", "debug"},
        {"tag", "None"},
        {"slowly", "false"},
        {"save_frames", "0"},
        {"gif_path", "eval.gif"},
        {"fps", "15"},
    };

    for (size_t i = 0; i < sizeof(defaults) / sizeof(defaults[0]); i++) {
        if (!puf_config_get(base, defaults[i].key)) {
            puf_config_put(base, defaults[i].key, defaults[i].val);
        }
    }
}

static py::dict py_load_config_native(const std::string& env_name, py::list argv,
        const std::string& root_path) {
    PufConfigFile cfg = {0};
    const fs::path root = root_path.empty() ? fs::current_path() : fs::path(root_path);
    const fs::path config_dir = root / "config";
    const fs::path default_path = config_dir / "default.ini";

    if (!puf_config_load_file(&cfg, default_path.string().c_str(), true)) {
        puf_config_free(&cfg);
        throw std::runtime_error("failed to load config/default.ini");
    }

    if (env_name != "default") {
        bool found = false;
        for (const auto& entry : fs::recursive_directory_iterator(config_dir)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".ini") {
                continue;
            }
            if (entry.path().filename() == "default.ini") {
                continue;
            }
            if (!puf_config_matches_env(entry.path(), env_name.c_str())) {
                continue;
            }
            if (!puf_config_load_file(&cfg, entry.path().string().c_str(), true)) {
                puf_config_free(&cfg);
                throw std::runtime_error("failed to load env config");
            }
            found = true;
            break;
        }
        if (!found) {
            puf_config_free(&cfg);
            throw std::runtime_error("No config for env_name " + env_name);
        }
    }

    puf_config_put(puf_config_get_section(&cfg, "base"), "env_name", env_name.c_str());
    puf_add_compat_defaults(&cfg);
    if (!puf_apply_cli(&cfg, argv)) {
        puf_config_free(&cfg);
        throw std::runtime_error("failed to parse CLI overrides");
    }

    py::dict out = puf_config_to_pydict(&cfg);
    puf_config_free(&cfg);
    return out;
}

}  // namespace puf_cfg
