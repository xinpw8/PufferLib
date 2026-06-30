#pragma once

static float py_protein_scale(const std::string& dist, float min_v, float max_v,
        const py::object& scale_obj) {
    if (py::isinstance<py::str>(scale_obj)) {
        std::string s = scale_obj.cast<std::string>();
        if (s == "time") {
            return 1.0f / (log2f(max_v) - log2f(min_v));
        }
        return 0.5f;
    }
    return scale_obj.cast<float>();
}

struct PyProteinKey {
    std::vector<std::string> path;
};

class PyProtein {
    ProteinSweep* sw_;
    std::vector<PyProteinKey> keys_;
    std::vector<Space> spaces_;
    int cost_idx_;

    PyProtein() : sw_(nullptr), cost_idx_(-1) {
    }

    void check() const {
        if (!sw_ || !sw_->hypers) {
            throw std::runtime_error("Protein is not initialized");
        }
    }

    static py::object dict_get(py::dict d, const PyProteinKey& key) {
        py::object cur = d;
        for (size_t i = 0; i < key.path.size(); i++) {
            cur = cur.attr("__getitem__")(py::str(key.path[i]));
        }
        return cur;
    }

    static void dict_set(py::dict d, const PyProteinKey& key, float val) {
        py::object cur = d;
        for (size_t i = 0; i + 1 < key.path.size(); i++) {
            cur = cur.attr("__getitem__")(py::str(key.path[i]));
        }
        cur.attr("__setitem__")(py::str(key.path.back()), py::float_(val));
    }

    void build_spaces(py::dict sweep_config, const std::set<std::string>& skip,
            const std::set<std::string>* only, std::vector<std::string>& prefix) {
        for (auto item : sweep_config) {
            std::string name = item.first.cast<std::string>();
            if (skip.count(name)) {
                continue;
            }

            py::object val = item.second.cast<py::object>();
            if (!py::isinstance<py::dict>(val)) {
                continue;
            }

            py::dict section = val.cast<py::dict>();
            bool has_subsection = false;
            for (auto sub : section) {
                if (py::isinstance<py::dict>(sub.second)) {
                    has_subsection = true;
                    break;
                }
            }

            if (has_subsection) {
                prefix.push_back(name);
                build_spaces(section, skip, only, prefix);
                prefix.pop_back();
                continue;
            }

            if (only && !only->empty()) {
                bool found = false;
                for (auto& needle : *only) {
                    if (name.find(needle) != std::string::npos) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    continue;
                }
            }

            std::string dist = section["distribution"].cast<std::string>();
            float min_v = section["min"].cast<float>();
            float max_v = section["max"].cast<float>();
            float scale = py_protein_scale(dist, min_v, max_v, section["scale"]);

            int is_integer = 0;
            SpaceType type;
            if (dist == "uniform") {
                type = SPACE_LINEAR;
            } else if (dist == "int_uniform") {
                type = SPACE_LINEAR;
                is_integer = 1;
            } else if (dist == "uniform_pow2") {
                type = SPACE_POW2;
                is_integer = 1;
            } else if (dist == "log_normal") {
                type = SPACE_LOG;
            } else if (dist == "logit_normal") {
                type = SPACE_LOGIT;
            } else {
                throw std::runtime_error("Unknown distribution: " + dist);
            }

            Space space;
            space_init(&space, type, min_v, max_v, scale, is_integer);
            spaces_.push_back(space);

            PyProteinKey key;
            key.path = prefix;
            key.path.push_back(name);
            keys_.push_back(key);
        }
    }

public:
    PyProtein(py::dict sweep_config) : sw_(nullptr), cost_idx_(-1) {
        static const std::set<std::string> skip = {
            "method", "metric", "metric_distribution", "goal",
            "downsample", "use_gpu", "prune_pareto", "sweep_only",
            "max_suggestion_cost", "early_stop_quantile", "gpus",
            "max_runs", "match_enemy_model_path", "match_num_games",
            "match_enemy_hidden_size", "match_enemy_num_layers"};

        std::set<std::string> only_set;
        const std::set<std::string>* only = nullptr;
        if (sweep_config.contains("sweep_only")) {
            std::string raw = sweep_config["sweep_only"].cast<std::string>();
            std::istringstream stream(raw);
            std::string tok;
            while (std::getline(stream, tok, ',')) {
                size_t a = tok.find_first_not_of(' ');
                size_t b = tok.find_last_not_of(' ');
                if (a != std::string::npos) {
                    only_set.insert(tok.substr(a, b - a + 1));
                }
            }
            only = &only_set;
        }

        std::vector<std::string> prefix;
        build_spaces(sweep_config, skip, only, prefix);

        int dim = (int)spaces_.size();
        if (dim == 0) {
            throw std::runtime_error("No sweep parameters found");
        }

        int optimize_direction = 1;
        if (sweep_config.contains("goal")
                && sweep_config["goal"].cast<std::string>() == "minimize") {
            optimize_direction = -1;
        }

        for (int i = 0; i < dim; i++) {
            std::string flat;
            for (size_t j = 0; j < keys_[i].path.size(); j++) {
                if (j) {
                    flat += "/";
                }
                flat += keys_[i].path[j];
            }
            if (flat == "train/total_timesteps") {
                cost_idx_ = i;
                break;
            }
        }

        Hyperparameters* hypers = hyperparameters_create(
            spaces_.data(), dim, cost_idx_, optimize_direction);

        bool prune_pareto = sweep_config.contains("prune_pareto")
            ? sweep_config["prune_pareto"].cast<bool>() : true;
        float max_cost = sweep_config.contains("max_suggestion_cost")
            ? sweep_config["max_suggestion_cost"].cast<float>() : 3600.0f;
        float early_stop_quantile = sweep_config.contains("early_stop_quantile")
            ? sweep_config["early_stop_quantile"].cast<float>() : 0.3f;
        int downsample = sweep_config.contains("downsample")
            ? sweep_config["downsample"].cast<int>() : 5;
        int max_runs = sweep_config.contains("max_runs")
            ? sweep_config["max_runs"].cast<int>() : 1200;
        std::string metric_distribution = sweep_config.contains("metric_distribution")
            ? sweep_config["metric_distribution"].cast<std::string>() : "linear";
        int use_logit = metric_distribution == "percentile" || metric_distribution == "logit";

        int success_cap = max_runs * downsample * 2;
        if (success_cap < 8192) {
            success_cap = 8192;
        }

        sw_ = protein_sweep_create(hypers,
            10, 256, 50, 0.001f, 50, 750, 4096,
            downsample == 1, prune_pareto, use_logit,
            1.0f, max_cost, 0.1f, -0.8f, early_stop_quantile,
            success_cap, 1024, 5, 73ULL);
    }

    ~PyProtein() {
        if (sw_) {
            protein_sweep_destroy(sw_);
        }
    }

    PyProtein(const PyProtein&) = delete;
    PyProtein& operator=(const PyProtein&) = delete;

    PyProtein(PyProtein&& other) noexcept
        : sw_(other.sw_), keys_(std::move(other.keys_)),
          spaces_(std::move(other.spaces_)), cost_idx_(other.cost_idx_) {
        other.sw_ = nullptr;
    }

    py::tuple suggest(py::dict fill, py::object fixed_total_timesteps) {
        check();
        int dim = sw_->hypers->num;
        float fixed_cost_norm = NAN;
        if (!fixed_total_timesteps.is_none() && cost_idx_ >= 0) {
            fixed_cost_norm = space_normalize(&spaces_[(size_t)cost_idx_],
                fixed_total_timesteps.cast<float>());
        }

        std::vector<float> out((size_t)dim);
        ProteinSweepInfo info = protein_sweep_suggest(sw_, out.data(), fixed_cost_norm);
        for (int i = 0; i < dim; i++) {
            float val = space_unnormalize(&spaces_[(size_t)i], out[(size_t)i]);
            dict_set(fill, keys_[(size_t)i], val);
        }

        py::dict info_dict;
        info_dict["is_random"] = (bool)info.is_random;
        info_dict["score"] = info.predicted_score;
        info_dict["cost"] = info.predicted_cost;
        info_dict["rating"] = info.rating;
        info_dict["score_loss"] = info.score_loss;
        info_dict["cost_loss"] = info.cost_loss;
        info_dict["n_gp_obs"] = info.n_gp_obs;
        info_dict["n_pareto"] = info.n_pareto;
        info_dict["n_candidates"] = info.n_candidates;
        return py::make_tuple(fill, info_dict);
    }

    void observe(py::dict hypers, float score, float cost, bool is_failure) {
        check();
        int dim = sw_->hypers->num;
        std::vector<float> norm((size_t)dim);
        for (int i = 0; i < dim; i++) {
            float val = dict_get(hypers, keys_[(size_t)i]).cast<float>();
            norm[(size_t)i] = space_normalize(&spaces_[(size_t)i], val);
        }
        protein_sweep_observe(sw_, norm.data(), score, cost, is_failure ? 1 : 0);
    }

    bool early_stop(py::dict logs, const std::string& target_key) {
        check();
        if (logs.contains("loss")) {
            py::dict loss = logs["loss"].cast<py::dict>();
            for (auto item : loss) {
                if (std::isnan(item.second.cast<float>())) {
                    logs["is_loss_nan"] = true;
                    return true;
                }
            }
        }
        if (!logs.contains("uptime") || !logs.contains("env")) {
            return false;
        }

        py::dict env = logs["env"].cast<py::dict>();
        std::string key = target_key;
        if (key.substr(0, 4) == "env/") {
            key = key.substr(4);
        }
        if (!env.contains(key.c_str())) {
            return false;
        }

        float metric = env[key.c_str()].cast<float>();
        float cost = logs["uptime"].cast<float>();
        protein_sweep_add_running(sw_, metric);
        float running_mean = protein_sweep_running_mean(sw_);
        float threshold = protein_sweep_get_threshold(sw_, cost);
        logs["early_stop_threshold"] = std::max(threshold, -5.0f);
        if (protein_sweep_should_stop(sw_, std::max(running_mean, metric), cost)) {
            logs["is_loss_nan"] = false;
            return true;
        }
        return false;
    }
};
