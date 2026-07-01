#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>

#include "vecenv.h"
#include "config.h"
#include "puffernet.h"

cudaError_t cudaHostAlloc(void** p, size_t n, unsigned int flags) {
    (void)flags;
    *p = calloc(1, n);
    return *p ? cudaSuccess : 1;
}

cudaError_t cudaMalloc(void** p, size_t n) {
    *p = calloc(1, n);
    return *p ? cudaSuccess : 1;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind kind) {
    (void)kind;
    memcpy(dst, src, n);
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n,
        cudaMemcpyKind kind, cudaStream_t stream) {
    (void)stream;
    return cudaMemcpy(dst, src, n, kind);
}

cudaError_t cudaMemset(void* p, int value, size_t n) {
    memset(p, value, n);
    return cudaSuccess;
}

cudaError_t cudaFree(void* p) {
    free(p);
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void* p) {
    free(p);
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    (void)device;
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    (void)stream;
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
    (void)flags;
    *stream = NULL;
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    (void)stream;
    return cudaSuccess;
}

const char* cudaGetErrorString(cudaError_t err) {
    (void)err;
    return "cpu stub";
}

static int has_suffix(const char* s, const char* suffix) {
    size_t n = strlen(s);
    size_t m = strlen(suffix);
    if (n < m) {
        return 0;
    }
    return strcmp(s + n - m, suffix) == 0;
}

static void find_latest_checkpoint(const char* dir_name,
        char* out, size_t out_size, time_t* best_time) {
    DIR* dir = opendir(dir_name);
    if (!dir) {
        return;
    }

    struct dirent* ent = NULL;
    while ((ent = readdir(dir))) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }

        char full[1024];
        snprintf(full, sizeof(full), "%s/%s", dir_name, ent->d_name);
        struct stat st;
        if (stat(full, &st) != 0) {
            continue;
        }

        if (S_ISDIR(st.st_mode)) {
            find_latest_checkpoint(full, out, out_size, best_time);
        } else if (S_ISREG(st.st_mode) && has_suffix(full, ".bin") &&
                st.st_ctime >= *best_time) {
            *best_time = st.st_ctime;
            snprintf(out, out_size, "%s", full);
        }
    }

    closedir(dir);
}

static const char* model_path(Dict* cfg, const char* env_name,
        char* out, size_t out_size) {
    const char* path = puf_config_get(cfg, "base.load_model_path");
    if (path && strcmp(path, "latest") == 0) {
        char root[1024];
        snprintf(root, sizeof(root), "%s/%s",
            puf_config_str(cfg, "base.checkpoint_dir"), env_name);
        out[0] = 0;
        time_t best_time = 0;
        find_latest_checkpoint(root, out, out_size, &best_time);

        if (!out[0]) {
            fprintf(stderr, "no .bin checkpoints found in %s\n", root);
            exit(1);
        }
        return out;
    }

    if (path && strcmp(path, "None") != 0) {
        return path;
    }

    snprintf(out, out_size, "resources/%s/%s_weights.bin", env_name, env_name);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s ENV [section.key=value ...]\n", argv[0]);
        return 1;
    }

    const char* env_name = argv[1];
    Dict cfg = {0};
    puf_config_load_env(&cfg, env_name, argc - 2, argv + 2);

    if (strcmp(get_obs_dtype(), "FloatTensor") != 0) {
        fprintf(stderr, "cpu eval currently requires FloatTensor observations, got %s\n", get_obs_dtype());
        return 1;
    }

    char path_buf[1024];
    const char* path = model_path(&cfg, env_name, path_buf, sizeof(path_buf));
    Weights* weights = load_weights(path);
    if (!weights) {
        return 1;
    }

    int act_sizes[64];
    int num_actions = get_num_act_sizes();
    int* raw_act_sizes = get_act_sizes();
    for (int i = 0; i < num_actions; i++) {
        act_sizes[i] = raw_act_sizes[i];
    }

    int hidden_size = (int)puf_config_val(&cfg, "policy.hidden_size");
    int num_layers = (int)puf_config_val(&cfg, "policy.num_layers");
    PufferNet* net = make_puffernet(weights, 1, get_obs_size(),
        hidden_size, num_layers, act_sizes, num_actions);

    Dict* vec_kwargs = create_dict(4);
    dict_set(vec_kwargs, "total_agents", 1);
    dict_set(vec_kwargs, "num_buffers", 1);
    Dict* env_kwargs = dict_copy_prefix(&cfg, "env.");
    StaticVec* vec = create_static_vec(1, 1, 0, vec_kwargs, env_kwargs);
    static_vec_reset(vec);

    int frame = 0;
    static_vec_render(vec, 0);
    while (!WindowShouldClose()) {
        if (frame % 4 == 0) {
            forward_puffernet(net, (float*)vec->observations.data, vec->actions);
        }
        frame = (frame + 1) % 4;
        cpu_vec_step(vec);
        static_vec_render(vec, 0);
    }

    static_vec_close(vec);
    dict_free(vec_kwargs);
    dict_free(env_kwargs);
    free_puffernet(net);
    free(weights);
    puf_config_free(&cfg);
    return 0;
}
