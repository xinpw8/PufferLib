#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "ocean/osrs/encounters/encounter_inferno.h"

static char* read_script_line(FILE* file) {
    size_t cap = 256;
    size_t len = 0;
    char* line = (char*)malloc(cap);
    if (!line) {
        fprintf(stderr, "inferno lab: out of memory\n");
        abort();
    }

    int ch;
    while ((ch = fgetc(file)) != EOF) {
        if (len + 2 > cap) {
            if (cap > SIZE_MAX / 2) {
                fprintf(stderr, "inferno lab: script line too large\n");
                abort();
            }
            cap *= 2;
            char* next = (char*)realloc(line, cap);
            if (!next) {
                fprintf(stderr, "inferno lab: out of memory\n");
                abort();
            }
            line = next;
        }
        line[len++] = (char)ch;
        if (ch == '\n') break;
    }

    if (len == 0 && ch == EOF) {
        free(line);
        return NULL;
    }
    line[len] = '\0';
    return line;
}

static void run_script(
    FILE* file,
    InfernoState* state,
    InfernoContext* context
) {
    for (;;) {
        char* line = read_script_line(file);
        if (!line) break;

        char* dump = NULL;
        InfLabLineResult result = inf_lab_apply_script_line_impl_ctx(
            state, context, line, &dump);
        if (result == INF_LAB_LINE_DUMP) {
            printf("%s\n", dump);
            free(dump);
        }
        free(line);
    }
}

int main(int argc, char** argv) {
    InfernoState* state = (InfernoState*)inf_create();
    InfernoContext context;
    inf_init_context_typed(&context);
    inf_finalize_route_topology(&context);
    inf_put_float_ctx(
        (EncounterState*)state,
        (EncounterContext*)&context,
        "late_start_supply_profile_scale",
        1.0f);
    inf_reset_ctx(
        (EncounterState*)state, (EncounterContext*)&context, 20260515u);
    inf_lab_apply_command_ctx(state, &context, &(InfernoLabCommand){
        .kind = INF_LAB_COMMAND_CLEAR_NPCS,
    });

    if (argc == 1) {
        run_script(stdin, state, &context);
    } else if (argc == 2) {
        FILE* file = fopen(argv[1], "r");
        if (!file) {
            fprintf(stderr, "inferno lab: cannot open %s\n", argv[1]);
            abort();
        }
        run_script(file, state, &context);
        fclose(file);
    } else {
        fprintf(stderr, "usage: inferno_lab [script]\n");
        inf_destroy((EncounterState*)state);
        return 2;
    }

    inf_destroy((EncounterState*)state);
    return 0;
}
