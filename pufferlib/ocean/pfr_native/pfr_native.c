#define PFR_STATIC_ENV 1
#include "pfr_native_env.h"
#include <dlfcn.h>

/* Validate X11 display BEFORE calling InitWindow — raylib/GLFW crashes
 * with a NULL function pointer SEGV if GLFW init fails (no graceful error).
 * We dlopen libX11 to avoid header conflicts with raylib's Font type. */
static int check_x11_display(void) {
    void *libx11 = dlopen("libX11.so.6", RTLD_LAZY);
    if (!libx11) return 0;
    void *(*fn_open)(const char *) = dlsym(libx11, "XOpenDisplay");
    int (*fn_close)(void *) = dlsym(libx11, "XCloseDisplay");
    if (!fn_open || !fn_close) { dlclose(libx11); return 0; }
    void *dpy = fn_open(NULL);
    if (!dpy) { dlclose(libx11); return 0; }
    fn_close(dpy);
    dlclose(libx11);
    return 1;
}

int main() {
    Env env = {0};

    if (!check_x11_display()) {
        fprintf(stderr, "ERROR: Cannot open X11 display (DISPLAY=%s)\n"
                "Run with a valid display or set DISPLAY correctly.\n",
                getenv("DISPLAY") ? getenv("DISPLAY") : "(unset)");
        return 1;
    }

    allocate(&env);
    c_reset(&env);

    SetTraceLogLevel(LOG_WARNING);
    InitWindow(PFR_WINDOW_W, PFR_WINDOW_H, "PufferLib PFR Native");
    SetTargetFPS(15);
    pfr_load_world_map_texture();
    pfr_load_player_sprite();

    while (!WindowShouldClose()) {
        if      (IsKeyDown(KEY_UP))        env.actions[0] = 1;
        else if (IsKeyDown(KEY_DOWN))      env.actions[0] = 2;
        else if (IsKeyDown(KEY_LEFT))      env.actions[0] = 3;
        else if (IsKeyDown(KEY_RIGHT))     env.actions[0] = 4;
        else if (IsKeyDown(KEY_Z))         env.actions[0] = 5; /* A */
        else if (IsKeyDown(KEY_X))         env.actions[0] = 6; /* B */
        else if (IsKeyDown(KEY_ENTER))     env.actions[0] = 7; /* Start */
        else if (IsKeyDown(KEY_BACKSPACE)) env.actions[0] = 8; /* Select */
        else                               env.actions[0] = 0; /* None */

        c_step(&env);
        if (env.terminals[0]) c_reset(&env);
        c_render(&env);
    }

    free_allocated(&env);
    CloseWindow();
    return 0;
}
