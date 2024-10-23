#include <SDL2/SDL.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480
#define VANISHING_POINT_Y 120  // Higher up in the screen
#define ROAD_BASE_WIDTH 400

SDL_Window* window = NULL;
SDL_Renderer* renderer = NULL;

bool initSDL() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL initialization failed: %s\n", SDL_GetError());
        return false;
    }

    window = SDL_CreateWindow("Enduro Road Renderer",
                            SDL_WINDOWPOS_UNDEFINED,
                            SDL_WINDOWPOS_UNDEFINED,
                            WINDOW_WIDTH,
                            WINDOW_HEIGHT,
                            SDL_WINDOW_SHOWN);
    if (!window) {
        printf("Window creation failed: %s\n", SDL_GetError());
        return false;
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        printf("Renderer creation failed: %s\n", SDL_GetError());
        return false;
    }

    return true;
}

void renderModernRoad(uint8_t frame) {
    // Clear screen with sky color
    SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0xFF, 0xFF);
    SDL_RenderClear(renderer);

    // Draw snow ground
    SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
    SDL_Rect ground = {0, VANISHING_POINT_Y, WINDOW_WIDTH, WINDOW_HEIGHT - VANISHING_POINT_Y};
    SDL_RenderFillRect(renderer, &ground);

    int centerX = WINDOW_WIDTH / 2;
    float baseOffset = sin(frame * 0.05) * 150; // 150; // Base curve offset
    
    // Store points for road edges
    SDL_Point leftEdge[WINDOW_HEIGHT];
    SDL_Point rightEdge[WINDOW_HEIGHT];
    int numPoints = 0;

    // Calculate road edges that converge to vanishing point
    for (int y = WINDOW_HEIGHT; y >= VANISHING_POINT_Y; y -= 2) {
        float progress = (float)(y - VANISHING_POINT_Y) / (WINDOW_HEIGHT - VANISHING_POINT_Y);
        float roadWidth = ROAD_BASE_WIDTH * progress;
        
        // Calculate curve that's more pronounced at the bottom
        float curveIntensity = progress * 0.5 * progress; // Square for more dramatic curve
        float xOffset = baseOffset * curveIntensity;
        
        leftEdge[numPoints].x = centerX - (roadWidth/2) + xOffset;
        leftEdge[numPoints].y = y;
        rightEdge[numPoints].x = centerX + (roadWidth/2) + xOffset;
        rightEdge[numPoints].y = y;
        numPoints++;
    }

    // Draw road edges
    SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0xFF);
    
    // Draw both edges with slight thickness for visibility
    for (int i = 0; i < numPoints - 1; i++) {
        // Left edge
        SDL_RenderDrawLine(renderer,
                          leftEdge[i].x, leftEdge[i].y,
                          leftEdge[i+1].x, leftEdge[i+1].y);
        SDL_RenderDrawLine(renderer,
                          leftEdge[i].x + 1, leftEdge[i].y,
                          leftEdge[i+1].x + 1, leftEdge[i+1].y);
                          
        // Right edge
        SDL_RenderDrawLine(renderer,
                          rightEdge[i].x, rightEdge[i].y,
                          rightEdge[i+1].x, rightEdge[i+1].y);
        SDL_RenderDrawLine(renderer,
                          rightEdge[i].x + 1, rightEdge[i].y,
                          rightEdge[i+1].x + 1, rightEdge[i+1].y);
    }
}

void cleanup() {
    if (renderer) SDL_DestroyRenderer(renderer);
    if (window) SDL_DestroyWindow(window);
    SDL_Quit();
}

int main(int argc, char* argv[]) {
    if (!initSDL()) {
        cleanup();
        return 1;
    }

    uint8_t frame = 0;
    bool running = true;
    SDL_Event event;

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }

        renderModernRoad(frame);
        SDL_RenderPresent(renderer);
        
        frame++;
        SDL_Delay(16);
    }

    cleanup();
    return 0;
}


// Road curve template:

// seconds in between road curve changes
// R=rightward curve
// L=leftward curve
// S=straight road (no curve)


// R->S
// 4s
// S->L
// 16s
// L->S
// 2s
// S->R
// 4s
// R->L
// 8s
// L->S
// 16s
// S->L
// 12s
// L->S
// 4s
// S->R
// 22s
// R->S
// 24s
// S->R
// 20s
// R->S
// 8s
// S->R




