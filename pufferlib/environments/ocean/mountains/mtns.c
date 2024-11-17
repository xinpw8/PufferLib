// mtns.c (this was the separate rendering file I wrote that rendered all background elements correctly)
// it was mostly integrated into the enduro_clone.h file, but I left it here for reference
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h> // Include for bool type

// Mountain PNG Bottom-Left Corner: (37, 61)
// Scoreboard Bottom-Left Corner: (48, 190)
// Score Section Bottom-Left Corner: (56, 172)
// Day Number Bottom-Left Corner: (56, 188)
// Cars Passed Bottom-Left Corner: (72, 187)

// Screen dimensions
#define SCREEN_WIDTH  160
#define SCREEN_HEIGHT 210  // Updated to 210 as per your specifications

// Number of digits in the scoreboard
#define SCORE_DIGITS 5
#define DAY_DIGITS   1
#define CARS_DIGITS  4

// Digit dimensions
#define DIGIT_WIDTH 8
#define DIGIT_HEIGHT 9

// Enumeration for road direction
typedef enum {
    ROAD_STRAIGHT,
    ROAD_TURN_LEFT,
    ROAD_TURN_RIGHT
} RoadDirection;

// Structure to hold game state related to rendering
typedef struct {
    SDL_Texture* backgroundTextures[16]; // 16 different backgrounds for time of day
    SDL_Texture* digitTextures[10];      // Textures for digits 0-9
    SDL_Texture* carDigitTexture;        // Texture for the "CAR" digit
    SDL_Texture* mountainTextures[16];   // Mountain textures corresponding to backgrounds

    SDL_Texture* levelCompleteFlagLeftTexture;  // New texture for left flag
    SDL_Texture* levelCompleteFlagRightTexture; // New texture for right flag
    SDL_Texture* greenDigitTextures[10];        // New textures for green digits
    SDL_Texture* yellowDigitTextures[10];       // New textures for yellow digits

    int currentBackgroundIndex;
    int previousBackgroundIndex;
    int score;
    int day;
    int carsToPass;
    float mountainPosition; // Position of the mountain texture

    bool victoryAchieved;   // Flag to indicate victory condition

    // Variables for alternating flags
    int flagTimer;
    bool showLeftFlag;

    // Variables for scrolling yellow digits
    float yellowDigitOffset; // Offset for scrolling effect
    int yellowDigitCurrent;  // Current yellow digit being displayed
    int yellowDigitNext;     // Next yellow digit to scroll in

    // Variables for scrolling digits
    float scoreDigitOffsets[SCORE_DIGITS];   // Offset for scrolling effect for each digit
    int scoreDigitCurrents[SCORE_DIGITS];    // Current digit being displayed for each position
    int scoreDigitNexts[SCORE_DIGITS];       // Next digit to scroll in for each position
    bool scoreDigitScrolling[SCORE_DIGITS];  // Scrolling state for each digit

    int scoreTimer; // Timer to control score increment
    int victoryDisplayTimer; // To track how long the victory flags have been displayed

} GameState;

// Function prototypes
int initSDL(SDL_Window** window, SDL_Renderer** renderer);
void loadTextures(SDL_Renderer* renderer, GameState* gameState);
void cleanup(SDL_Window* window, SDL_Renderer* renderer, GameState* gameState);
void updateBackground(GameState* gameState, int timeOfDay);
void renderBackground(SDL_Renderer* renderer, GameState* gameState);
void renderScoreboard(SDL_Renderer* renderer, GameState* gameState);
void updateMountains(GameState* gameState, RoadDirection direction);
void renderMountains(SDL_Renderer* renderer, GameState* gameState);
void handleEvents(int* running, int* timeOfDay, RoadDirection* roadDirection, GameState* gameState);
void updateVictoryEffects(GameState* gameState); // New function to update victory effects
void updateScore(GameState* gameState); // New function to update score over time

int main(int argc, char* argv[]) {
    SDL_Window* window = NULL;
    SDL_Renderer* renderer = NULL;
    GameState gameState = {0};

    if (initSDL(&window, &renderer) != 0) {
        return -1;
    }

    loadTextures(renderer, &gameState);

    // Main game loop variables
    int running = 1;
    int timeOfDay = 0; // This should be updated according to your game logic
    RoadDirection roadDirection = ROAD_STRAIGHT;

    // Initialize game state variables
    gameState.previousBackgroundIndex = -1; // Initialize to an invalid index

    // Initialize new game state variables
    gameState.victoryAchieved = false;
    gameState.flagTimer = 0;
    gameState.showLeftFlag = true;
    gameState.yellowDigitOffset = 0.0f;

    // Initialize score digit scrolling variables
    for (int i = 0; i < SCORE_DIGITS; i++) {
        gameState.scoreDigitOffsets[i] = 0.0f;
        int digitValue = (gameState.score / (int)pow(10, SCORE_DIGITS - i - 1)) % 10;
        gameState.scoreDigitCurrents[i] = digitValue;
        gameState.scoreDigitNexts[i] = (digitValue + 1) % 10;
        gameState.scoreDigitScrolling[i] = false;
    }
    gameState.scoreTimer = 0;
    
    // Initialize victory display timer
    gameState.victoryDisplayTimer = 0;

    // Update background to set initial state and print initial images
    updateBackground(&gameState, timeOfDay);

    // Render scoreboard once to print initial digit images
    renderScoreboard(renderer, &gameState);

    while (running) {
        // Event handling
        handleEvents(&running, &timeOfDay, &roadDirection, &gameState);

        // Update victory effects
        updateVictoryEffects(&gameState);

        // Update game state
        updateBackground(&gameState, timeOfDay);
        updateMountains(&gameState, roadDirection);

        // Update victory effects
        updateVictoryEffects(&gameState);

        // Update score
        updateScore(&gameState);

        // Clear screen
        SDL_RenderClear(renderer);

        // Render functions
        renderBackground(renderer, &gameState);
        renderMountains(renderer, &gameState);
        renderScoreboard(renderer, &gameState);

        // Present renderer
        SDL_RenderPresent(renderer);

        // Delay to control frame rate
        SDL_Delay(16); // Approximately 60 FPS
    }

    // Cleanup
    cleanup(window, renderer, &gameState);

    return 0;
}

void handleEvents(int* running, int* timeOfDay, RoadDirection* roadDirection, GameState* gameState) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            *running = 0;
        } else if (event.type == SDL_KEYDOWN) {
            if (event.key.keysym.sym == SDLK_SPACE && !gameState->victoryAchieved) {
                // Advance time of day
                *timeOfDay = (*timeOfDay + 1) % 16; // Cycle through 0 to 15

                // Increment score by 1
                gameState->score += 1;
                if (gameState->score > 99999) { // Max score based on SCORE_DIGITS
                    gameState->score = 0;
                }

                // For testing, decrement cars to pass
                gameState->carsToPass -= 1;
                if (gameState->carsToPass < 0) {
                    gameState->carsToPass = 0;
                }
            } else if (event.key.keysym.sym == SDLK_LEFT) {
                *roadDirection = ROAD_TURN_LEFT;
            } else if (event.key.keysym.sym == SDLK_RIGHT) {
                *roadDirection = ROAD_TURN_RIGHT;
            } else if (event.key.keysym.sym == SDLK_UP) {
                *roadDirection = ROAD_STRAIGHT;
            }
        }
    }
}

int initSDL(SDL_Window** window, SDL_Renderer** renderer) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return -1;
    }

    // Create window with exact background size
    *window = SDL_CreateWindow("Enduro Port Framework",
                               SDL_WINDOWPOS_UNDEFINED,
                               SDL_WINDOWPOS_UNDEFINED,
                               SCREEN_WIDTH,
                               SCREEN_HEIGHT,
                               SDL_WINDOW_SHOWN);

    if (*window == NULL) {
        printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        return -1;
    }

    // Create renderer without any scaling
    *renderer = SDL_CreateRenderer(*window, -1, SDL_RENDERER_ACCELERATED);

    if (*renderer == NULL) {
        printf("Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
        return -1;
    }

    // Initialize PNG loading
    if (!(IMG_Init(IMG_INIT_PNG) & IMG_INIT_PNG)) {
        printf("SDL_image could not initialize! SDL_image Error: %s\n", IMG_GetError());
        return -1;
    }

    return 0;
}

void loadTextures(SDL_Renderer* renderer, GameState* gameState) {
    // Load background and mountain textures for different times of day
    char backgroundFile[20];
    char mountainFile[20];

    for (int i = 0; i < 16; ++i) {
        snprintf(backgroundFile, sizeof(backgroundFile), "%d_bg.png", i);
        SDL_Surface* bgSurface = IMG_Load(backgroundFile);
        if (!bgSurface) {
            printf("Failed to load background image %s! SDL_image Error: %s\n", backgroundFile, IMG_GetError());
            continue;
        }
        gameState->backgroundTextures[i] = SDL_CreateTextureFromSurface(renderer, bgSurface);
        SDL_FreeSurface(bgSurface);
        printf("Loaded background image: %s\n", backgroundFile);

        snprintf(mountainFile, sizeof(mountainFile), "%d_mtns.png", i);
        SDL_Surface* mtnSurface = IMG_Load(mountainFile);
        if (!mtnSurface) {
            printf("Failed to load mountain image %s! SDL_image Error: %s\n", mountainFile, IMG_GetError());
            continue;
        }
        gameState->mountainTextures[i] = SDL_CreateTextureFromSurface(renderer, mtnSurface);
        SDL_FreeSurface(mtnSurface);
        printf("Loaded mountain image: %s\n", mountainFile);
    }

    // Load digit textures 0-9
    char filename[30];
    for (int i = 0; i < 10; ++i) {
        snprintf(filename, sizeof(filename), "digits_%d.png", i);
        SDL_Surface* surface = IMG_Load(filename);
        if (!surface) {
            printf("Failed to load digit image %s! SDL_image Error: %s\n", filename, IMG_GetError());
            continue;
        }
        gameState->digitTextures[i] = SDL_CreateTextureFromSurface(renderer, surface);
        SDL_FreeSurface(surface);
        printf("Loaded digit image: %s\n", filename);
    }

    // Load the "CAR" digit texture
    SDL_Surface* carSurface = IMG_Load("digits_car.png");
    if (!carSurface) {
        printf("Failed to load digit image digits_car.png! SDL_image Error: %s\n", IMG_GetError());
    } else {
        gameState->carDigitTexture = SDL_CreateTextureFromSurface(renderer, carSurface);
        SDL_FreeSurface(carSurface);
        printf("Loaded digit image: digits_car.png\n");
    }

    // Load level complete flag textures
    SDL_Surface* flagLeftSurface = IMG_Load("level_complete_flag_left.png");
    if (!flagLeftSurface) {
        printf("Failed to load image level_complete_flag_left.png! SDL_image Error: %s\n", IMG_GetError());
    } else {
        gameState->levelCompleteFlagLeftTexture = SDL_CreateTextureFromSurface(renderer, flagLeftSurface);
        SDL_FreeSurface(flagLeftSurface);
        printf("Loaded image: level_complete_flag_left.png\n");
    }

    SDL_Surface* flagRightSurface = IMG_Load("level_complete_flag_right.png");
    if (!flagRightSurface) {
        printf("Failed to load image level_complete_flag_right.png! SDL_image Error: %s\n", IMG_GetError());
    } else {
        gameState->levelCompleteFlagRightTexture = SDL_CreateTextureFromSurface(renderer, flagRightSurface);
        SDL_FreeSurface(flagRightSurface);
        printf("Loaded image: level_complete_flag_right.png\n");
    }

    // Load green digits
    for (int i = 0; i < 10; ++i) {
        snprintf(filename, sizeof(filename), "green_digits_%d.png", i);
        SDL_Surface* surface = IMG_Load(filename);
        if (!surface) {
            printf("Failed to load image %s! SDL_image Error: %s\n", filename, IMG_GetError());
        } else {
            gameState->greenDigitTextures[i] = SDL_CreateTextureFromSurface(renderer, surface);
            SDL_FreeSurface(surface);
            printf("Loaded image: %s\n", filename);
        }
    }

    // Load yellow digits
    for (int i = 0; i < 10; ++i) {
        snprintf(filename, sizeof(filename), "yellow_digits_%d.png", i);
        SDL_Surface* surface = IMG_Load(filename);
        if (!surface) {
            printf("Failed to load image %s! SDL_image Error: %s\n", filename, IMG_GetError());
        } else {
            gameState->yellowDigitTextures[i] = SDL_CreateTextureFromSurface(renderer, surface);
            SDL_FreeSurface(surface);
            printf("Loaded image: %s\n", filename);
        }
    }

    // Initialize other game state variables
    gameState->currentBackgroundIndex = 0;
    gameState->score = 0;
    gameState->day = 1;
    gameState->carsToPass = 200;
    gameState->mountainPosition = 0.0f;
}

void cleanup(SDL_Window* window, SDL_Renderer* renderer, GameState* gameState) {
    // Destroy textures
    for (int i = 0; i < 16; ++i) {
        if (gameState->backgroundTextures[i]) {
            SDL_DestroyTexture(gameState->backgroundTextures[i]);
        }
        if (gameState->mountainTextures[i]) {
            SDL_DestroyTexture(gameState->mountainTextures[i]);
        }
    }

    for (int i = 0; i < 10; ++i) {
        if (gameState->digitTextures[i]) {
            SDL_DestroyTexture(gameState->digitTextures[i]);
        }
        if (gameState->greenDigitTextures[i]) {
            SDL_DestroyTexture(gameState->greenDigitTextures[i]);
        }
        if (gameState->yellowDigitTextures[i]) {
            SDL_DestroyTexture(gameState->yellowDigitTextures[i]);
        }
    }

    if (gameState->carDigitTexture) {
        SDL_DestroyTexture(gameState->carDigitTexture);
    }

    if (gameState->levelCompleteFlagLeftTexture) {
        SDL_DestroyTexture(gameState->levelCompleteFlagLeftTexture);
    }

    if (gameState->levelCompleteFlagRightTexture) {
        SDL_DestroyTexture(gameState->levelCompleteFlagRightTexture);
    }

    // Destroy renderer and window
    if (renderer) SDL_DestroyRenderer(renderer);
    if (window) SDL_DestroyWindow(window);

    // Quit SDL subsystems
    IMG_Quit();
    SDL_Quit();
}

void updateScore(GameState* gameState) {
    // Increase the score every 30 frames (~0.5 seconds at 60 FPS)
    gameState->scoreTimer++;
    if (gameState->scoreTimer >= 30) {
        gameState->scoreTimer = 0;
        gameState->score += 1;
        if (gameState->score > 99999) { // Max score based on SCORE_DIGITS
            gameState->score = 0;
        }

        // Determine which digits have changed and start scrolling them
        int tempScore = gameState->score;
        for (int i = SCORE_DIGITS - 1; i >= 0; i--) {
            int newDigit = tempScore % 10;
            tempScore /= 10;

            if (newDigit != gameState->scoreDigitCurrents[i]) {
                gameState->scoreDigitNexts[i] = newDigit;
                gameState->scoreDigitOffsets[i] = 0.0f;
                gameState->scoreDigitScrolling[i] = true;
            }
        }
    }

    // Update scrolling digits
    for (int i = 0; i < SCORE_DIGITS; i++) {
        if (gameState->scoreDigitScrolling[i]) {
            gameState->scoreDigitOffsets[i] += 0.5f; // Adjust scroll speed as needed
            if (gameState->scoreDigitOffsets[i] >= DIGIT_HEIGHT) {
                gameState->scoreDigitOffsets[i] = 0.0f;
                gameState->scoreDigitCurrents[i] = gameState->scoreDigitNexts[i];
                gameState->scoreDigitScrolling[i] = false; // Stop scrolling
            }
        }
    }
}

void updateBackground(GameState* gameState, int timeOfDay) {
    gameState->previousBackgroundIndex = gameState->currentBackgroundIndex;
    gameState->currentBackgroundIndex = timeOfDay % 16;

    // Print the background and mountain images whenever they are displayed
    printf("Background image displayed: %d_bg.png\n", gameState->currentBackgroundIndex);
    printf("Mountain image displayed: %d_mtns.png\n", gameState->currentBackgroundIndex);

    // Check for victory condition
    if (!gameState->victoryAchieved && gameState->currentBackgroundIndex == 0 && gameState->previousBackgroundIndex == 15) {
        // Victory condition achieved
        gameState->victoryAchieved = true;
        gameState->carsToPass = 0; // Set cars to pass to 0
        printf("Victory achieved!\n");
    }
}


void renderBackground(SDL_Renderer* renderer, GameState* gameState) {
    SDL_Texture* bgTexture = gameState->backgroundTextures[gameState->currentBackgroundIndex];
    if (bgTexture) {
        // Render background at its native size without scaling
        SDL_RenderCopy(renderer, bgTexture, NULL, NULL);
    }
}

void renderScoreboard(SDL_Renderer* renderer, GameState* gameState) {
    // Positions and sizes
    int digitWidth = DIGIT_WIDTH;
    int digitHeight = DIGIT_HEIGHT;

    // Convert bottom-left coordinates to SDL coordinates (top-left origin)
    int scoreStartX = 56 + digitWidth;
    int scoreStartY = 173 - digitHeight;
    int dayX = 56;
    int dayY = 188 - digitHeight;
    int carsX = 72;
    int carsY = 188 - digitHeight;

    // Render score with scrolling effect
    for (int i = 0; i < SCORE_DIGITS; ++i) {
        int digitX = scoreStartX + i * digitWidth;
        SDL_Rect destRect = { digitX, scoreStartY, digitWidth, digitHeight };

        SDL_Texture* currentDigitTexture = gameState->digitTextures[gameState->scoreDigitCurrents[i]];
        SDL_Texture* nextDigitTexture = gameState->digitTextures[gameState->scoreDigitNexts[i]];

        if (gameState->scoreDigitScrolling[i]) {
            // Scrolling effect for this digit
            float offset = gameState->scoreDigitOffsets[i];

            // Render current digit moving up
            SDL_Rect srcRectCurrent = { 0, 0, digitWidth, digitHeight - (int)offset };
            SDL_Rect destRectCurrent = { digitX, scoreStartY + (int)offset, digitWidth, digitHeight - (int)offset };
            SDL_RenderCopy(renderer, currentDigitTexture, &srcRectCurrent, &destRectCurrent);

            // Render next digit coming up from below
            SDL_Rect srcRectNext = { 0, digitHeight - (int)offset, digitWidth, (int)offset };
            SDL_Rect destRectNext = { digitX, scoreStartY, digitWidth, (int)offset };
            SDL_RenderCopy(renderer, nextDigitTexture, &srcRectNext, &destRectNext);

            printf("Rendering scrolling score digit: digits_%d.png and digits_%d.png at position (%d, %d)\n",
                   gameState->scoreDigitCurrents[i], gameState->scoreDigitNexts[i], destRect.x, destRect.y);
        } else {
            // No scrolling, render the current digit normally
            SDL_RenderCopy(renderer, currentDigitTexture, NULL, &destRect);
            printf("Rendering score digit: digits_%d.png at position (%d, %d)\n",
                   gameState->scoreDigitCurrents[i], destRect.x, destRect.y);
        }
    }

    // Render day number
    int day = gameState->day % 10;
    SDL_Rect dayRect = { dayX, dayY, digitWidth, digitHeight };

    if (gameState->victoryAchieved) {
        // Use green digits during victory
        SDL_Texture* greenDigitTexture = gameState->greenDigitTextures[day];
        SDL_RenderCopy(renderer, greenDigitTexture, NULL, &dayRect);
        printf("Rendering day digit: green_digits_%d.png at position (%d, %d)\n", day, dayRect.x, dayRect.y);
    } else {
        // Use normal digits
        SDL_RenderCopy(renderer, gameState->digitTextures[day], NULL, &dayRect);
        printf("Rendering day digit: digits_%d.png at position (%d, %d)\n", day, dayRect.x, dayRect.y);
    }

    // Render "CAR" digit or flags for cars to pass
    if (gameState->victoryAchieved) {
        // Alternate between level_complete_flag_left and level_complete_flag_right
        SDL_Texture* flagTexture = gameState->showLeftFlag ? gameState->levelCompleteFlagLeftTexture : gameState->levelCompleteFlagRightTexture;
        SDL_Rect flagRect = { carsX, carsY, digitWidth * 4, digitHeight };
        SDL_RenderCopy(renderer, flagTexture, NULL, &flagRect);
        printf("Rendering level complete flag: %s at position (%d, %d)\n",
               gameState->showLeftFlag ? "level_complete_flag_left.png" : "level_complete_flag_right.png", flagRect.x, flagRect.y);
    } else {
        // Render "CAR" digit for the first position in cars to pass
        SDL_Rect carDestRect = { carsX, carsY, digitWidth, digitHeight };
        SDL_RenderCopy(renderer, gameState->carDigitTexture, NULL, &carDestRect);
        printf("Rendering cars to pass digit: digits_car.png at position (%d, %d)\n", carDestRect.x, carDestRect.y);

        // Render the remaining digits for cars to pass
        int cars = gameState->carsToPass;
        for (int i = 1; i < CARS_DIGITS; ++i) {
            int digit = (cars / (int)pow(10, CARS_DIGITS - i - 1)) % 10;
            SDL_Rect destRect = { carsX + i * digitWidth + i * 1, carsY, digitWidth, digitHeight };
            SDL_RenderCopy(renderer, gameState->digitTextures[digit], NULL, &destRect);
            printf("Rendering cars to pass digit: digits_%d.png at position (%d, %d)\n", digit, destRect.x, destRect.y);
        }
    }
}




void updateVictoryEffects(GameState* gameState) {
    if (gameState->victoryAchieved) {
        // Update flag timer
        gameState->flagTimer++;
        if (gameState->flagTimer >= 30) { // Switch every 30 frames (~0.5 sec at 60 FPS)
            gameState->flagTimer = 0;
            gameState->showLeftFlag = !gameState->showLeftFlag;
        }

        // Update victory display timer
        gameState->victoryDisplayTimer++;
        if (gameState->victoryDisplayTimer >= 180) { // Display flags for 180 frames (~3 seconds)
            // Reset victory state
            gameState->victoryAchieved = false;
            gameState->victoryDisplayTimer = 0;

            // Increment day
            gameState->day += 1;
            if (gameState->day > 9) { // Assuming single-digit day numbers
                gameState->day = 1;
            }

            // Reset cars to pass for the new day
            gameState->carsToPass = 200; // Or set according to your game logic

            // Reset flags
            gameState->flagTimer = 0;
            gameState->showLeftFlag = true;

            printf("Starting new day: %d\n", gameState->day);
        }
    }
}


void updateMountains(GameState* gameState, RoadDirection direction) {
    // Adjust the mountain position based on the road direction
    float speed = 1.0f; // Adjust the speed as needed
    int mountainWidth = 100;

    if (direction == ROAD_TURN_LEFT) {
        gameState->mountainPosition += speed;
        if (gameState->mountainPosition >= mountainWidth) {
            gameState->mountainPosition -= mountainWidth;
        }
    } else if (direction == ROAD_TURN_RIGHT) {
        gameState->mountainPosition -= speed;
        if (gameState->mountainPosition <= -mountainWidth) {
            gameState->mountainPosition += mountainWidth;
        }
    }
    // If the road is straight, the mountains don't move
}

void renderMountains(SDL_Renderer* renderer, GameState* gameState) {
    SDL_Texture* mountainTexture = gameState->mountainTextures[gameState->currentBackgroundIndex];
    if (mountainTexture) {
        int mountainWidth = 100;
        int mountainHeight = 6;
        int mountainX = (int)gameState->mountainPosition + 37;
        int mountainY = 45; // Corrected Y-coordinate

        SDL_Rect destRect1 = { mountainX, mountainY, mountainWidth, mountainHeight };
        SDL_RenderCopy(renderer, mountainTexture, NULL, &destRect1);

        // Handle wrapping
        if (mountainX > SCREEN_WIDTH - mountainWidth) {
            SDL_Rect destRect2 = { mountainX - mountainWidth, mountainY, mountainWidth, mountainHeight };
            SDL_RenderCopy(renderer, mountainTexture, NULL, &destRect2);
        } else if (mountainX < 0) {
            SDL_Rect destRect2 = { mountainX + mountainWidth, mountainY, mountainWidth, mountainHeight };
            SDL_RenderCopy(renderer, mountainTexture, NULL, &destRect2);
        }
    }
}