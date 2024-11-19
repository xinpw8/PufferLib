
// Sky vs ground dimensions:
// Sky: 0 to 266
// Ground: 267 to 613

// Background, day mountains:
// mount_color_1_gold (40, 155, 77), (134, 134, 29) (at 20 seconds)
// mount_color_2_whitish (48, 14, 199), (213, 214, 209) (at 40 seconds)
// mount_color_3_less_dark_green (78, 240, 28), (3, 59, 0) (at 1:40) (at 1:48) (at 1:54) (at 1:56)
// Background, night mountains:
// mount_color_4_gray (160, 0, 134), (142, 142, 142) (at 2:18 through 2:50)
// at 2:50, there are no mountains because this is fog weather.
// at 3:18, it is back to mount_color_4_gray.
// at 3:34, mountains are color (15, 161, 105), (187, 92, 37).
// at 3:52, mountains are color darkish_gold (36, 170, 77), (140, 128, 24).
// the cycle repeats from here, starting at 4:08 with mount_color_1_gold.
// at 4:28, mountains are mount_color_2_whitish again, etc.


// Background, sky:
// sky_color_1_blue (156, 149, 110), (44, 75, 189) (at 20 seconds) (at 40 seconds) (at 60 seconds)
// sky_color_2_dark_blue (157, 184, 92), (23, 34, 173) (at 1:40)
// sky_color_purp_dark_blue ((185, 141, 91), (112, 40, 153)), ((161, 155, 96), (40, 36, 168)) (at 1:48) (in order from horizon to top: purple, blue; purple fades into blue)
// 1:54 - sky now transitions from pinkish (217, 118, 90), (143, 49, 103) to purp to dark blue
// 1:56 - sky now transitions from orange-brown (13, 179, 63), (117, 49, 17) to red-orange (4, 181, 96), (179, 42, 25) to pinkish to purp to dark blue
// 2:00 - sky now transitions from lighter-orange-brown (11, 185, 95), (179, 66, 23) to red-orange to pinkish to purp to violet (183, 144, 94), (107, 40, 159). mountains are now almost-black (80, 240, 2), (0, 5, 0).
// 2:04 - sky now transitions from dark-pastel-orange (12, 163, 109), (195, 86, 37) to lighter-orange-brown to red-orange to pinkish.
// 2:10 - sky now transitions from brown-gold (19, 140, 95), (159,99,42) to (14, 170, 103), (187, 88, 32) to (12, 198, 90), (175, 63, 17) to (4, 176, 99), (182, 42, 28).
// 2:14 - sky now transitions from (40, 131, 75), (123, 123, 36) to (31, 151, 87), (150, 124, 34) to (26, 173, 80), (147, 104, 24) to (17, 163, 102), (182, 96, 35) to (13, 197, 89), (173, 66, 17).
// 2:18 - sky is solid gray (160, 0, 71), (75, 75, 75). this continues until 2:50, when the sky disappears due to fog weather.
// 3:18 - sky is again solid gray.
// 3:34 - sky is lighter gray (120, 1, 104), (110, 111, 111).
// 3:52 - sky is again sky_color_2_dark_blue.
// the cycle repeats from here, starting at 4:08 with sky_color_1_blue.


// Background, grass:
// grass_color_1_green (80, 240, 32), (0, 68, 0) (daytime grass color) (at 20 seconds) (at 40 seconds)
// snow_color_1_white (160, 0, 222), (236, 236, 236) (at 60 seconds)
// grass_color_2_dark_green (65, 240, 24), (19, 52, 0) (at 1:40) (at 1:48) (at 1:54) (at 1:56)
// grass_color_3_army_green (44, 240, 25), (48, 53, 0) (at 1:48 through 2:14)
// grass_color_4_black (160, 0, 0), (0, 0, 0) (at 2:18 through 2:50)
// grass_color_5_gray (160, 0, 71), (75, 75, 75) (at 2:50)
// grass_color_4_black again appears at 3:18
// at 3:52, grass_color_2_dark_green appears again.
// at 4:08, grass_color_1_green appears again, and the cycle repeats.


// mountain dimensions: there are only 2 mountains. these px values are scaled to 1016 width 613 height. they should be rescaled to 152 width 154 height.
// mountain_1 is a ziggarut-style step-mountain that is 38 px high, with the base on the horizon at 266 px on screen.
// each step going up is as follows: 264, 260, 258, 254, 250, 247 (top of plateau). steps going down follow the same pattern.
// mountain_1 starts on left at 34 px and spans to 247 px at its base.
// from L to R, steps ascend: 34-47 is the first step, 47-60 is the second step, 60-73 is the third step, 73-86 is the fourth step, 86-99 is the fifth step, 99-112 is the sixth step.
// from 99-127 is the plateau. there are 3 descending steps from 127-140, 140-153, with a 'valley' from 153-180. 
// then, there is 1 step up to a mini-plateau from 180-207, then steps down from 207-220, then from 220-233, and finally from 233-246.

// mountain_2 starts at 514 px on the horizon and spans to 701 px at its base. it is 17 px high. the top of its plateau is at 249 px.
// 528, 541, 581, and 594 are the starts of the steps. 594-621 is the plateau. 621-662, 662-675, 675-688 are the steps down, with the last step from 688-701.

// sky color transitions are from 266-260-254-249-244. Above 244 (from 244-0) is always a solid sky color. 244 is always the highest point of different sky colors.


// Close tail light color: (212, 108, 134), (193, 92, 163)
// Far tail light color: (239, 145, 140), (213, 85, 88)

// Color palette (HSV), (RGB):
// Enemy cars: (colors seem to be randomly distributed)
// teal (108, 99, 105), (66, 158, 130)
// goldenrod (40, 141, 96), (162, 162, 42)
// gold-gray (29, 117, 103), (162, 134, 56)
// perriwinkle (145, 123, 122), (66, 114, 194)

// Road palette:
// Road boundaries closest to horizon:
// slate (160, 0, 70), (74, 74, 74)
// Road boundaries middle:
// gray (160, 0, 104), (111, 111, 111)
// Road boundaries furthest from horizon:
// lightish gray (160, 0, 181), (192, 192, 192)

// HUD:
// Background:
// darkish red (0, 175, 91), (167, 26, 26)
// Text backlight:
// Brown-yellow (25, 126, 120), (195, 144, 61)
// Text:
// Sat black(==black, but still) (160, 0, 0), (0, 0, 0)
// Victory flag:
// Dark Pastel Green (82, 85, 117), (80, 168, 84)




// Update the init_color_transitions function
void init_color_transitions() {
    // Initialize sky transitions with all provided timings and colors

    // Note: Convert timings from minutes:seconds to steps (assuming 60 FPS)
    // For example, 0:20 (20 seconds) is 20 * 60 = 1200 steps

    // Sky colors provided in HSV, converted to RGB
    // sky_color_1_blue (156, 149, 110), (44, 75, 189)
    skyTransitions[0] = (SkyTransition){0, {{44, 75, 189, 255}}, 1}; // 0:00

    // sky_color_1_blue continues at 0:20 and 0:40
    skyTransitions[1] = (SkyTransition){1200, {{44, 75, 189, 255}}, 1}; // 0:20
    skyTransitions[2] = (SkyTransition){2400, {{44, 75, 189, 255}}, 1}; // 0:40

    // sky_color_snow (assuming white for snow)
    skyTransitions[3] = (SkyTransition){3600, {{236, 236, 236, 255}}, 1}; // 1:00

    // sky_color_2_dark_blue (157, 184, 92), (23, 34, 173)
    skyTransitions[4] = (SkyTransition){6000, {{23, 34, 173, 255}}, 1}; // 1:40

    // sky_color_purp_dark_blue ((185, 141, 91), (112, 40, 153)), ((161, 155, 96), (40, 36, 168))
    // At 1:48
    skyTransitions[5] = (SkyTransition){6480, {{143, 49, 103, 255}, {44, 75, 189, 255}}, 2};

    // At 1:54 - transition to pinkish
    skyTransitions[6] = (SkyTransition){6840, {{143, 49, 103, 255}, {112, 40, 153, 255}, {44, 75, 189, 255}}, 3};

    // At 1:56 - transition to orange-brown to red-orange to pinkish
    skyTransitions[7] = (SkyTransition){6960, {{117, 49, 17, 255}, {179, 42, 25, 255}, {143, 49, 103, 255}, {112, 40, 153, 255}}, 4};

    // Continue adding transitions as per the provided data...

    // For the rest of the sky transitions, you can fill them similarly based on the colors and timings provided.

    // Mountain transitions
    mountainTransitions[0] = (ColorTransition){0, (Color){134, 134, 29, 255}}; // mount_color_1_gold at 0:00
    mountainTransitions[1] = (ColorTransition){2400, (Color){213, 214, 209, 255}}; // mount_color_2_whitish at 0:40
    mountainTransitions[2] = (ColorTransition){6000, (Color){3, 59, 0, 255}}; // mount_color_3_less_dark_green at 1:40
    mountainTransitions[3] = (ColorTransition){8280, (Color){142, 142, 142, 255}}; // mount_color_4_gray at 2:18
    mountainTransitions[4] = (ColorTransition){10200, (Color){0, 0, 0, 0}}; // No mountains during fog at 2:50
    mountainTransitions[5] = (ColorTransition){11880, (Color){142, 142, 142, 255}}; // mount_color_4_gray at 3:18
    mountainTransitions[6] = (ColorTransition){12840, (Color){187, 92, 37, 255}}; // New mountain color at 3:34
    mountainTransitions[7] = (ColorTransition){13920, (Color){140, 128, 24, 255}}; // darkish_gold at 3:52
    mountainTransitions[8] = (ColorTransition){14880, (Color){134, 134, 29, 255}}; // mount_color_1_gold at 4:08

    // Grass transitions
    grassTransitions[0] = (ColorTransition){0, (Color){0, 68, 0, 255}}; // grass_color_1_green at 0:00
    grassTransitions[1] = (ColorTransition){3600, (Color){236, 236, 236, 255}}; // snow_color_1_white at 1:00
    grassTransitions[2] = (ColorTransition){6000, (Color){19, 52, 0, 255}}; // grass_color_2_dark_green at 1:40
    grassTransitions[3] = (ColorTransition){6480, (Color){48, 53, 0, 255}}; // grass_color_3_army_green at 1:48
    grassTransitions[4] = (ColorTransition){8280, (Color){0, 0, 0, 255}}; // grass_color_4_black at 2:18
    grassTransitions[5] = (ColorTransition){10200, (Color){75, 75, 75, 255}}; // grass_color_5_gray at 2:50
    grassTransitions[6] = (ColorTransition){11880, (Color){0, 0, 0, 255}}; // grass_color_4_black at 3:18
    grassTransitions[7] = (ColorTransition){13920, (Color){19, 52, 0, 255}}; // grass_color_2_dark_green at 3:52
    grassTransitions[8] = (ColorTransition){14880, (Color){0, 68, 0, 255}}; // grass_color_1_green at 4:08
}



// Implement the render_mountains function
void render_mountains(Client* client, Enduro* env) {
    // Don't render mountains during foggy weather
    if (env->currentMountainColor.a == 0 || (env->currentGrassColor.r == 0 && env->currentGrassColor.g == 0 && env->currentGrassColor.b == 0)) {
        return;
    }

    // Mountain dimensions scaled to your game's screen size
    // Simplified mountain rendering using triangles

    // Mountain 1
    Vector2 m1_base_left = {PLAYABLE_AREA_LEFT, VANISHING_POINT_Y};
    Vector2 m1_peak = {SCREEN_WIDTH / 4, VANISHING_POINT_Y - 38};
    Vector2 m1_base_right = {SCREEN_WIDTH / 2, VANISHING_POINT_Y};

    DrawTriangle(m1_base_left, m1_peak, m1_base_right, env->currentMountainColor);

    // Mountain 2
    Vector2 m2_base_left = {SCREEN_WIDTH / 2, VANISHING_POINT_Y};
    Vector2 m2_peak = {3 * SCREEN_WIDTH / 4, VANISHING_POINT_Y - 17};
    Vector2 m2_base_right = {SCREEN_WIDTH - PLAYABLE_AREA_LEFT, VANISHING_POINT_Y};

    DrawTriangle(m2_base_left, m2_peak, m2_base_right, env->currentMountainColor);
}



void render_grass(Client* client, Enduro* env) {
    DrawRectangle(0, VANISHING_POINT_Y, SCREEN_WIDTH, PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y, env->currentGrassColor);
}


// Update the update_background_colors function to use MyColorLerp
void update_background_colors(Enduro* env) {
    int step = env->step_count;

    // Update sky colors with interpolation
    for (int i = 0; i < NUM_SKY_TRANSITIONS - 1; i++) {
        if (step >= skyTransitions[i].step && step < skyTransitions[i + 1].step) {
            float t = (float)(step - skyTransitions[i].step) / (skyTransitions[i + 1].step - skyTransitions[i].step);
            for (int j = 0; j < skyTransitions[i].numColors; j++) {
                Color c1 = skyTransitions[i].colors[j];
                Color c2 = skyTransitions[i + 1].colors[j];
                env->currentSkyColors[j] = MyColorLerp(c1, c2, t);
            }
            printf("Current sky transition index: %d\n", i);

            env->currentSkyColorCount = skyTransitions[i].numColors;
            break;
        }
    }

    // Update mountain color with interpolation
    for (int i = 0; i < NUM_MOUNTAIN_TRANSITIONS - 1; i++) {
        if (step >= mountainTransitions[i].step && step < mountainTransitions[i + 1].step) {
            float t = (float)(step - mountainTransitions[i].step) / (mountainTransitions[i + 1].step - mountainTransitions[i].step);
            env->currentMountainColor = MyColorLerp(mountainTransitions[i].color, mountainTransitions[i + 1].color, t);
            break;
        }
    }

    // Update grass color with interpolation
    for (int i = 0; i < NUM_GRASS_TRANSITIONS - 1; i++) {
        if (step >= grassTransitions[i].step && step < grassTransitions[i + 1].step) {
            float t = (float)(step - grassTransitions[i].step) / (grassTransitions[i + 1].step - grassTransitions[i].step);
            env->currentGrassColor = MyColorLerp(grassTransitions[i].color, grassTransitions[i + 1].color, t);
            break;
        }
    }
}



// Update rendering functions to use the updated colors
void render_sky(Client* client, Enduro* env) {
    if (env->currentSkyColorCount == 1) {
        // Solid sky color
        DrawRectangle(0, 0, SCREEN_WIDTH, VANISHING_POINT_Y, env->currentSkyColors[0]);
    } else {
        // Gradient sky
        for (int y = 0; y < VANISHING_POINT_Y; y++) {
            float t = (float)y / VANISHING_POINT_Y;
            Color color = env->currentSkyColors[0];
            for (int i = 1; i < env->currentSkyColorCount; i++) {
                color = MyColorLerp(color, env->currentSkyColors[i], t);
            }
            DrawLine(0, y, SCREEN_WIDTH, y, color);
        }
    }
}


void render_scoreboard(Client* client, Enduro* env) {
    // Draw scoreboard as a red rectangle
    DrawRectangle(PLAYABLE_AREA_LEFT, PLAYABLE_AREA_BOTTOM, SCREEN_WIDTH - PLAYABLE_AREA_LEFT, SCREEN_HEIGHT - PLAYABLE_AREA_BOTTOM, RED);

    // Render the score information within the scoreboard
    DrawText(TextFormat("Score: %05i", env->score), 10, PLAYABLE_AREA_BOTTOM + 10, 10, WHITE);
    DrawText(TextFormat("Cars to Pass: %03i", env->carsToPass), 10, PLAYABLE_AREA_BOTTOM + 25, 10, WHITE);
    DrawText(TextFormat("Day: %i", env->day), 10, PLAYABLE_AREA_BOTTOM + 40, 10, WHITE);
    DrawText(TextFormat("Speed: %.2f", env->speed), 10, PLAYABLE_AREA_BOTTOM + 55, 10, WHITE);
    DrawText(TextFormat("Step: %i", env->step_count), 10, PLAYABLE_AREA_BOTTOM + 70, 10, WHITE);

    
    if (env->victoryFlagTimer > 0) {
        DrawText("Victory!", SCREEN_WIDTH / 2 - 30, PLAYABLE_AREA_BOTTOM + 85, 20, GREEN);
        env->victoryFlagTimer--;
    }
}



static inline Color MyColorLerp(Color c1, Color c2, float t) {
    Color result;
    result.r = (unsigned char)(c1.r + t * (c2.r - c1.r));
    result.g = (unsigned char)(c1.g + t * (c2.g - c1.g));
    result.b = (unsigned char)(c1.b + t * (c2.b - c1.b));
    result.a = (unsigned char)(c1.a + t * (c2.a - c1.a));
    return result;
}

// Pixel representation of the car, stored as coordinate pairs
static const int car_pixels[CAR_PIXELS_COUNT][2] = {
    {77, 147}, {77, 149}, {77, 151}, {77, 153},
    {78, 147}, {78, 149}, {78, 151}, {78, 153},
    {79, 144}, {79, 145}, {79, 146}, {79, 148}, {79, 150}, {79, 152}, {79, 154},
    {80, 144}, {80, 145}, {80, 146}, {80, 148}, {80, 150}, {80, 152}, {80, 154},
    {81, 145}, {81, 146}, {81, 148}, {81, 149}, {81, 150}, {81, 151}, {81, 152}, {81, 153},
    {82, 145}, {82, 146}, {82, 148}, {82, 149}, {82, 150}, {82, 151}, {82, 152}, {82, 153},
    {83, 144}, {83, 145}, {83, 146}, {83, 147}, {83, 148}, {83, 149}, {83, 150}, {83, 151},
    {83, 152}, {83, 153}, {83, 154},
    {84, 144}, {84, 145}, {84, 146}, {84, 147}, {84, 148}, {84, 149}, {84, 150}, {84, 151},
    {84, 152}, {84, 153}, {84, 154},
    {85, 144}, {85, 145}, {85, 146}, {85, 147}, {85, 148}, {85, 149}, {85, 150}, {85, 151},
    {85, 152}, {85, 153}, {85, 154},
    {86, 144}, {86, 145}, {86, 146}, {86, 147}, {86, 148}, {86, 149}, {86, 150}, {86, 151},
    {86, 152}, {86, 153}, {86, 154},
    {87, 145}, {87, 146}, {87, 148}, {87, 149}, {87, 150}, {87, 151}, {87, 152}, {87, 153},
    {88, 145}, {88, 146}, {88, 148}, {88, 149}, {88, 150}, {88, 151}, {88, 152}, {88, 153},
    {89, 144}, {89, 145}, {89, 146}, {89, 147}, {89, 149}, {89, 151}, {89, 153},
    {90, 144}, {90, 145}, {90, 146}, {90, 147}, {90, 149}, {90, 151}, {90, 153},
    {91, 148}, {91, 150}, {91, 152}, {91, 154},
    {92, 148}, {92, 150}, {92, 152}, {92, 154}
};