#pragma once

/*
 * Current-build REK combat identities shared by contact measurement,
 * hit acceptance, and referee state. HandSide is the native REKApp enum
 * after BodyPartTag.TryGetSide conversion: left is 0 and right is 1.
 */

typedef enum RekG1BodyPartType {
    REK_G1_BODY_PART_NONE = 0,
    REK_G1_BODY_PART_HAND = 1,
    REK_G1_BODY_PART_FOOT = 2,
    REK_G1_BODY_PART_HEAD = 3,
    REK_G1_BODY_PART_TORSO = 4,
    REK_G1_BODY_PART_PELVIS = 5,
    REK_G1_BODY_PART_FOREARM = 6,
    REK_G1_BODY_PART_UPPER_ARM = 7,
    REK_G1_BODY_PART_SHIN = 8,
    REK_G1_BODY_PART_THIGH = 9,
    REK_G1_BODY_PART_KNEE = 10,
    REK_G1_BODY_PART_ELBOW = 11,
} RekG1BodyPartType;

typedef enum RekG1BodyZone {
    REK_G1_BODY_ZONE_UNKNOWN = 0,
    REK_G1_BODY_ZONE_HEAD = 1,
    REK_G1_BODY_ZONE_TORSO = 2,
    REK_G1_BODY_ZONE_PELVIS = 3,
    REK_G1_BODY_ZONE_LEFT_SHOULDER = 4,
    REK_G1_BODY_ZONE_RIGHT_SHOULDER = 5,
    REK_G1_BODY_ZONE_LEFT_ELBOW = 6,
    REK_G1_BODY_ZONE_RIGHT_ELBOW = 7,
    REK_G1_BODY_ZONE_LEFT_WRIST = 8,
    REK_G1_BODY_ZONE_RIGHT_WRIST = 9,
    REK_G1_BODY_ZONE_LEFT_FIST = 10,
    REK_G1_BODY_ZONE_RIGHT_FIST = 11,
    REK_G1_BODY_ZONE_LEFT_HIP = 12,
    REK_G1_BODY_ZONE_RIGHT_HIP = 13,
    REK_G1_BODY_ZONE_LEFT_KNEE = 14,
    REK_G1_BODY_ZONE_RIGHT_KNEE = 15,
    REK_G1_BODY_ZONE_LEFT_ANKLE = 16,
    REK_G1_BODY_ZONE_RIGHT_ANKLE = 17,
} RekG1BodyZone;

typedef enum RekG1HandSide {
    REK_G1_HAND_LEFT = 0,
    REK_G1_HAND_RIGHT = 1,
} RekG1HandSide;
