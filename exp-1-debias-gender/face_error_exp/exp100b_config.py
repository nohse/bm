"""Round 2: face prompts made FRONTAL/eye-specific to stop firing on hands & bodies
(the systematic false positives), vs strong generic no-face negatives. cond scheme."""
FACE_PROMPTS = {
    "F_face":     "a photo of a face",
    "F_portrait": "a portrait photo of a person",
    "F_eyes":     "a photo of a face with eyes, nose and mouth",
    "F_lookcam":  "a photo of a face looking at the camera",
    "F_closeeye": "a close-up photo of a human face with eyes",
    "F_headface": "a headshot photo of a person's face",
}
NONFACE_PROMPTS = {
    "N_faceless": "a faceless photo",
    "N_without":  "a photo without a face",
    "N_no":       "a photo of no face",
    "N_novis":    "a photo with no visible face",
    "N_noperson": "a photo with no person",
    "N_blurry":   "a blurry out-of-focus photo",
}
TRANGES = {
    "t50_950":  (50, 950),
    "t100_900": (100, 900),
    "t200_800": (200, 800),
    "t400_800": (400, 800),
}
NT = 15
SEEDS = [11, 22, 33, 44, 55, 66]
K_LIST = [1, 2, 3, 4, 6, 8, 10, 12, 15]
