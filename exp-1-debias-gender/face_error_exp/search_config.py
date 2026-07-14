"""Shared search space for the cond-only face/no-face sweep (scheme fixed = cond)."""

FACE_PROMPTS = {
    "F_face":      "a photo of a face",
    "F_human":     "a photo of a human face",
    "F_close":     "a close-up photo of a person's face",
    "F_persface":  "a photo of a person's face",
    "F_portrait":  "a portrait photo of a person",
    "F_person":    "a photo of a person",
    "F_headshot":  "a headshot photo of a person",
    "F_showface":  "a photo showing a face",
    "F_facevis":   "a photo of a person's face, face clearly visible",
}
NONFACE_PROMPTS = {
    "N_without":   "a photo without a face",
    "N_non":       "a photo of a non face",
    "N_no":        "a photo of no face",
    "N_withno":    "a photo with no face",
    "N_faceless":  "a faceless photo",
    "N_back":      "a photo of the back of a person",
    "N_novis":     "a photo where no face is visible",
    "N_hidden":    "a photo of a person with no visible face",
    "N_nopeople":  "a photo of a scene without any people",
    "N_object":    "a photo of an object",
}

# diverse timestep ranges (t_lo, t_hi); K (count) is applied uniformly, swept later
TRANGES = {
    "t50_950":   (50, 950),
    "t100_900":  (100, 900),
    "t200_800":  (200, 800),
    "t300_700":  (300, 700),
    "t400_800":  (400, 800),
    "t400_600":  (400, 600),
    "t300_600":  (300, 600),
    "t200_600":  (200, 600),
    "t250_750":  (250, 750),
    "t350_650":  (350, 650),
    "t150_550":  (150, 550),
    "t300_800":  (300, 800),
    "t200_700":  (200, 700),
    "t100_600":  (100, 600),
    "t450_850":  (450, 850),
    "t500_700":  (500, 700),
}
K_MAIN = 25
SEEDS = [111, 222, 333, 444, 555]
