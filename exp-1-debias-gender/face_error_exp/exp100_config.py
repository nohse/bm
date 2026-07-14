"""Shared config for the 100-image cond-only prompt x t-range x K study.
'back of a person' style content-describing negatives are intentionally excluded."""
FACE_PROMPTS = {
    "F_face":     "a photo of a face",
    "F_human":    "a photo of a human face",
    "F_close":    "a close-up photo of a person's face",
    "F_persface": "a photo of a person's face",
    "F_portrait": "a portrait photo of a person",
    "F_showface": "a photo showing a face",
}
NONFACE_PROMPTS = {
    "N_without":  "a photo without a face",
    "N_non":      "a photo of a non face",
    "N_no":       "a photo of no face",
    "N_faceless": "a faceless photo",
    "N_noperson": "a photo with no person",
    "N_empty":    "a photo of an empty scene",
}
TRANGES = {
    "t50_950":  (50, 950),
    "t100_900": (100, 900),
    "t200_800": (200, 800),
    "t300_700": (300, 700),
    "t400_800": (400, 800),
    "t300_600": (300, 600),
    "t200_600": (200, 600),
    "t500_700": (500, 700),
}
NT = 15                       # max number of timesteps (per the K<=15 constraint)
SEEDS = [11, 22, 33]          # fresh-noise repeats, averaged for robustness
K_LIST = [1, 2, 3, 4, 6, 8, 10, 12, 15]   # timestep counts to test (subsets of the 15)
