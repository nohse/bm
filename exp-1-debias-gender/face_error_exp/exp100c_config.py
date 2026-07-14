"""Round 3: CONTRASTIVE minimal-pair prompts (face vs no-face differ ONLY in the
face concept, same template) + many more t-ranges incl. low/mid bands, to try to
separate hands/bodies from faces and reach >=95/100. cond scheme."""
FACE_PROMPTS = {
    "F_with":     "a photo with a face",
    "F_of":       "a photo of a face",
    "F_contain":  "a photo containing a face",
    "F_visible":  "a photo where a face is visible",
    "F_human":    "a photo with a human face",
    "F_closeup":  "a close-up photo of a face",
}
NONFACE_PROMPTS = {
    "N_with":     "a photo with no face",
    "N_of":       "a photo of no face",
    "N_contain":  "a photo containing no face",
    "N_visible":  "a photo where no face is visible",
    "N_human":    "a photo with no human face",
    "N_closeup":  "a close-up photo with no face",
}
# matched minimal pairs (only the face concept differs)
PAIRS = [("F_with", "N_with"), ("F_of", "N_of"), ("F_contain", "N_contain"),
         ("F_visible", "N_visible"), ("F_human", "N_human"), ("F_closeup", "N_closeup")]

TRANGES = {
    # low noise (fine facial features present)
    "t50_400":  (50, 400),
    "t100_400": (100, 400),
    "t50_550":  (50, 550),
    "t100_500": (100, 500),
    "t200_500": (200, 500),
    # mid
    "t200_600": (200, 600),
    "t300_600": (300, 600),
    "t300_700": (300, 700),
    "t400_800": (400, 800),
    # broad
    "t50_950":  (50, 950),
    "t100_900": (100, 900),
    "t200_800": (200, 800),
    # narrow / higher
    "t500_700": (500, 700),
    "t400_600": (400, 600),
}
NT = 15
SEEDS = [11, 22, 33]
K_LIST = [1, 2, 3, 4, 6, 8, 10, 12, 15]
