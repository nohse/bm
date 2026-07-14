"""Attention-weighted cond experiment. Face prompts MUST contain the word 'face'
(its cross-attention map is the spatial weight). Weight is built from the face
prompt and applied to BOTH face/non-face error maps (shared, as in the repo)."""
FACE_PROMPTS = {          # token whose attention = spatial weight is 'face'
    "F_face":    "a photo of a face",
    "F_human":   "a photo of a human face",
    "F_close":   "a close-up photo of a face",
    "F_show":    "a photo showing a face",
}
NONFACE_PROMPTS = {
    "N_faceless": "a faceless photo",
    "N_without":  "a photo without a face",
    "N_no":       "a photo of no face",
    "N_withno":   "a photo with no face",
}
ATTN_WORD = "face"
TRANGES = {
    "t50_950":  (50, 950),
    "t100_900": (100, 900),
    "t200_800": (200, 800),
    "t300_700": (300, 700),
    "t400_800": (400, 800),
    "t200_600": (200, 600),
}
NT = 15
SEEDS = [11, 22, 33]
K_LIST = [1, 2, 3, 4, 6, 8, 10, 12, 15]
