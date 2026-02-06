CLASS_NAMES = [
    "Clean",
    "Shorts",
    "Opens",
    "Bridges",
    "Vias",
    "Scratches",
    "Cracks",
    "LER",
    "Other",
]

# This is the order ImageFolder uses
IMAGEFOLDER_CLASSES = [
    "Bridges",
    "Clean",
    "Cracks",
    "LER",
    "Opens",
    "Other",
    "Scratches",
    "Shorts",
    "Vias",
]

# Map from ImageFolder label index -> index in CLASS_NAMES
IF_TO_FINAL = {
    0: CLASS_NAMES.index("Bridges"),
    1: CLASS_NAMES.index("Clean"),
    2: CLASS_NAMES.index("Cracks"),
    3: CLASS_NAMES.index("LER"),
    4: CLASS_NAMES.index("Opens"),
    5: CLASS_NAMES.index("Other"),
    6: CLASS_NAMES.index("Scratches"),
    7: CLASS_NAMES.index("Shorts"),
    8: CLASS_NAMES.index("Vias"),
}

NORM_MEAN = 0.45
NORM_STD = 0.23

DATASET_ROOT = r"C:\EdgeDefectNet\dataset"
