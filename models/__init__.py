# ------------------------------------------------------------------------
# Conditional DETR for Image Cropping
# ------------------------------------------------------------------------
# Modified from ConditionalDETR (https://github.com/Atten4Vis/ConditionalDETR)
# ------------------------------------------------------------------------


from .conditional_detr import build


def build_model(args):
    return build(args)
