"""Pinned official photographs for BF16 held-out quality/performance controls.

Boxes are explicit manual readings of the published green rectangles, not an
upstream detector invocation. Published input_bbox images retain their green
overlay: compare original/native on those identical pixels, not to an assumed
pixel-exact reproduction of the paper's unpublished detector/camera settings.
"""
CASES={
    'dancer':('notebook/images/dancing.jpg','0112b0a32ea5860db6a6fc700804751528341a02bf07fed0329a0c2610f905d3',[600.,80.,1330.,1250.]),
    'football':('assets/qualitative_comparisons/sample1/input_bbox.png','d46f926e8067b79a18a8e657b93d8998e4aec63ec06dc6bd36814a29aafe840a',[11.,13.,683.,614.]),
    'rider':('assets/qualitative_comparisons/sample2/input_bbox.png','4df2c680c6aafcc3fd510382473961f4e0faceb3c5d6a15fe54835bfa44d6253',[223.,74.,499.,525.]),
    'yoga':('assets/qualitative_comparisons/sample3/input_bbox.png','3457c25ee225a39db76179eaed03a383a4f827ebe768d752d3d6137d768ac2a9',[438.,168.,849.,705.]),
}
