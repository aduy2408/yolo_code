# Taste (Continuously Learned by [CommandCode][cmd])

[cmd]: https://commandcode.ai/

# data-processing
- For the Varroa dataset, use class policy that maps class 3 to class 1 (map-3-to-1) instead of filtering to only class 1, since classes 1 and 3 represent the same class with different box tightness. Confidence: 0.65
- Use `round()` instead of `int()` for dataset split calculations (e.g., val/test sizes) to avoid flooring down. Confidence: 0.70

