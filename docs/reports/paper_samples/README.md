# Paper dataset sample figures

Each dataset now has separate standalone PNG files. No panel montage is used.

- `*_full.png`: full image with all annotations, all boxes in red.
- `*_context_crop.png` and `*_object_zoom.png`: only for LEVIR-Ship and TinyPerson.
- Varroa intentionally has only `varroa_full.png`, with no crop or zoom.

All annotations use the same red bbox color.

## Files

- `levirship_full.png`, `levirship_context_crop.png`, `levirship_object_zoom.png`
- `varroa_full.png`, `varroa_context_crop.png`, `varroa_object_zoom.png`
- `tinyperson_full.png`, `tinyperson_context_crop.png`, `tinyperson_object_zoom.png`
- `single_images_manifest.json`: exact source paths and selected boxes.

Example LaTeX usage:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.48\columnwidth]{figures/levirship_context_crop.png}
  \includegraphics[width=0.48\columnwidth]{figures/levirship_object_zoom.png}
  \caption{LEVIR-Ship context crop and standalone object zoom.}
  \label{fig:sample_levirship}
\end{figure}
```
