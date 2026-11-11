# Documentation

Read in this order:

1. [GLOSSARY.md](GLOSSARY.md): plain-language definitions (SAR, GCOV, GUNW, backscatter,
   coherence, gamma-0, ENL, zafra) for readers new to radar remote sensing.
2. [METHODS.md](METHODS.md): the pipeline step by step: data and sites, field polygons,
   cane gate, radar features, harvest detector, baseline rule, alerts, summaries, outputs
   and web rasters. It includes the exercise and script map.
3. [VALIDATION.md](VALIDATION.md): what each step was checked against, with the numbers
   from the committed results, and the known limitations.
4. [DATA_SOURCES.md](DATA_SOURCES.md): every dataset, its resolution, dates, access and
   licence, and the data licence of this repository.
5. [AI_COMPONENTS.md](AI_COMPONENTS.md): the trained models (cane gate, harvest detector),
   the LLM summary generator with its verifier, and the rejected SAM segmentation.
6. [REFERENCES.md](REFERENCES.md): the numbered references cited as [n].

Figures used in these documents are in [figures/](figures/); the full generated reports
are in [`../results/`](../results/), one folder per exercise.
