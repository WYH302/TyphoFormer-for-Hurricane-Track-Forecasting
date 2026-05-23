# Open-source packaging notes

This directory is the lightweight code release candidate for TyphoFormerPlus.

Local experiment artifacts have been moved out of this folder to:

```text
../local_artifacts/TyphoFormerPlus
```

The moved artifacts include checkpoint directories, generated data directories, embedding chunks, logs, Python caches, and large local assets. They are intentionally ignored by `.gitignore` so they are not accidentally committed.

Before publishing, keep only code, README files, small examples, and result summaries in this folder. Put large datasets, trained weights, and full experiment outputs in a release artifact, external storage, or a separate download note.
