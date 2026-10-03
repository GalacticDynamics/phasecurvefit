# Citation

If you use **phasecurvefit** in published work, please cite the package itself,
together with the paper behind each component you used. What to cite depends on
which parts of the library ran, so the table below is keyed on the names you
actually call.

## Citing the package

Cite the software via its Zenodo DOI,
[10.5281/zenodo.18714340](https://doi.org/10.5281/zenodo.18714340). Machine-readable
metadata, including the component papers below, is in
[`CITATION.cff`](https://github.com/GalacticDynamics/phasecurvefit/blob/main/CITATION.cff);
GitHub's "Cite this repository" button reads it and produces BibTeX or APA.

## What else to cite

The default pipeline, `pcf.order(pos, vel)`, is an MST backbone refined by a SOM, so
the ordering needs the SOM paper and no other:

| If you use … | also cite |
| --- | --- |
| the default pipeline: `pcf.order(pos, vel)` with no orderer, {func}`~phasecurvefit.orderers.default_pipeline` | Starkman et al. (2023) |
| {class}`~phasecurvefit.orderers.SOMOrderer`, the {mod}`phasecurvefit.som` module | Starkman et al. (2023) |
| **momentum-weighted ordering**: {class}`~phasecurvefit.orderers.LocalFlowOrderer` (and the deprecated `pcf.walk_local_flow`), the default {class}`~phasecurvefit.metrics.AlignedMomentumDistanceMetric` | Nibauer et al. (2022) |
| the **autoencoder**: {class}`~phasecurvefit.nn.PathAutoencoder`, {func}`~phasecurvefit.nn.train_autoencoder`, {func}`~phasecurvefit.fit_track` | Nibauer et al. (2022) |
| **mixture-model membership** for outlier rejection: {class}`~phasecurvefit.nn.MixtureMembershipConfig` (see {doc}`guides/outliers`) | Hogg, Bovy & Lang (2010) |
| {class}`~phasecurvefit.orderers.MSTOrderer` alone | nothing beyond the package |

These add up. {func}`~phasecurvefit.fit_track` runs the default pipeline and then
trains the autoencoder, so it cites **both** Starkman et al. (2023) and Nibauer et
al. (2022). A local-flow walk followed by the autoencoder cites Nibauer et al.
(2022) alone.

Nibauer et al. (2022) is the source of the local-flow walk and the autoencoder,
not of the default ordering: cite it when you use one of those, not merely because
you called `pcf.order`.

## BibTeX

### Nibauer et al. (2022): momentum-weighted ordering and the autoencoder

```bibtex
@article{nibauer2022charting,
  title={Charting Galactic Accelerations with Stellar Streams and Machine Learning},
  author={Nibauer, Jacob and Belokurov, Vasily and Cranmer, Miles and
          Goodman, Jeremy and Ho, Shirley},
  journal={The Astrophysical Journal},
  volume={940},
  pages={22},
  year={2022},
  doi={10.3847/1538-4357/ac93ee},
  eprint={2205.11767},
  archivePrefix={arXiv},
  primaryClass={astro-ph.GA}
}
```

### Starkman et al. (2023): the SOM stage, and so the default pipeline

The method implemented is in §2.2 and Appendix A of the paper; see the
[SOM guide](guides/som) for where this implementation deviates.

```bibtex
@article{starkman2023fasttrack,
  title={On the Fast Track: Rapid construction of stellar stream paths},
  author={Starkman, Nathaniel and Bovy, Jo and Webb, Jeremy J. and
          Calvetti, Daniela and Somersalo, Erkki},
  journal={Monthly Notices of the Royal Astronomical Society},
  volume={522},
  number={4},
  pages={5022--5036},
  year={2023},
  doi={10.1093/mnras/stad1166},
  eprint={2212.00949},
  archivePrefix={arXiv},
  primaryClass={astro-ph.GA}
}
```

### Hogg, Bovy & Lang (2010): mixture-model membership

§3 ("Pruning outliers") is the model implemented.

```bibtex
@article{hogg2010data,
  title={Data analysis recipes: Fitting a model to data},
  author={Hogg, David W. and Bovy, Jo and Lang, Dustin},
  journal={arXiv preprint arXiv:1008.4686},
  year={2010},
  eprint={1008.4686},
  archivePrefix={arXiv},
  primaryClass={astro-ph.IM}
}
```
