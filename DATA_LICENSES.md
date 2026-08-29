# Public data and attribution

The Apache License 2.0 in this repository applies only to OpenTrace ML source code. It
does not relicense external datasets, map data, imagery, annotations, or model
weights. No external dataset is bundled in the package.

## RDD2022

- Purpose: road-crack and pothole object detection.
- Source: <https://figshare.com/articles/dataset/21431547>
- Licence shown by the authoritative Figshare record: CC BY 4.0.
- Attribution: Arya et al., *RDD2022: A multi-national image dataset for
  automatic Road Damage Detection*.

## UCI Metro Interstate Traffic Volume

- Purpose: traffic-volume forecasting baseline.
- Source: <https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume>
- Licence: CC BY 4.0.
- Attribution: Hogue, J. (2019), UCI Machine Learning Repository,
  DOI 10.24432/C5X60B.

## OpenStreetMap

- Purpose: road-network geometry and routing features.
- Source and terms: <https://www.openstreetmap.org/copyright>
- Licence: Open Database Licence (ODbL) 1.0.
- Required attribution: © OpenStreetMap contributors.

OpenStreetMap attribution should be visible by default in applications using
the library. Publicly used or distributed derivative databases may carry ODbL
share-alike obligations. Consult the OSM Foundation guidance before combining
OSM-derived geometry into a new public road database.

## Model weights

OpenTrace ML does not currently distribute model weights. Future weights must
have their own model card and licence after checking the training data,
architecture implementation, pretrained checkpoint, and export format. The
Apache-2.0 licence for this repository must not be assumed to cover weights.

## Redistribution policy

Download scripts and adapters may refer to external sources, but dataset files
must not be committed to this repository. Contributors should add source,
licence, version, attribution, and checksum information for every new dataset.
