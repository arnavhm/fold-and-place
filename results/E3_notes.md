# E3 - Marker-based size accuracy (synthetic)

144 synthetic photos (6 garments, camera tilt 0-25 deg, A4-sheet corner noise 0.5
and 1.5 px). Mean absolute error: width 2.34 cm, length 1.9 cm;
95th percentile 6.66 cm. See E3_summary.csv per garment.

For comparison, published garment-measurement systems report about 1 cm average error on real
photos (Paulauskaite-Taraseviciene et al., Applied Sciences 2022). Our synthetic test has a perfect
mask, so real-photo error will be higher: segmentation is the dominant error source, not geometry.
Stock photos carry no marker; their sizes come from size charts / fabric standards and are labelled so.
