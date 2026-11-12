# E3 - Marker-based size accuracy (synthetic)

144 synthetic photos (6 garments, camera tilt 0-25 deg, A4-sheet corner noise 0.5
and 1.5 px). Mean absolute error: width 2.34 cm, length 1.9 cm;
95th percentile 6.53 cm. See E3_summary.csv per garment.

## The whole chain on realistic photos

54 realistic synthetic photos (textured bedsheet, prints, creases, shadows,
uneven light, any rotation, tilt up to 25 deg, JPEG), the A4 sheet **found automatically**, the garment
segmented by the plain-background segmenter and turned upright: sheet found in
100.0% of photos; mean absolute error width 1.46 cm,
length 1.8 cm (median 1.0 /
1.0 cm); 83.3% within 3 cm on both. Without a
sheet, scaled to the correct size label, width error 2.53 cm
(5.0%; this mode cannot undo camera tilt, so shoot from straight above).
Segmenting a plain-background photo is not the bottleneck here: as in the perfect-mask test, the error is
driven by how precisely the sheet's corners are located relative to the sheet's size in the photo (a small,
distant sheet magnifies corner error). The synthetic garments have wrinkled edges, which adds a little.
E8 repeats this on real photos.

For comparison, published garment-measurement systems report about 1 cm average error on real
photos (Paulauskaite-Taraseviciene et al., Applied Sciences 2022). Our synthetic test has a perfect
mask, so real-photo error will be higher: segmentation is the dominant error source, not geometry.
Stock photos carry no marker; their sizes come from size charts / fabric standards and are labelled so.
