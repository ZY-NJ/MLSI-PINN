# Example observations

Two separate 2023-08-14 FISS Ca II rasters are supplied as 64 × 64 crops. Each contains 4,096 spectra with all 502 wavelengths retained. Crops use zero-based rows `[93:157)` and columns `[43:107)` of the source arrays, preserving their spatial order.

- `train.fits.gz`: 17:59:02.489, corrected intensities and conventional MLSI targets.
- `inference.fits.gz`: 17:25:19.159, corrected intensities only.

The corrections and normalization were performed before these crops were created. No additional smoothing, normalization, resampling or spatial filtering was applied to the packaged spectra. The separate vₚ estimator smooths profiles internally when fitting its reference absorption line.

The corrected source files did not contain wavelength calibration cards. The example preserves the wavelength reconstruction used by the retained loader:

```python
import torch
pixel = torch.arange(502, dtype=torch.float32)
wavelength_A = 8542.09 + (pixel + 1.0 - 225.358535767) * 0.0256741359106
```

The resulting float32 vector is stored in the `WAVELENGTH` extension. It runs from approximately 8536.3301 to 8549.1924 Å. This construction is specific to these retained files and must not be assumed for an unrelated observation.

`REFERENCE` has shape `(64, 64, 13)` and stores scaled floating-point conventional MLSI parameters in the output order documented in the main README. It is present only in the training file. `manifest.json` records hashes of the source spectra, source parameter file and packaged examples. Original paths and server credentials are not included.

Observation source: FISS data supplied for the MLSI-PINN study. Instrument and archive information is available from [FISS](http://fiss.snu.ac.kr/). The observational data retain the rights of their original providers and are not covered by a new software license from this package. Cite the data provider and the associated study when using these examples.
