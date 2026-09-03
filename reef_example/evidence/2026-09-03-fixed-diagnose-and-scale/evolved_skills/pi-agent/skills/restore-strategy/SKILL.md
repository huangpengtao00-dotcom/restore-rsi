# restore-strategy

Only these commands may touch images: `restore diagnose IN` and `restore run TOOL IN OUT`. Never use Python/PIL/numpy/ImageMagick or any image viewer/editor/copy command. Never open, copy, or inspect the clean original, the task manifest, or the degradation code. Never write or predict a PSNR/score. The judge uses the last `restore run` output, so make the last restored image be the final candidate and name its exact output path on the final line.

## Read the diagnosis

The diagnose JSON gives four estimates in 0..1: haze, low_light, noise, blur. Use approximate bands:

- less than 0.20: negligible
- 0.20 to 0.40: mild
- 0.40 to 0.60: strong
- greater than 0.60: severe

Do not try to drive every value to zero. Stop when all remaining values are roughly below 0.25, or when the next operation would address a value below 0.25, or when one more pass of the same tool would risk overprocessing. Run `restore diagnose` on intermediates before deciding the next step.

## Safe order

Apply noise removal before brightening or dehaze; brightening amplifies noise and dehaze can darken. Apply low-light correction before dehaze in mixed dark+hazy cases. Apply dehaze before local contrast enhancement, not after. Use sharpen only last, and only if blur is the remaining defect.

## Low-light + noise

If noise is strong (about 0.4 or higher) and the image also needs lightening:

1. `restore run denoise_bilateral IN p1.png`
2. `restore run lowlight_gamma p1.png p2.png`
3. `restore diagnose p2.png`

If the brightening made noise strong again, use one more `denoise_bilateral` pass. If `low_light` is still high after one gamma pass and noise is not high, use `lowlight_clahe` as the second contrast pass; do not stack two gamma passes. Maximum useful bilateral passes is two. Do not use `denoise_median` unless you are sure the noise is salt-and-pepper; with only diagnosis numbers, ordinary noise is better handled by bilateral, because median removes texture and will reduce PSNR.

## Haze + low-light

If both haze and low_light are significant (about 0.3 or higher):

1. Denoise first if noise is also high.
2. `restore run lowlight_gamma IN p1.png` so the dark image is not dehazed first. Dehazing a dark image first makes it darker.
3. `restore run dehaze_dcp p1.png p2.png`
4. `restore diagnose p2.png`

If `dehaze_dcp` made the image dark again, repair it once with `lowlight_gamma`; if only local contrast is flat and noise is low, use `lowlight_clahe` instead. If haze is still strong after the image is brightened, one second `dehaze_dcp` pass is allowed, but no third dehaze pass. Over-dehazing causes color distortion and lowers PSNR.

## Haze only

If haze is significant but low_light is low:

1. `restore run dehaze_dcp IN p1.png`
2. `restore diagnose p1.png`
3. If dehaze darkened it, `lowlight_gamma p1.png p2.png`. If haze remains strong and the image is not dark, one second `dehaze_dcp` can be run, then re-diagnose.

## Blur and sharpening

Do not sharpen after dehaze or lighten unless blur is the dominant remaining problem. Use `sharpen_unsharp` once and only at the end when blur is about 0.4 or higher, and noise, low_light, and haze are all below about 0.25. Sharpening amplifies noise and halo artifacts; over-sharpening is a common way to fail PSNR. If noise is not low, skip sharpening entirely.

## Finishing

Keep intermediate names simple: `p1.png`, `p2.png`, ..., and use `final.png` for the final candidate in the current directory. Do not overwrite the input.

After the last operation, run `restore diagnose final.png` only if it helps confirm no target value is still strong. If an operation made the image better in one value but much worse in another, stop and do not compensate with extra sharpening/denoising.

The very last line of your reply must be the exact path of the final image, alone on the last line. If your final `restore run` command was `restore run TOOL p2.png final.png`, the last line is exactly `final.png`. If you use `./final.png` in the command, repeat `./final.png` on the last line.
