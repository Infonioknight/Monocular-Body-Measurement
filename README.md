# Monocular Body Measurement under Weak Calibration

Empirical evaluation of three monocular approaches for estimating human body measurements
(chest, waist) under weak calibration and constrained capture conditions.

## Methods
- Method 1 (baseline.py): Anthropometric Calibration Baseline
- Method 2 (regression.py): Pose-Based Regression for Circumferential Estimation
- Method 3 (object-calibrated.py): Object Calibrated, Silhouette-Assisted Estimation

## Assumptions & Constraints
- Frontal camera
- Single subject
- Minimal pose variation
- Clothing effects discussed explicitly

## Repository Structure
methods/    # runnable scripts for each method
models/     # trained regression models
notebooks/  # training notebook for the Regression models + EDA

## Reproducibility
```
pip install -r requirements.txt
``` 
(Common to install requirements for all the approaches)

### Method 1: (Anthropometric Calibration Baseline)
This method requires a video, where the user stands approximately frontal to a camera, which has a clear and non-skewed view of the person from head to toe. Ideally, ensure you are stationary for a few seconds for best results.

```
python methods/baseline.py 
    --video <insert video path> 
    --shoulder_width <calibration shoulder measurement>
    --output (Optional) <path to save JSON of results>
```
The method requires a calibration shoulder measurement, which you as the user must accurately measure using a tape measure.

**Note:** The measurement is your bone-to-bone measurement (Acromion)

### Method 2: (Pose-Based Regression for Circumferential Estimation)
This method shares the same shoulder-width calibration requirement as Method 1, but replaces geometric estimation with a pose-based regression model.

It also uses the exact same capture criterion as the previous method.

```
python methods/regression.py 
    --video <path to front view video>
    --shoulder_width <same as the previous method>
    --gender <string in quotes Eg: 'Male' or 'M'>
    --chest_model (Optional) <path to chest predictor regression model (if you choose to use one you train by yourself with the exact same features as the one used in the notebook, else tweak script accordingly)>
    --waist_model (Optional) <waist predictor model path>
    --output (Optional) <same as previous method>
```

### Method 3: (Object Calibrated, Silhouette Assisted Method)

This method requires the user to have two videos.
    - First being a front-view with the user holding a card (credit card or another card with the EXACT same dimensions) flat on their chest. Also ensure the view is clear from head to toe, and ensure you are stationary for best measurements.

    - Second being a side view with the user clearly standing with their side facing the camera, being visible head to toe.

**Note:** Ensure there is sufficient colour contrast between the your clothing and the card to ensure reliable detection.

```
python methods/object-calibrated.py 
--front_video <Path to front view video>
--side_video <Path to side profile view video>
--height <height in centimeters>
--output (Optional) <Save path>
```

This method requires the user to add their height in **centimeters** as a calibration measurement.

## Data
Training data for the regression models is not included in this repository. The training data used was the [BodyM Dataset](https://registry.opendata.aws/bodym/).

The `measurements.csv` (for all numeric measurements) and `metadata.csv` (for gender bias in the regression models) were used.

## Outputs:
All methods output JSON results with the measurements in **inches**
