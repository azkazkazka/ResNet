# ResNet 

This repository contains a ResNet-based architecture implemented based on the one referred in [Alzantot et al.](https://www.researchgate.net/publication/334161923_Deep_Residual_Neural_Networks_for_Audio_Spoofing_Detection) for spoof detection in audio datasets

## Setup Instructions

### 1. Install Dependencies

Before running any scripts, ensure you have the correct environment set up. 

You can install the required dependencies by running the `conda.sh` script:

```bash
bash scripts/conda.sh
```

This script will install the environment specified in the `env.yml` file.

### 2. Training the Model

To train the model, you can use the following scripts based on the scenario you wish to run:

- **Logical Access (LA) Scenario**:  
  ```bash
  bash scripts/train_LA.sh
  ```
  
- **Physical Access (PA) Scenario**:  
  ```bash
  bash scripts/train_PA.sh
  ```

Both scripts handle the training process and save the trained models in the `models` folder.

### 3. Evaluating the Model

Once training is complete, you can evaluate the model using the respective evaluation scripts:

- **Logical Access (LA) Evaluation**:  
  ```bash
  bash scripts/eval_LA.sh
  ```

- **Physical Access (PA) Evaluation**:  
  ```bash
  bash scripts/eval_PA.sh
  ```

The predictions will be saved inside the `predictions` folder

## Generating Metric Scores

After running the evaluation, you can calculate the metric scores from the predictions generated before using the following process:

1. Navigate to the `utils/calculate_metrics` folder:

   ```bash
   cd utils/calculcate_metrics
   ```

2. Run the `calculate_eval.py` script to generate the scores:

   ```bash
   python calculate_eval.py --scores <path_to_scores_file> --scenario <LA/PA> --output_filename <optional_output_name>
   ```

   - Replace `<path_to_scores_file>` with the path to the scores file generated during evaluation.
   - Specify the scenario using `<LA/PA>`.
   - Optionally, provide an output filename. If not provided, the result will be named as `result_<timestamp>.txt`.

## Folder Structure

```
|- ResNet
|  |
|  |- models/              # Contains trained models
|  |- predictions/         # Contains predictions
|  |- scripts/
|  |    |- conda.sh        # Sets up the environment
|  |    |- train_LA.sh     # Script to train on LA scenario
|  |    |- train_PA.sh     # Script to train on PA scenario
|  |    |- eval_LA.sh      # Script to evaluate on LA scenario
|  |    |- eval_PA.sh      # Script to evaluate on PA scenario
|  |- utils/
|  |    |- calculate_metrics/
|  |        |- calculate_eval.py   # Script to generate evaluation scores
|  |        |- calculate_metrics.py
|  |        |- a_dcf.py            # Helper file for calculating metrics
|  |        |- results/            # Stores metric results
|  |- main.py            # Main script for training and evaluation
|  |- model.py           # Contains the ResNet model definition
|  |- env.yml            # Conda environment file
```

## Additional Notes

- **Customizing Training**: You can adjust parameters such as the number of epochs, batch size, or learning rate directly inside the training scripts (`train_LA.sh`, `train_PA.sh`).
