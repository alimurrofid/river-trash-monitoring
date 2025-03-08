# River Trash Monitoring

## Dataset

### 1. Download the Dataset

Download the dataset from [Roboflow](https://universe.roboflow.com/amril-syah-lubis/data-plastic).

### 2. Extract the Dataset

Unpack the dataset into a folder named `dataset2`, with the following structure:

```
dataset2
├── test
│   ├── images
│   ├── labels
├── train
│   ├── images
│   ├── labels
├── valid
│   ├── images
│   ├── labels
├── data.yaml
```

### 3. `data.yaml` File

The `data.yaml` file contains dataset configuration, including paths and class names:

```yaml
train: F:\Kuliah\Skripsi\Code\river-trash-monitoring\dataset2\train
val: F:\Kuliah\Skripsi\Code\river-trash-monitoring\dataset2\valid
test: F:\Kuliah\Skripsi\Code\river-trash-monitoring\dataset2\test

nc: 1
names: ["plastic"]
```

## YOLO Model

The `yolo_model` folder contains the YOLO models used for training.

---

## Installation & Setup

### 1. Create a Conda Environment

Ensure you have Conda installed. To create the environment, run:

```bash
conda env create -f env.yaml
```

> Note:
> This environment uses Python 3.10.16 and was developed on Windows 11.

### 2. Activate the Environment

```bash
conda activate river-trash-monitoring
```

---

## Train the Model

Run the following command to train the model:

```bash
python train.py
```

---

## Predict Objects in an Image

Use the following command to perform object detection:

```bash
python predict.py
```
