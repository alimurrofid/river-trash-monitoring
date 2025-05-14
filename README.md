# River Trash Monitoring

## Dataset

### 1. Download the Dataset

Download the dataset from [Dataset]().

### 2. Extract the Dataset

Unpack the dataset into a folder named `dataset`, with the following structure:

```
dataset
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
train: F:\Kuliah\Skripsi\Code\river-trash-monitoring\dataset\train
test: F:\Kuliah\Skripsi\Code\river-trash-monitoring\dataset\test
val: F:\Kuliah\Skripsi\Code\river-trash-monitoring\dataset\valid

nc: 2
names: ["nonplastic", "plastic"]
```

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

## Counting Objects
Use the following command to count objects in an image:

```bash
python counting.py
```