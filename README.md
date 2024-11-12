# financial fraud detection model

This repository contains a model for predicting financial statement fraud in the Chinese stock market, developed as part of my internship at China's security.

## Overview

The model identifies fraudulent financial statements by analyzing key financial indicators. It is designed for application within the Chinese stock market context and uses data sourced from the CSMAR and Wind databases.

## Setup and Usage

### 1. Install Dependencies

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Ensure that your environment supports PyTorch with GPU acceleration, as the model relies on it for faster computation.

### 3. Data Requirement

Due to confidentiality constraints, the /input folder is empty and should contain a file named "财务违规数据表" (Financial Violation Data Table) obtained from the [CSMAR](https://data.csmar.com/) database. You will also need access to the Wind database, with [paths](./main.py#L27) updated in the source code.

### 4. Running the Model

Control the model's behavior through main.py, where you can specify different stages of the processing pipeline. For more details on usage, functions, and indicator calculations, refer to the documentation provided in [程序说明文档](./程序说明（使用方法，函数文档，指标计算公式）.docx).


