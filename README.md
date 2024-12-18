[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)

# Dynamic Protein Localization
This repository contains code and data for the second ML project on dynamic protein localization by Leonardo Bocchieri, Emilia Farina, RIccardo Rota. It covers all the pipeline followed for solving our problem, from datacleaning to postprocessing and interpretation of the results.

<details open><summary><b>Table of contents</b></summary>

- [Problem Descprition](#problem-description)
- [Usage](#usage)
  - [Quick Start](#quickstart)
  - [Data Cleaning](#data-cleaning)
  - [Run Models](#run-models)
- [Models](#models)
  - [ESM Models](#esm-models)
  - [Static Models](#static-models)
  - [Dynamic Models](#dynamic-models)
- [Citations](#citations)
- [License](#license)
</details>

## Problem Description <a name="problem-desciption"></a>

DESCRIZIONE RINO GATTUSO BLABLA
For a much deeper insite, refer to the report of the project. ADD REPORT LINK

## Usage <a name="usage"></a>

### Quick start <a name="quickstart"></a>

For setting up all the needed packages, you need to run on your bash file:
```
pip install -r requirements.txt
```
Adding if necessary the full path of the `requirements.txt` on your computer.

### Data Cleaning <a name="data-cleaning"></a>

You can create dataset running `data_cleaning.ipynb`

### Run Models <a name="run-models"></a>




An easy way to get started is to load ESM or ESMFold through the [HuggingFace transformers library](https://huggingface.co/docs/transformers/model_doc/esm),
which has simplified the ESMFold dependencies and provides a standardized API an

## Main models you should use <a name="main-models"></a>

| Shorthand | `esm.pretrained.`           | Dataset | Description  |
|-----------|-----------------------------|---------|--------------|
| ESM-2    | `esm2_t36_3B_UR50D()` `esm2_t48_15B_UR50D()`       | UR50 (sample UR90)  | SOTA general-purpose protein language model. Can be used to predict structure, function and other protein properties directly from individual sequences. Released with [Lin et al. 2022](https://www.science.org/doi/abs/10.1126/science.ade2574) (Aug 2022 update). |
| ESMFold   | `esmfold_v1()`         | PDB + UR50 | End-to-end single sequence 3D structure predictor (Nov 2022 update). |
| ESM-MSA-1b| `esm_msa1b_t12_100M_UR50S()` |  UR50 + MSA  | MSA Transformer language model. Can be used to extract embeddings from an MSA. Enables SOTA inference of structure. Released with [Rao et al. 2021](https://www.biorxiv.org/content/10.1101/2021.02.12.430858v2) (ICML'21 version, June 2021).  |
| ESM-1v    | `esm1v_t33_650M_UR90S_1()` ... `esm1v_t33_650M_UR90S_5()`| UR90  | Language model specialized for prediction of variant effects. Enables SOTA zero-shot prediction of the functional effects of sequence variations. Same architecture as ESM-1b, but trained on UniRef90. Released with [Meier et al. 2021](https://doi.org/10.1101/2021.07.09.450648). |
| ESM-IF1  | `esm_if1_gvp4_t16_142M_UR50()` | CATH + UR50 | Inverse folding model. Can be used to design sequences for given structures, or to predict functional effects of sequence variation for given structures. Enables SOTA fixed backbone sequence design. Released with [Hsu et al. 2022](https://doi.org/10.1101/2022.04.10.487779). |

For a complete list of available models, with details and release notes, see [Pre-trained Models](#available-models).







