# Read Me
This repository is the implementation of the NAACL 2021 accepted paper " Time-Stamped Language Model: Teaching Language Models toUnderstand The Flow of Events".

To use this implmentation you need to install the following packages:

> pytorch 1.7.0
> 
> transformers 3.5.1
> 
> tqdm
> 
> spacy
> 
> difflib

## Getting Started
### Data Preprocessing
You can either generate the preprocessed input from Propara or use the already generated jsons in the `data` folder.
To generate the preprocessed jsons from the original data in Propara, use the following command.

> python prepare/run_prepare.py

The resulting jsons will be stored in the `data` folder and replace the existing ones.

### Downloading the pretrained models and data
Please follow this link to download the pretrained model for Propara and the Data for the NPN Cooking recipes.
[Google Drive Download Link](https://drive.google.com/drive/folders/1gw0E2R81NgcCZwsahmAH--5rJp56sSVZ?usp=sharing)

Unzip the npn data and put the json under the `cooking_dataset/`.
Put the `best_model` under the `saves/` folder.
### Testing with the pretrained model
First download the `best_model` and place the model in the `saves` folder under the root.
To test with the pretrained model please use the following command and answer `y` when the question `test only (y/n)` prompts to the terminal.

> python main.py

### Training the model
To train the model use the previous command and answer `n` when the question is prompted.

### Table of Results ( Paper)
 
|                                        |                 |                  |                |                 |                 |                |                |                |
|----------------------------------------|-----------------|------------------|----------------|-----------------|-----------------|----------------|----------------|----------------|
|                                        | Sentence-level  |                  |                |                 |                 | Document-level |                |                |
| Model                                  | C1              | C2               | C3             | Macro-Avg       | Micro-Avg       | P              | R              | F1             |
| ProLocal | 62.7            | 30.5             | 10.4           | 34.5            | 34.0            | **77.4**    | 22.9           | 35.3           |
| ProGlobal  | 63.0            | 36.4             | 35.9           | 45.1            | 45.4            | 46.7           | 52.4           | 49.4           |
| EntNet    | 51.6            | 18.8             | 7.8            | 26.1            | 26.0            | 50.2           | 33.5           | 40.2           |
| QRN            | 52.4            | 15.5             | 10.9           | 26.3            | 26.5            | 55.5           | 31.3           | 40.0           |
| KG-MRC       | 62.9            | 40.0             | 38.2           | 47.0            | 46.6            | 64.5           | 50.7           | 56.8           |
| NCET        | 73.7            | 47.1             | 41.0           | 53.9            | 54.0            | 67.1           | 58.5           | 62.5           |
| XPAD      | \-              | \-               | \-             | \-              | \-              | 70.5           | 45.3           | 55.2           |
| ProStruct | \-              | \-               | \-             | \-              | \-              | 74.3           | 43.0           | 54.5           |
| DYNAPRO   | 72.4            | 49.3             | **44.5** | 55.4            | 55.5            | 75.2           | 58.0           | 65.5           |
| TSLM\~(Our Model)                      | **78.81** | **56.798** | 40.9           | **58.83** | **58.37** | 68.4 | **68.9** | **68.6** |


### Table of Results ( Reimplemented)
|                                        |                 |                  |                |               |                |                |                |                |
|----------------------------------------|----------------|-----------------|-----------------|----------------|----------------|----------------|----------------|----------------|
|                                        | Sentence-level  |              |                 | Document-level |                |                |
| Model                                  | C1              | C2               | C3     | Macro-Avg     | Micro-Avg     | P              | R              | F1             |
| TSLM  | **79.37** | **53.97** | 40.97       | 58.81        | 58.25           | 67.9 | 67.2 | 67.6 |
| TSLM + Constraints | - | - | -    | -  | -        | **68.3** | **68.8** | **68.5** |

### Running the NPN model
To run the NPN model run the `npn_main.py` instead of the `main.py`. 
The data we are using as input for NPN is also stored in `cooking_dataset/npn_data.json`.

## Citation
Please use the following to cite our paper.

```
@misc{faghihi2021timestamped,
      title={Time-Stamped Language Model: Teaching Language Models to Understand the Flow of Events}, 
      author={Hossein Rajaby Faghihi and Parisa Kordjamshidi},
      year={2021},
      eprint={2104.07635},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
