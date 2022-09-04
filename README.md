# User_similarity_netflix_challenge
Finding user similarities in the netflix challenge dataset by using Jaccard Similarity, Cosine Similarity and Discrete Cosine Similarity.<br/>
This repository is the resolution for the second assignment of the course `Advances in Data Mining` taught at Leiden University and is based on chapter 3 of the `Mining Massive Datasets` book by Jure Leskovec, Anand Rajaraman and Jeff Ullman.

## Installation
A `Python 3` environment with packages specified in the `requirements.txt` file is required to run the user similarity script.


## Usage
Running the script will create a text file, named as the similarity metric used, that will contain the pairs of users with a similarity equal or higher than the threshold defined in the parameters.

To run the script use the `main.py` script together with the following command line arguments:

`-d`: directory of the netflix challenge dataset.<br/>
`-s`: seed (to reproduce experiments).<br/>
`-m`: similarity measure. Can be either 'js' (Jaccard Similarity), 'cs' (Cosine similarity) or 'dcs' (DIscrete Cosine Similarity).<br/>
`-b`: number of bands.<br/>
`-p`: number of permutations.<br/>
`-t`: threshold to consider as similarity. (0. to 1.).<br/>
`-v`: verbose parameter, defines the debug print intensity. Should be 0, 1 or 2.