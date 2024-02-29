# Amazon-Product-Recommender
The growth of reviews on Amazon has provided a wealth of data for understanding customer preferences. This study focuses on building a recommender system for digital music tracks using a dataset of 200,000 reviews. We explore traditional models and propose a deep neural network (DNN) architecture for predicting review ratings. Our data preprocessing includes feature selection based on statistical analysis and feature engineering to extract valuable information from attributes such as review text, summary, category, and review times.

The DNN architecture comprises a text feature extractor using an Embedding layer and LSTM layer, along with other features like category, reviewer ID, and item ID. Training the DNN with Adam optimization and MSE loss over 100 epochs, we achieve promising results compared to traditional models like multinomial naive Bayes and logistic regression, with an MSE of 0.51-0.53.


Challenges faced included text preprocessing and model selection due to the limited dataset size. We overcame these challenges by following a systematic approach and experimenting with different model architectures. Our results demonstrate the effectiveness of DNNs in recommending products based on customer reviews.


See the report here: https://arxiv.org/abs/2102.04238 for more details


cite:
@misc{rezaei2021amazon,
      title={Amazon Product Recommender System}, 
      author={Mohammad R. Rezaei},
      year={2021},
      eprint={2102.04238},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
