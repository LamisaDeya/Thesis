# Bangla Text Summarization from Video Content using Sequential Method of Abstractive and Extractive Summarization

## Supervisor:
**Abdul Aziz**, Assistant Professor, Department of CSE, Khulna University of Engineering & Technology (KUET)

## Overview
This project presents a novel approach to summarizing Bangla video content by using a sequential combination of abstractive and extractive summarization techniques. Given the increasing volume of Bangla video content and the scarcity of Bangla-specific summarization tools, this work aims to address the need for efficient Bangla video summarization.

## Objectives
- Develop an efficient Bangla video summarization system.
- Create a benchmark dataset for Bangla video summarization.
- Generate concise summaries of lengthy videos.
- Enable users to quickly identify relevant video content.
- Enhance existing Bangla summarization approaches.

## Key Contributions

### 1. Dataset Collection
- Collection of Bangla articles and summaries, incorporating **19,096 data points** for abstractive summarization and additional manually collected summaries.

### 2. Model Design
- **Abstractive Summarization**: Using **Bangla-T5** for capturing contextual information.
- **Extractive Summarization**: Utilizing **Bangla-BERTSum** for extracting key sentences.
- **Proposed Methodology**: Combining abstractive summarization followed by extractive summarization for generating coherent and concise summaries.

## Methodology

### 1. Preprocessing
- **Audio Extraction**: Using MoviePy for extracting audio.
- **Speech Recognition**: Google’s Speech Recognition API.
- **Text Cleaning**: DeepPunct model for punctuation and stop word removal.

### 2. Model Design
- **Bangla-T5**:
  - Sequence-to-sequence model for abstractive summarization.
  - Trained on the Bangla2B+ dataset.
- **Bangla-BERTSum**:
  - Extractive summarization model based on Bangla-BERT.
  - Capable of handling multiple sentences and generating summaries using attention scoring.

### 3. Sequential Summarization
- **Abstractive Summarization**: Generates contextually rich summaries.
- **Extractive Summarization**: Filters key sentences for brevity and relevance.

## Results

### Quantitative Results
A table of metrics demonstrating the effectiveness of both models in summarizing Bangla video content.

### Qualitative Results
Example summaries showcasing the coherence and accuracy of the generated summaries.

## Conclusion
- Developed a dual-model approach for Bangla video summarization using both **Bangla-T5** and **Bangla-BERTSum**.
- Created a benchmark dataset and fine-tuned models for Bangla-specific summarization tasks.
- Demonstrated the effectiveness of a sequential abstractive-extractive approach.

## References
- **Talukder et al. (2019)**: Bengali abstractive text summarization using RNNs.
- **Chowdhury et al. (2021)**: Unsupervised abstractive summarization of Bengali documents.
- **Bhattacharjee et al. (2023)**: BanglaNLG and BanglaT5 benchmarks for Bangla NLG tasks.
- **Kowsher et al. (2022)**: Transformer-based Bangla-BERT for language understanding.

## Acknowledgments
This project was conducted as part of the undergraduate thesis under the supervision of **Abdul Aziz**, Department of CSE, KUET.

## Contact
For further details, please contact:
- **Farhatun Shama**
- **Lamisa Bintee Mizan Deya**

---

This README provides a structured overview of the Bangla Text Summarization project, detailing objectives, methodology, results, and references.
