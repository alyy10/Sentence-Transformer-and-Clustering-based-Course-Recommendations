# Sentence-Transformer-and-Clustering-based-Course-Recommendations

This repository contains a project that leverages Sentence Transformer models for converting course descriptions into numerical vectors and then applies clustering techniques to group similar courses.

## Project Overview

In the rapidly expanding landscape of online education, MOOC platforms face the challenge of organizing and recommending courses effectively. With hundreds of new courses being added daily, and often only a course description available, it becomes difficult to identify underlying specializations and assign courses to them. This project addresses this challenge by:

1. **Transforming Course Descriptions:** Utilizing state-of-the-art Sentence Transformer models to convert textual course descriptions into high-dimensional numerical vectors (embeddings).
2. **Clustering Similar Courses:** Applying clustering algorithms to these embeddings to group courses with similar content, thereby identifying potential specializations.
3. **Determining Optimal Clusters:** Employing the Elbow method and Silhouette analysis to determine the optimal number of clusters, which directly translates to the optimal number of specializations.
4. **Assigning Specializations:** Assigning each course to a specific specialization based on its cluster membership.

This approach provides a data-driven method for structuring course catalogs and enhancing the user experience by offering tailored specialization recommendations.

## Features

- **Sentence Embedding:** Uses `sentence-transformers` to generate meaningful embeddings from course descriptions.
- **Text Preprocessing:** Includes steps for cleaning and normalizing text data, such as lowercasing, punctuation removal, number replacement, stemming, and lemmatization.
- **Optimal Cluster Determination:** Implements the Elbow method and Silhouette score to find the most appropriate number of clusters for the dataset.
- **Clustering Algorithms:** Demonstrates the application of K-Means clustering for grouping courses.
- **Visualization:** Generates various plots to visualize clustering results and aid in the interpretation of specializations.

## Getting Started

### Prerequisites

To run this notebook, you will need Python 3.7+ and the following libraries. These are installed within the notebook itself, but for a local setup, you can install them using `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn yellowbrick texthero pyclustering nltk sentence-transformers transformers safetensors
```

### Data

The project uses a `course-catalog.csv` file. This file is expected to be in the same directory as the Jupyter notebook. The notebook reads this CSV file to extract course information, primarily the `Description` column for analysis.

## Dependencies

The following Python libraries are used in this project:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `scikit-learn`: For machine learning algorithms, particularly clustering.
- `matplotlib`: For plotting and visualization.
- `seaborn`: For enhanced statistical data visualization.
- `yellowbrick`: For visualizing machine learning estimators, especially for the Elbow method.
- `texthero`: For simplified text preprocessing.
- `pyclustering`: For advanced clustering algorithms.
- `nltk`: For natural language processing tasks like stemming and lemmatization.
- `sentence-transformers`: For generating sentence embeddings.
- `transformers`: Underlying library for `sentence-transformers`.
- `safetensors`: A dependency for `transformers`.

## Methodology

### Data Loading and Initial Exploration

The notebook begins by loading the `course-catalog.csv` file into a pandas DataFrame. It then performs initial data cleaning, including:

- Sorting data by course `Name`.
- Removing duplicate entries based on `Description`.
- Selecting relevant columns: `Subject`, `Name`, and `Description`.
- Filtering for the top 20 most frequent subjects to focus the analysis on well-represented categories.

### Text Preprocessing

To prepare course descriptions for embedding, a series of text preprocessing steps are applied:

1. **Lowercasing:** All text is converted to lowercase to ensure consistency.
2. **Punctuation Removal:** Punctuation marks are removed as they typically do not contribute to the semantic meaning for clustering.
3. **Number Replacement:** All numerical digits are replaced with a placeholder token (`num`) to generalize numerical values.
4. **Stemming (NLTK PorterStemmer):** Words are reduced to their root form (e.g.,

accessing, accessed -> access) to group together different forms of a word.
5.  **Lemmatization (NLTK WordNetLemmatizer):** Words are reduced to their base form based on their intended meaning (e.g., was, were, am -> be), providing a more linguistically accurate normalization than stemming.

These steps ensure that the text data is clean and consistent, which is crucial for generating high-quality embeddings.

### Sentence Embedding with SBERT

After preprocessing, the course descriptions are transformed into numerical vectors using a pre-trained Sentence Transformer model, specifically `all-MiniLM-L6-v2`. This model is chosen for its balance of performance and efficiency, providing dense vector representations that capture the semantic meaning of sentences. The embeddings are crucial for enabling distance-based clustering.

```python
from sentence_transformers import SentenceTransformer
model_SBERT = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model_SBERT.encode(text_lem, convert_to_numpy=True, show_progress_bar=True, batch_size=100)
```

Generating these embeddings can be computationally intensive and may take several minutes, depending on the size of the dataset and the computational resources available.

### Determining the Optimal Number of Clusters

Identifying the optimal number of clusters (k) is a critical step in unsupervised learning. This project employs two popular methods:

1. **Elbow Method:** This method involves plotting the within-cluster sum of squares (WCSS) against the number of clusters. The 'elbow point' on the plot, where the rate of decrease in WCSS sharply changes, is considered an indicator of the optimal `k`.

   ```python
   from sklearn.cluster import KMeans
   from yellowbrick.cluster import KElbowVisualizer

   model = KMeans(random_state=0, n_init=10) # n_init is set to 10 to suppress future warnings
   visualizer = KElbowVisualizer(model, k=(2,15), timings=False)
   visualizer.fit(embeddings)
   visualizer.show()
   ```
2. **Silhouette Analysis:** The Silhouette score measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The score ranges from -1 to +1, where a higher value indicates better-defined clusters. The optimal `k` often corresponds to the highest Silhouette score.

   ```python
   from sklearn.metrics import silhouette_score

   # Example for calculating silhouette score for a given k
   kmeans_model = KMeans(n_clusters=k, random_state=0, n_init=10).fit(embeddings)
   score = silhouette_score(embeddings, kmeans_model.labels_)
   ```

Both methods are used in conjunction to provide a robust estimate for the ideal number of specializations.

### Clustering with K-Means

Once the optimal number of clusters is determined, the K-Means algorithm is applied to the sentence embeddings. K-Means partitions the `n` observations into `k` clusters, where each observation belongs to the cluster with the nearest mean (centroid), serving as a prototype of the cluster.

```python
from sklearn.cluster import KMeans

# Assuming optimal_k is determined from the previous step
kmeans_model = KMeans(n_clusters=optimal_k, random_state=0, n_init=10).fit(embeddings)
data['Cluster'] = kmeans_model.labels_
```

The cluster labels are then assigned back to the original DataFrame, allowing for the identification of courses belonging to each specialization.

### Visualization of Clusters

To better understand the distribution and characteristics of the identified clusters, various visualizations are generated:

- **2D Scatter Plot (PCA/t-SNE):** Although not explicitly shown in the provided snippet, dimensionality reduction techniques like PCA or t-SNE are typically used to visualize high-dimensional embeddings in 2D, allowing for a visual inspection of cluster separation.
- **Word Clouds:** Word clouds are generated for each cluster to highlight the most frequent and significant terms within course descriptions of that specialization. This provides an intuitive understanding of the thematic focus of each cluster.

  ```python
  from wordcloud import WordCloud
  import matplotlib.pyplot as plt

  # Example for generating a word cloud for a specific cluster
  text_cluster = " ".join(data[data['Cluster'] == cluster_id]['Description'])
  wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text_cluster)
  plt.figure()
  plt.imshow(wordcloud, interpolation="bilinear")
  plt.axis("off")
  plt.show()
  ```

These visualizations help in validating the clustering results and in interpreting the nature of the derived specializations.

## Results and Discussion

The notebook demonstrates an effective pipeline for identifying course specializations from unstructured text descriptions. The combination of Sentence Transformers for semantic embedding and clustering algorithms provides a robust method for organizing large course catalogs.

Key observations from the analysis:

- **Effectiveness of SBERT:** The `all-MiniLM-L6-v2` model successfully captures the semantic nuances of course descriptions, leading to meaningful clusters. Courses within the same cluster exhibit high thematic similarity, indicating that the embeddings effectively represent the content.
- **Optimal Cluster Number:** The Elbow method and Silhouette analysis provide converging evidence for an optimal number of clusters, suggesting a natural grouping within the course data. This number directly translates to the recommended number of specializations for the MOOC platform.
- **Interpretable Specializations:** The word clouds generated for each cluster clearly reveal the dominant themes and keywords associated with each specialization. For instance, a cluster might show terms like 'machine learning', 'deep learning', and 'artificial intelligence', clearly indicating a specialization in AI/ML.
- **Course Assignment:** Each course is successfully assigned to a specific specialization, providing a structured way to categorize new and existing courses. This can significantly improve course discoverability and user experience on the MOOC platform.

## Conclusion

This project successfully implements a Python-powered course recommendation system that identifies and assigns courses to specializations using clustering and sentence transformation models. The methodology is scalable and adaptable to growing course catalogs, offering a valuable tool for MOOC platforms to enhance content organization and user engagement. By leveraging advanced NLP techniques, the system automates the process of identifying thematic groupings, which would otherwise be a labor-intensive manual task.
