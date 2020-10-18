from sklearn.cluster import DBSCAN
from sklearn import metrics
import pandas as pd
import numpy as np
from pprint import pprint


def my_db(eps, min_sample, metric, corpus_embeddings):
    """
    :param eps:邻ϵ-域的距离阈值，和样本距离超过ϵ的样本点不在ϵ-邻域内；
    eps过大，更多的点会落在核心对象的邻域内，此时类别数会减少；反之类别数增大；
    :param min_sample:样本点要成为核心对象所需要的ϵ-邻域的样本数阈值；通常和eps一起调参；
    在eps一定的情况下，min_samples过大，则核心对象会过少，此时簇内部分本来是一类的样本可能会被标为噪音点，类别数也会变多；
    反之min_samples过小的话，则会产生大量的核心对象，可能会导致类别数过少
    :param corpus_embeddings: 待聚类的表征数据
    :return: 聚类后的标签信息，核心点信息
    """
    print("eps:", eps)
    print("min_samples", min_sample)
    clf = DBSCAN(eps=eps, min_samples=min_sample, metric=metric)
    # 根据数据训练模型
    db = clf.fit(corpus_embeddings)
    # source = list(clf.fit_predict(training_data))  # 进行聚类,
    return db


def show_db(label_data, db, corpus, corpus_embeddings, corpus_file=None, label_file=None):
    if label_file is not None:
        # 读取真实标签数据
        label_data = pd.read_csv(label_file)
    labels_true = label_data.flag.to_list()
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    clustered_sentences = [[] for i in range(n_clusters_)]
    if corpus_file is not None:
        # 读取原始文本
        corpus = pd.read_csv(corpus_file).content.to_list()
    for sentence_id, cluster_id in enumerate(labels):
        clustered_sentences[cluster_id].append(corpus[sentence_id])

    for i, cluster in enumerate(clustered_sentences):
        print("Cluster ", i + 1)
        print(cluster)
        print("")

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(corpus_embeddings, labels))

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = corpus_embeddings[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = corpus_embeddings[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
