from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


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


def show_db(labels_true, db, corpus, corpus_embeddings, show=False):
    """
    呈现函数结果
    :param labels_true: 聚类数据的真实标签
    :param db: 训练得到模型
    :param corpus: 原始文本数据的list
    :param corpus_embeddings:  待聚类的表征数据
    :param show:bool值，是否输出聚类的图形效果；默认为false
    :return:
    """
    labels = db.labels_   # 获取预测标签数据
    # 获取聚类数量
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # 获取噪音点信息，在DBSCAN聚类中，噪音点用-1标记
    n_noise_ = list(labels).count(-1)
    # 生成存放聚类数据的容器[[],[],[],[],[]...]
    clustered_sentences = [[] for i in range(n_clusters_)]
    # 将同一类文本放到一个list中
    for sentence_id, cluster_id in enumerate(labels):
        clustered_sentences[cluster_id].append(corpus[sentence_id])
    # 输出聚类结果
    for i, cluster in enumerate(clustered_sentences):
        print("Cluster ", i + 1)
        print(cluster)
        print("")
    # 数据聚类评价信息
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
    if show:
        # 获取聚类核心点信息
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]  # Black used for noise.
            class_member_mask = (labels == k)
            xy = corpus_embeddings[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)
            xy = corpus_embeddings[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
