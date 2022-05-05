import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

def print_stats(ser, label=''):
    print(label, "mid-value:", (ser.min() + ser.max()) / 2)
    print(label, "mean:", ser.mean())
    print(label, "median:", ser.median())
    print(label, "mode:\n", ser.mode(), sep="")
    print(label, "range:", ser.max() - ser.min())
    print(label, "25th percentile:", ser.quantile([0.25]))
    print(label, "50th percentile:", ser.quantile([0.50]))
    print(label, "75th percentile:", ser.quantile([0.75]))
    print(label, "variance:", ser.var())
    print(label, "standard deviation:", ser.std())

def banded_bars(categories=[], cat_label='label', results=[]):
    categories = categories
    label = cat_label
    results = results
    data = np.array(results)
    results_cumsum = data.cumsum()
    category_colors = plt.colormaps['RdYlGn'](np.linspace(0.15, 0.85, data.shape[0]))
    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data).max())
    for i, (colname, color) in enumerate(zip(categories, category_colors)):
        widths = data[i]
        starts = results_cumsum[i] - widths
        rects = ax.barh(label, widths, left=starts, height=0.5, label=colname, color=color)
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)
        ax.legend(ncol=len(categories), bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')
    plt.show()

def box_and_whisker(ser, y_label='y label', x_label='x label', title='Title'):
    plt.boxplot(ser)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    plt.title(title)
    plt.show()

def hist_trend(ser, x_label='x label', y_label='y_label', title='Title'):
    N = ser.size
    mu = ser.mean()
    sigma = ser.std()
    fig, ax = plt.subplots(figsize=(7, 5))
    values, bins, _ = ax.hist(ser, bins=30, color='green', density=True)
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    ax.plot(bin_centers, stats.norm.pdf(x=bin_centers, loc=mu, scale=sigma), color='red', linewidth=3)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{title} (N = {N}): $\mu=$ {mu:.2f} $\sigma=$ {sigma:.2f}')
    fig.tight_layout()

def grouped_bar_chart(df, grouping='Survived', groupby='Pclass', x_label='x label', y_label='y label', title='Title', \
    group1=1, group2=0, group1_label='Survived', group2_label='Deceased', label_decoder=None, how='count'):
    label_decoder = label_decoder
    labels = []
    class_grouping_1 = []
    class_grouping_2 = []

    for group_name, group in df.groupby(groupby):
        labels.append(label_decoder[group_name])
        if how=='count':
            try:
                class_grouping_1.append(group.groupby(grouping).get_group(group1).shape[0])
            except:
                class_grouping_1.append(0)
            try:
                class_grouping_2.append(group.groupby(grouping).get_group(group2).shape[0])
            except:
                class_grouping_2.append(0)
        elif how=='probability':
            try:
                probability = (group.groupby(grouping).get_group(group1).shape[0])/group.shape[0]
                class_grouping_1.append(round(probability, 2))
            except:
                class_grouping_1.append(0)
            try:
                probability = (group.groupby(grouping).get_group(group2).shape[0])/group.shape[0]
                class_grouping_2.append(round(probability, 2))
            except:
                class_grouping_2.append(0)

    x = np.arange(len(labels))  # label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, class_grouping_1, width, label=group1_label, color='green')
    rects2 = ax.bar(x + width/2, class_grouping_2, width, label=group2_label, color='red')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()

# def dual_bar_chart(x_label='x label', y_label='y label', title='Title', \
    # group1=1, group2=0, group1_label='Survived', group2_label='Deceased'):
    # fig, ax = plt.subplots(figsize=(11, 5))
    # width = 0.35
    # x = np.arange(len(labels))
    # rects1 = ax.bar(x - width/2, class_grouping_1, width, label=group1_label, color='green')
    # rects2 = ax.bar(x + width/2, class_grouping_2, width, label=group2_label, color='red')

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel(y_label)
    # ax.set_xlabel(x_label)
    # ax.set_title(title)
    # ax.set_xticks(x, labels)
    # ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

def normalize(df, label='', scaler=MinMaxScaler()):
    X_train = df.drop(label, axis=1)
    y_train = df[label]
    scaler = scaler
    scaler.fit(X_train)
    print(scaler.data_min_)
    print(scaler.data_max_)
    X_train_normalized = scaler.transform(X_train)
    return X_train_normalized, y_train

def train_test(df, test_case=[], label='', k_val=5):
    #train
    scaler = MinMaxScaler()
    X_train_normalized, y_train = normalize(df, label=label, scaler=scaler)
    model = KNeighborsClassifier(n_neighbors=k_val)
    model.fit(X_train_normalized, y_train)

    # test
    X_test = pd.Series(test_case, index=df.columns.drop(label))
    X_test = scaler.transform([X_test])
    y_test_prediction = model.predict(X_test)
    print(y_test_prediction)
    return y_test_prediction
