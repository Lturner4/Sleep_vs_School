from matplotlib import dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

def print_stats(ser, label=''):
    '''
    Prints statistics on a given pandas.Series object
    @Param: ser is pandas.Series object
    @Param: label is the string label to display in print statements. 
    '''
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

def banded_bars(categories=[], cat_labels=[], results=[]):
    '''
    Displays banded bar graph for 1 or more categories using [matplotlib.pyplot](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.html)
    @Param: categories is a list object of category names
    @Param: cat_label is a list object of labels to print next to the category
    @Param: results is a list of results to graph
    '''
    categories = categories
    label = cat_labels
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
        rects = ax.barh(label[0], widths, left=starts, height=0.5, label=colname, color=color)
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)
        ax.legend(ncol=len(categories), bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')
    plt.show()

def box_and_whisker(ser, y_label='y label', x_label='x label', title='Title'):
    '''
    Displays box and whisker plot using [matplotlib.pyplot.boxplot](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.boxplot.html)
    @Param: ser is ArrayLike data to model
    @Param: y_label is string label for y axis
    @Param: x_label is string label for x axis
    @Param: Title is string label for figure
    '''
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
    '''
    Displays histogram with trend line using [matplotlib.pyplot.hist](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.hist.html)
    and [scipy.stats.norm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html?highlight=norm%20pdf)
    @Param: ser is ArrayLike data to model
    @Param: x_label is string label for x axis
    @Param: y_label is string label for y axis
    @Param: Title is string label for figure
    '''
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

def grouped_bar_chart(df, grouping='Attribute', groupby='Groupby', x_label='x label', y_label='y label', title='Title', \
    group1=1, group2=0, group1_label='Group 1', group2_label='Group 2', label_decoder=None, how='count'):
    '''
    Displays grouped bar chart for an attribute with two possible values 
    using [matplotlib.pyplot.bar](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.bar.html)
    and [pandas.DataFrame.groupby](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html)
    @Param: df is a pandas DataFrame object with data to model
    @Param: grouping is string identifier of column to model
    @Param: groupby is string identifier of column to group data by
    @Param: x_label is string label for x axis
    @Param: y_label is string label for y axis
    @Param: Title is string label for figure
    @Param: group1 is first possible outcome of `groupby.get_group()`
    @Param: group2 is second possible outcome of `groupby.get_group()`
    @Param: group1_label is label of group1 for legend
    @Param: group2_label is label of group2 for legend
    @Param: label_decoder allows chart to label graph with appropriate identifiers in the case of numerical data
    @Param: how is string representing 'count' or 'probability'. 'count' will return count of instances in each group, probability will return the 
            probability of this value for the given group
    '''
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
        else:
            print('invalid entry for data analysis, how must be count or probability')

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

def scale(df, *columns):
    '''
    Scales a dataframe using min/max scaling
    @Param: df is a multidimensional ArrayLike object, usually pandas.DataFrame
    @Param: *columns are string identifiers of columns in the DataFrame to scale
    '''
    mini, maxi, rangeof = 0, 0, 0
    for arg in columns:
        mini = min(df[arg])
        maxi = max(df[arg])
        rangeof = maxi - mini
        df[arg] = [((x - mini)/rangeof) for x in df[arg]]
    return df

def normalize(df, label='', scaler=MinMaxScaler()):
    '''
    Normalizes data using [sklearn.preprocessing.MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
    @Param: df is a pandas DataFrame object with data to model
    @Param: label is string id of column for kNN class
    @Param: scaler is MinMaxScaler() to use if not instantiated in calling function. If this function is moved to a class, it should be private. 
    '''
    scaler=scaler
    X_train = df.drop(label, axis=1)
    y_train = df[label].copy()
    scaler = scaler
    scaler.fit(X_train)
    X_train_normalized = scaler.transform(X_train)
    return X_train_normalized, y_train

def train_test(df, test_case=[], label='', k_val=5):
    '''
    Trains and tests kNN model using [sklearn.neighbors](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors)
    @Param: df is a pandas DataFrame object with data to model
    @Param: test_case is ArrayLike data for values of test attributes
    @Param: label is string id of column for kNN class
    @Param: k_val is integer with number of neighbors to use, default is 5
    '''
    X_train = df.drop(label, axis=1)
    y_train = df[label].copy()

    #train
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_normalized, y_train = normalize(df, label=label, scaler=scaler)
    model = KNeighborsClassifier(n_neighbors=k_val)
    model.fit(X_train_normalized, y_train)

    # test
    X_test = pd.Series(test_case, index=df.columns.drop(label))
    X_test = scaler.transform([X_test])
    y_test_prediction = model.predict(X_test)
    print(y_test_prediction)
    return y_test_prediction

def t_test_two(exp, cont, alpha=0.05, test_type='two-tailed'): 
    '''
    Performs t-test using [scipy.stats.ttest_ind](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
    @Param: exp is ArrayLike experimental data
    @Param: cont is ArrayLike control data
    @Param: alpha is float alpha value, default to 0.05
    @Param: test_type is 'one-tailed' or 'two-tailed' for test type. 
    '''
    t, pval = stats.ttest_ind(exp, cont, equal_var=False)
    if test_type == 'one-tailed':
        pval /= 2 # divide by two because 1 rejection region
    if pval < alpha:
        print(f"Reject H0. t: {t}, p: {pval}")
    else:
        print(f"Do not reject H0. t: {t}, p: {pval}")

def t_test_one(exp, hyp_mean, alpha=0.05, test_type='two-tailed'): 
    '''
    Performs t-test using [scipy.stats.ttest_1samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html)
    @Param: exp is ArrayLike experimental data
    @Param: hyp_mean is the hypothesized mean in integer or float form
    @Param: alpha is float alpha value, default to 0.05
    @Param: test_type is 'one-tailed' or 'two-tailed' for test type. 
    '''
    t, pval = stats.ttest_1samp(exp, hyp_mean)
    if test_type == 'one-tailed':
        pval /= 2 # divide by two because 1 rejection region
    if pval < alpha:
        print(f"Reject H0. t: {t}, p: {pval}")
    else:
        print(f"Do not reject H0. t: {t}, p: {pval}")

def line_chart(x_ser, x_label='x label', y_label='y label', title='Title', *args, **kwargs):
    '''
    Displays line graph using [matplotlib.pyplot.plot](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.plot.html)
    @Param: x_ser is ArrayLike data for x axis
    @Param: *args is ArrayLike data to plot
    @Param: y_label is string label for y axis
    @Param: x_label is string label for x axis
    @Param: Title is string label for figure
    '''
    plt.plot(x_ser, *args) # still need to fix this for **kwargs
    plt.legend()
    # lets beautify the plot 
    # fix overlapping x-tick labels 
    plt.xticks(rotation = 25, ha = 'right')
    # labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid() # grid lines, you can have major and minor grid lines and change colors etc
    plt.tight_layout()
    plt.show() # pop up window

def plot_from_df(df,  *args, x_label='x label', y_label='y label', title='Title'):
    '''
    Displays line graph using [matplotlib.pyplot.plot](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.plot.html)
    from a pandas.DataFrame
    @Param: df is 2D ArrayLike data (pd.DataFrame) containing series to plot
    @Param: *args are string identifiers of columns to plot
    @Param: y_label is string label for y axis
    @Param: x_label is string label for x axis
    @Param: Title is string label for figure
    '''
    fig, ax = plt.subplots()
    for x in args:
        ax.plot(df[x])
    # ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=dates.MO)) if dataframes are too large, use this
    plt.xlabel(x_label)
    plt.xticks(rotation = 25, ha = 'right')
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def knn_clf_acc(X, y, metric='euclidean', k=3):
    '''
    tests accuracy of a kNN model created using [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
    and [sklearn.neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.score)
    @Param: X is arrays of data to model, usually as pandas.DataFrame
    @Param: y is ArrayLike containing classifier data
    @Param: metric is string containing distance evaluator, defaulted to 'euclidean' to override method default of 'minikowski'.
    @Param: k is number of neighbors for kNN model
    '''
    # scaler = MinMaxScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    knn_clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn_clf.fit(X_train, y_train)
    accuracy = knn_clf.score(X_test, y_test) 
    print("accuracy = ", accuracy)

def tree_clf_acc(X, y, class_names={1: "weekday", 0: "weekend"}, max_depth=None):
    '''
    tests accuracy of a decision tree model created using [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
    and [sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
    @Param: X is arrays of data to model, usually as pandas.DataFrame
    @Param: y is ArrayLike containing classifier data
    @Param: class_names is a dictionary containing string representations of class attributes
    @Param: max_depth is max depth of decision tree classifier
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    clf = DecisionTreeClassifier(max_depth=max_depth) #random_state=0, max_depth=3)
    clf.fit(X_train, y_train)
    plt.figure(figsize=[30,30])
    X_column = X
    plot_tree(clf, feature_names=X_column.columns, class_names=class_names, filled=True)
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))