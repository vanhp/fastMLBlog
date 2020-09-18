---
toc: false
layout: post
description: Understanding and handling Tabular data
category: ML
title: Working with Tabular Data
image: /ML-section/assets/images/neuralNet1.jpg
---
<!-- ![]({{page.image | relative_url}}) -->
---



## Tabular Modeling Deep Dive


The objective is to predict the value in one column based on the values in the other columns.
There are other technique in ML field such as random forest that also provide good or better result as DL which suite certain type of problem. 

There are plethora of methods in ML many of them only applicable to certain situation. Most of the technique in machine learning can be boil-down to just two proven methods:

For structure data types:
1. Ensembles of decision trees (i.e., random forests and gradient boosting machines), mainly for structured data (such as you might find in a database table at most companies). 
    - Most importantly, the critical step of interpreting a model of tabular data is significantly easier for decision tree ensembles than DL.
    - There are tools and methods for answering the pertinent questions, like: Which columns in the dataset were the most important for your predictions? How are they related to the dependent variable? How do they interact with each other? And which particular features were most important for some particular observation?

For unstructure data types:
2. Multilayered neural networks learned with SGD (Deep Learning) (i.e., shallow and/or deep learning), mainly for unstructured data (such as audio, images, and natural language) 
    - DL also work well with structure data type may be slower to train and hard to intepret result than ensemble decision trees which also don't need GPU, less parameter tuning,more mature ecosystem.
    
    - There are some high-cardinality categorical variables that are very important 
      - ("cardinality" refers to the number of discrete levels representing categories, so a high-cardinality categorical variable is something like a zip code, which can take on thousands of possible levels).
    - There are some columns that contain data that would be best understood with a neural network, such as plain text data.

### Ensemble Decision Trees
Decision tree don't require matrix multiplication or derivative. So PyTorch is of no help. Scikit-learn is better suit for this task.







Handling Different data types from tables
Dataset:
the Blue Book for Bulldozers Kaggle competition
















*Note*

- Entity embedding not only reduces memory usage and speeds up neural networks compared with one-hot encoding, but more importantly by mapping similar values close to each other in the embedding space it reveals the intrinsic properties of the categorical variables... [It] is especially useful for datasets with lots of high cardinality features, where other methods tend to overfit... As entity embedding defines a distance measure for categorical variables it can be used for visualizing categorical data and for data clustering.
- In practice try both methods to see which one better suite for the task
- Deep learning generally work well with highly complex data type






key insign:
1. an embedding layer is exactly equivalent to placing an ordinary linear layer after every one-hot-encoded input layer
2. the embedding transforms the categorical variables into inputs that are both continuous and meaningful.


### Jagons:
- Continuous variables are numerical data, such as "age," that can be directly fed to the model, 
- Categorical variables contain a number of discrete levels, such as "movie ID," for which addition and multiplication don't have meaning

Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-filename.md`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `filename` is whatever file name you choose, to remind yourself what this post is about. `.md` is the file extension for markdown files.

The first line of the file should start with a single hash character, then a space, then your title. This is how you create a "*level 1 heading*" in markdown. Then you can create level 2, 3, etc headings as you wish but repeating the hash character, such as you see in the line `## File names` above.

## Basic formatting

You can use *italics*, **bold**, `code font text`, and create [links](https://www.markdownguide.org/cheat-sheet/). Here's a footnote [^1]. Here's a horizontal rule:

---

## Lists

Here's a list:

- item 1
- item 2

And a numbered list:

1. item 1
1. item 2

## Boxes and stuff

> This is a quotation

{% include alert.html text="You can include alert boxes" %}

...and...

{% include info.html text="You can include info boxes" %}

## Images

![]({{ site.baseurl }}/images/logo.png "fast.ai's logo")

## Code

You can format text and code per usual 

General preformatted text:

    # Do a thing
    do_thing()

Python code and output:

```python
# Prints '2'
print(1+1)
```

    2

Formatting text as shell commands:

```shell
echo "hello world"
./some_script.sh --option "value"
wget https://example.com/cat_photo1.png
```

Formatting text as YAML:

```yaml
key: value
- another_key: "another value"
```

## Tables

| Column 1 | Column 2 |
|-|-|
| A thing | Another thing |

## Tweetcards

{% twitter https://twitter.com/jakevdp/status/1204765621767901185?s=20 %}

## Footnotes


[^1]: This is the footnote.
