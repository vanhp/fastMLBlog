---
toc: true
layout: post
description: Tool and Technique that useful
category: ML
title: Tools and Techniques
image: /ML-section/assets/images/neuralNet1.jpg
hide: false
---
<!-- ![]({{page.image | relative_url}}) -->
---
{% include alert.html text="Warning: This page is under heavy construction and constant changing" %}

### Create test Data
Create a simplest dataset that will allow us to try out methods quickly and easily, and interpret the results is very valuable technique especially in ML where experiment is the norm since the field is active research.

The Human Number dataset

```
from fastai.text.all import *
path = untar_data(URLs.HUMAN_NUMBERS)
Path.BASE_PATH = path
path.ls()
lines = L()
with open(path/'train.txt') as f: lines += L(*f.readlines())
with open(path/'valid.txt') as f: lines += L(*f.readlines())
lines
text = ' . '.join([l.strip() for l in lines])
text[:100]
tokens = text.split(' ')
tokens[:10]
vocab = L(*tokens).unique()
vocab
word2idx = {w:i for i,w in enumerate(vocab)}
nums = L(word2idx[i] for i in tokens)
nums
```

## Basic setup

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
