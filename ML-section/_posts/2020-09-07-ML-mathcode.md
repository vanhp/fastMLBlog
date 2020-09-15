---
toc: false
layout: post
description: Equation,formula,code and ...x?*!
date: 2020-09-07 10:30
category: ML
title: The math and code don't mix?
image: /ML-section/assets/images/ai-pic1.jpg
---
<!-- ![]({{page.image | relative_url}}) -->
---


## <span style="color:red"> Take the red pill!</span>

Anyone who take the red pill and jump into ML pool will soon swim amoung the stream of equations, formula. there no avoiding it. Since you're already take the pill you might as well get use to it. If your day of school are long behind you. The best advice is go to [Khan academy](https://www.khanacademy.org/) site to refresh your memory on them.
Two of the main math subjects you need to have firm grasp of are Linear algebra, and differentiation.

### <span style="color:coral">Linear algebra </span>

Matrix operations are essential. Matrix play a major roll in Neural network operation give you some insight into how neural network works. And allow you to manupulate them to work for you.

### <span style="color:coral">Differention</span>

Do you still remember the chain-rule? If this didn't ring a bell, it's time to bush-up on it. It's at the heart of how the Neural network learn in the process call SGD and it is the logo of this site, take a little peek at the top of the page. The great news is that you don't have to do the math calculation yourself the Framework whichever one you choose will do this for you, your job is to understand them inorder to take full advantage of the tool you use.

---
**----------------------------------------------------------------------------------**

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

