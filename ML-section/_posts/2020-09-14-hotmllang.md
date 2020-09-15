---
toc: true
layout: post
description: A post about programming language that are in the field and new comer
date: 2020-09-14 04:30:26
category: ML
title: The hot ML Language
image: /Coding-section/assets/images/coding-pic.jpg
---
<!-- ![]({{page.image | relative_url}}) -->

---

# The new kid in town

## The current standard bearer

### <span style="color:green"> Python</span>
In the field of research especially ML the king is Python. 

#### <span style="color:coral">The Advantage</span>
Python is easy to learn language. It was design for non professional programmer. Well those that don't writing code all day. It's a dynamic language in the Language world. It doesn't petster coding to know in advance the specific type of their values are. You simply express what you want to do and the language try to obey. This's great for beginner or infrequence coder who still lot of time thinking about code work on other fields of their interest.


Another advantage of Python is it's ecosystem especially in Data Science field. It's tooling and libraries are top-notch that would serve any fledging data scientist's needed. For example, for matrix computational Numpy[^1] is the undispute king, for data base handling Panda[^2] is the champ. Want to display what you're working on Maplotlib is the goto library. 

Ofcourse, Jupyter notebook is the de-facto tool to write your code in, display the result, and fix the error in your code. You can add document along side your code so people who reading it can understand what it's about. It can even publish your research. It can be passed along to your colleague any one who interest to try out or work on. It's a live document


##### <span style="color:orange">Frameworks and libraries that support Python</span>
PyTorch

Pytorch is the current researcher favorite framework. Since it support most of their needed in a simple form of Python. The code that utilize PyTorch look like a normal Python code. PyTorch seemlessly blend into user code this increase clarity and understanding when reading the code.

TensorFlow

Another popular framework especially in enterprise. It has pletora of features that a professional MLer would needed. Although previous version of TensorFlow is a little harder to work with since it imposed certain workflow, the current version has fully adopt Python int the form of Kera which make coding it in Python a joy.


FastAI
The current state-of-the-Art library that is a layer on top PyTorch. FastAI provide best-practice, utilizing years of experience work in the field of ML in the form of simplication and default setting which let new comer and old-hat alike write simpler code and achieve state-of-the-art result.
And if they prefer to roll-up the sleep and dig into the gut of PyTorch the door is wide open.

#### <span style="color:coral">The Disadvantage</span>
This simplicity comes with a cost it makes the language slow and unreliable in production since it may crash in production due to its dynamic nature which mean the data value that user input is not verify as correct before its perform computation on. Since the machine always require a specific type of data to do computation if these data type is incorrect it simply give up which we call crash. 

Therefore, profession programmer prefers language that verify the type of data as soon as possible. This mean the coder must know the type of data and specify them before any computation can be performed on them. This type of language we call them static language. Since static language require data type be specified it can do verification during "compile-time" this is the time where the user code get translate into intermediate code, this is different from dynamic language which skip this step, before it later on get translate into machine language.


### <span style="color:green"> Julia</span>
Julia on the hand is a static language. It also has feature of dynamic language like Python, well it try to mimic Python to lua Python code to it. It does the data type verification in the background which is called type-inference then inform coder to correct it before proceed.

One of Julia claim to fame is its power. It's is fast blazing fast, it can do drag racing with C the king of speed.
One intesting feature is the code written in Julia look suprisingly like the math formula its try to do computation on.
It's ecosystem is growing fast especially in the scientific computation field. It becoming the favorite goto language beside Python in research community.

It can be "full stack" platform meaning you can write code in it, you can also write library code in it, you can even write driver code in it. The last two are not possible in Python. Which is the major drawing back of Python that the library code must be written in other language normally in C++ this is because Python is too slow. 

Library code must be fast since it is intend to be used by many situations and environment such as in production code. The same go with driver code which control the underlining hardware.

##### <span style="color:orange">Frameworks and libraries that support Julia</span>
Flux

Flux is Julia home-grown library for ML. It's design and written in Julia to work with Julia from top to bottom. This include user code, library code, and driver code all is written in Julia which is amazing in its own right. It's popularity growing by leaf-and-bound.

FastAI

This is the planing state under the label of "under heavy construction".

### <span style="color:green">Swift</span>

Another new comer in ML field it was develop by Apple as the replacement for their aging Objective-C language which born 30 years agos.

Swift come to fame is its speed that rival C++, it's being push by Google especially by the Tensorflow team as their next generation language.
However, its ecosystem is barren which make it hard for any fledging data scientist to work in.

##### <span style="color:orange">Framework and libraries that support Swift</span>
TensorFlow

This is not a hugh surprise mind you. It currently has auto differential library which allows user to skip writing backward pass if he/she prefers. This is an important step forward.

FastAI

This is the planing state under the label of "under heavy construction".













---

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



[^1]: numpy doc
[^2]: Panda doc

