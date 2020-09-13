---
toc: true
layout: post
description: A post about my experince building a web site
date: 2020-09-12 18:30:26
categories: [General]
title: Build Web site with fastpages
image: /images/beach1.jpg
---
<!-- ![]({{page.image | relative_url}}) -->


# Creating Web site using fastpages and Jekyll

As first I thought this going to take alot of work building a web site on your own. Why not pay someone to build one for you I thought.
Being a hand-on kind of person I decide to take the plung building my own site using fastpages and Jekyll software to build the blog and hosted it on github site which provide for free hosting.

Learning something new is going to take time and patient. After the initial hiccup* it's sailing smoothly now. The default site that fastpages created is a little barren for my taste so I have made some modification to fastpages initial site styling to fit my need. I have add a logo image, section link, feature image, image to represent articles. 
I've also hooked up most of social media links. And you can be sure I'll have more modification as I dig deep inside its gut.
I am also planning to improve the site gradually as circumstance permit.

## Set up custom domain
I've also decided to set up my own domain by purchasing domain name from google. Then set it to point to my github page.
To have your very own domain you first have to come up with the domain name you like to use.
Then to purchase a domain name googling for google's Domains site where you can buy and register the domain. Once you purchase the name just fill in the form on the site to register your domain. To use your newly purchase domain instead of github.io domain for your github page you can set it up. This is a two step processes:
1. At google DNS site to make it point to your github site [here](/images/dns-setting.png)
2. At your github page site to use your new domain instead of [yourname].github.io domain.  

To do step 1 you can following this [article](https://dev.to/trentyang/how-to-setup-google-domain-for-github-pages-1p58) 

*here some tips:*
 - The ip address shown there are the ip address of the server at github.io page you have to type them in.
 - The dig command must be run in the terminal console on your machine this is used to check if your setting at google domain DNS server is correct.

To do step 2 
Goto your repo settings tab at the bottom of the page where the GitHub Pages is setup
1. Put your domain name in the custom domain box then click save as shown [here](/images/set-customdomain.jpg)
2. Click enforce HTTPS to secure your site
3. Add a CNAME file at the root folder of your repo (make sure it's not in any folder)
4. In the CNAME file add your domain or subdomain (sub domain is the one with www) e.g. `www.mydomain.com`
5. Edit the _config.yml file
- To set the line `url: "www.mydomain.com"` to match what in the CNAME file
- Remove or comment out the line `"baseurl"` don't need this anymore see my hiccup*

*note:*
if it won't let you click on enforce HTTPS you might have to remove your domain then save the empty box then try again
until it works
 

## The Hiccup
I thought switching to use my own domain instead of github provide domain should be easy, well until it's not.
First whenever I made changed to the code I'd like see what happen to it so I always open the actions tab on my repos to see the CI in action, well, it throw up a build error when I remove the baseurl, dig into it there is an assert line that check if the baseurl in the config file is the same as in some default settings that cause the error

    `assert config['baseurl'] == settings['DEFAULT']['baseurl'], errmsg`

which led me to believe that comment out the `baseurl` is a bad idea, well I later findout it's the opposite. It must be comment out inorder to get my site working. This's delayed my progess afew days. as I posted [here](https://forums.fast.ai/t/fastpages-resources-of-the-page-wont-load-when-use-custom-domain/78790)



## The technical behind the site

Under the hood fastpages use Jekyll as the engine to drive it. Jekyll is a static website and blogs post generator.
It handle HTML, CSS, markdown Liquid and more to finally generate my blog. What's more, I only have to learn Liquid yet another language to my annoyance as if the world needs another language to make it works. And ofcourse, Jekeyll come with its own set of rule here are some of them:

Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-filename.md`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `filename` is whatever file name you choose, to remind yourself what this post is about. `.md` is the file extension for markdown files.

The first line of the file should start with a single hash character, then a space, then your title. This is how you create a "*level 1 heading*" in markdown. Then you can create level 2, 3, etc headings as you wish but repeating the hash character, such as you see in the line `## File names` above.

 Basic formatting

You can use *italics*, **bold**, `code font text`, and create [links](https://www.markdownguide.org/cheat-sheet/). Here's a footnote [^1]. Here's a horizontal rule:

---

 Lists

Here's a list:

- item 1
- item 2

And a numbered list:

1. item 1
1. item 2

 Boxes and stuff

> This is a quotation

{% include alert.html text="You can include alert boxes" %}

...and...

{% include info.html text="You can include info boxes" %}

 Images

![]({{ site.baseurl }}/images/logo.png "fast.ai's logo")

 Code

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


 Tables

| Column 1 | Column 2 |
|-|-|
| A thing | Another thing |


 Tweetcards

{% twitter https://twitter.com/jakevdp/status/1204765621767901185?s=20 %}


 Footnotes



[^1]: This is the footnote.

