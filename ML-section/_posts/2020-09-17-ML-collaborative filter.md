---
toc: false
layout: post
description: How Recommend System, Sentimental Analysis  work
category: ML
title: Collaborative Filtering Deep Dive
image: /ML-section/assets/images/ai-pic2.jpg
---
<!-- ![]({{page.image | relative_url}}) -->
{% include alert.html text="Warning: This page is under heavy construction and constant changing" %}
---
 Collaborative Filtering a general class of ML known as Recommend system that recommend a product to a customer use in Amazon, a movie or video to viewer which is used in Netflix, what's story to show in Facebook,Tweeter...
Which is base the user own history, and some data collect from other users that may have similar preference. For example, Netflix recommend a movie to you base on other people who watch the same movie. Or Amazon recommend a product base on other who bought or view the same product. 

These methods it's base on the idea of latent factors the unwritten or unspecify context that underline the item (product).
```
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
from fastbook import *
rom fastai.collab import *
from fastai.tabular.all import *
path = untar_data(URLs.ML_100k)
ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,
                      names=['user','movie','rating','timestamp'])
ratings.head()

last_skywalker = np.array([0.98,0.9,-0.9])
user1 = np.array([0.9,0.8,-0.6])
(user1*last_skywalker).sum()
casablanca = np.array([-0.99,-0.3,0.8])
(user1*casablanca).sum()
```
Learning the Latent Factors

DataLoaders
```
movies = pd.read_csv(path/'u.item',  delimiter='|', encoding='latin-1',
                     usecols=(0,1), names=('movie','title'), header=None)
movies.head()
ratings = ratings.merge(movies)
ratings.head()

#build dataloaders
#By default, it takes the first column for the user, the second column for the item 
dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
dls.show_batch()
dls.classes

n_users  = len(dls.classes['user'])
n_movies = len(dls.classes['title'])
n_factors = 5

user_factors = torch.randn(n_users, n_factors)
movie_factors = torch.randn(n_movies, n_factors)
```

To calculate the result for a particular movie and user combination, we have to look up the index of the movie in our movie latent factor matrix and the index of the user in our user latent factor matrix; then we can do our dot product between the two latent factor vectors. 
To do lookup in matrix form which model can calculate is to represent lookup with vector of one-hot encoding

```
one_hot_3 = one_hot(3, n_users).float()
user_factors.t() @ one_hot_3
user_factors[3]

```
### Embedding
A technique of look up item by using matrix multiplication with one-hot encode vector. PyTorch has a special layer that do this task in a fast and efficient way.

### Create Collaborative by hand from scratch

```
class DotProduct(Module):
    # constructor
    # use slightly higher range since sigmoid max of 5 to get 5 need to over
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
        self.user_factors = Embedding(n_users, n_factors)
        # account for user bias
        self.user_bias = Embedding(n_users, 1)
        self.movie_factors = Embedding(n_movies, n_factors)
        # account for bias in data
        self.movie_bias = Embedding(n_movies, 1)
        self.y_range = y_range
     
    # callback from Pytorch   
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        res = (users + movies).sum(dim=1,keepdim=True)
        res += self.user_bias(x[:,0]) + self.movie_bias(x[:,1])
        #force those predictions to be between 0 and 5 with sigmoid_range
        return sigmoid_range(res, *self.y_range)

model = DotProduct(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3)
```


### Overfitting

A phenomena that happen when the model start to recgonize the dataset and try to memorize them this make the result look very good for this particular dataset but fail when encounter different dataset. This is because the model fail to generalize the learning so that is applicable on different dataset. This usually happen when there not enough data or train too many times on the same dataset.

### Regularization

A technique to reduce overfitting by add in a value that act as penalty when the model start overfit.

### Weight Decay (L2 regularization)

Add the sum of all weight squared to the loss function to affect the gradient calculation when add the weight
 to reduce it value. Visually this is like make the canyon bigger in gradient curve space

 ```loss_with_wd = loss + wd * (parameters**2).sum()```

 using derivative equivalence to

```parameters.grad += wd * 2 * parameters```

in practive just pick a value and double it

```
def create_params(size):
    return nn.Parameter(torch.zeros(*size).normal_(0, 0.01))

class DotProductBias(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
        self.user_factors = create_params([n_users, n_factors])
        self.user_bias = create_params([n_users])
        self.movie_factors = create_params([n_movies, n_factors])
        self.movie_bias = create_params([n_movies])
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors[x[:,0]]
        movies = self.movie_factors[x[:,1]]
        res = (users*movies).sum(dim=1)
        res += self.user_bias[x[:,0]] + self.movie_bias[x[:,1]]
        return sigmoid_range(res, *self.y_range)

model = DotProductBias(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.1)

```
Understanding the bias
grap some lowest value in bias vector data for a user
```
movie_bias = learn.model.movie_bias.squeeze()
idxs = movie_bias.argsort()[:5]
[dls.classes['title'][i] for i in idxs]
```
analysis

even when a user is very well matched to its latent factors(action,age of movie) they still generally don't like it. Meaning even if it is their kind of movie, but they don't like these movies

grap some high bias value
```
idxs = movie_bias.argsort(descending=True)[:5]
[dls.classes['title'][i] for i in idxs]
```
this show even persons don't like the kind of movie they may still enjoy it.

visualize latent factors with more data using PCA 

```
g = ratings.groupby('title')['rating'].count()
top_movies = g.sort_values(ascending=False).index.values[:1000]
top_idxs = tensor([learn.dls.classes['title'].o2i[m] for m in top_movies])
movie_w = learn.model.movie_factors[top_idxs].cpu().detach()
movie_pca = movie_w.pca(3)
fac0,fac1,fac2 = movie_pca.t()
idxs = np.random.choice(len(top_movies), 50, replace=False)
idxs = list(range(50))
X = fac0[idxs]
Y = fac2[idxs]
plt.figure(figsize=(12,12))
plt.scatter(X, Y)
for i, x, y in zip(top_movies[idxs], X, Y):
    plt.text(x,y,i, color=np.random.rand(3)*0.7, fontsize=11)
plt.show()
```
### using fastai collab feature
```
learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))
learn.fit_one_cycle(5, 5e-3, wd=0.1)
learn.model
movie_bias = learn.model.i_bias.weight.squeeze()
idxs = movie_bias.argsort(descending=True)[:5]
[dls.classes['title'][i] for i in idxs]

```
Measuring Distance of embbeding value

If there were two movies that were nearly identical, then their embedding vectors would also have to be nearly identical, and the viewer also would have the same likeness with same similarity vectors. Which mean the distance between the two movies are closer together.
```
# find the movie similar to silence of the lambs
movie_factors = learn.model.i_weight.weight
idx = dls.classes['title'].o2i['Silence of the Lambs, The (1991)']
distances = nn.CosineSimilarity(dim=1)(movie_factors, movie_factors[idx][None])
idx = distances.argsort(descending=True)[1]
dls.classes['title'][idx]
```
The clean slate problem
When starting out there no data relationship (latent factor) for the user, movie,product... 
The solution is use common sense e.g. let user pick their movies from a list, assign a product base on environment, collect as much info about them as possible. Don't let a few group of user,items,products have too much influent which skew the whole dataset.

Deep Learning for Collaborative Filtering
To create DL collaborative filter do:
1. concastenate activation value result from lookup together
2. these matrices don't have be the same size (not doing dot-product)
3. use fastai function get_emb_sz to get the recommend sizeof embedding matrix for the dataset

```
embs = get_emb_sz(dls)
embs

class CollabNN(Module):
    def __init__(self, user_sz, item_sz, y_range=(0,5.5), n_act=100):
        self.user_factors = Embedding(*user_sz)
        self.item_factors = Embedding(*item_sz)
        self.layers = nn.Sequential(
            nn.Linear(user_sz[1]+item_sz[1], n_act),
            nn.ReLU(),
            nn.Linear(n_act, 1))
        self.y_range = y_range
        
    def forward(self, x):
        embs = self.user_factors(x[:,0]),self.item_factors(x[:,1])
        x = self.layers(torch.cat(embs, dim=1))
        return sigmoid_range(x, *self.y_range)

model = CollabNN(*embs)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.01)
# or using fastai version if nn=True it will auto call get_emb_sz 
learn = collab_learner(dls, use_nn=True, y_range=(0, 5.5), layers=[100,50])
learn.fit_one_cycle(5, 5e-3, wd=0.1)

# a look inside fastai internal code
@delegates(TabularModel)
class EmbeddingNN(TabularModel):
    def __init__(self, emb_szs, layers, **kwargs):
        super().__init__(emb_szs, layers=layers, n_cont=0, out_sz=1, **kwargs)


```






*Note*
- In PyTorch: To read parameter value use nn.Parameter which also auto call requires_grad_
- In Python:(variable length argument in C++) **kwargs in a parameter list means "put any additional keyword arguments into a dict called kwargs

    - **kwargs in an argument list means "insert all key/value pairs in the kwargs dict as named arguments here"
    - this make argments obscure from the tool since they all pack into a dictionary
    - fastai use @delegates docorator to auto unpack it so the tool can see


### Jagons
- item: refers to product,movie,story,a link, topic...
- latent factor: and unspecify info that affect the item
- Embedding: Multiplying by a one-hot-encoded matrix thaat is the same as lookup or index into array to get an item
- weight decay (L2) wd in fastai control the sum of square value
- PCA principle component analysis use to reduce the size of the matrix to smaller size
- probabilistic matrix factorization (PMF)
