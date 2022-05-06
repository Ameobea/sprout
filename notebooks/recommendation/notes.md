## regularized model training

### attempt 1

5 million ratings corpus

-1 to -1 scaled ratings, all retained.

![](https://ameo.link/u/9wt.png)

Looking at suggestions produced, cosine similarity seems much better than normal euclidean distance.  The latter just recommends the most popular anime.  Idk how it manages to have that be the closest to every user embedding.

### attempt 2

5 million ratings corpus

0.05 to 1 scaled ratings, only positive scaled ratings retained

![](https://ameo.link/u/9wu.png)

It *seems* like it's giving better recommendations.  it's still pretty bad though.  I'm going to let it run again overnight with higher embedding dimension count and more examples.


### attempt 3

25 million ratings corpus

0.05 to 1 scaled ratings, only positive scaled ratings retained

80->120 dimension embedding

learning rate 25->20

training 1500->2500 iterations

Did not converge after 2500 iterations.  Running for another 500.  I forgot to take a screenshot.

OK here's losses after the most recent 500 iters, bringing total up to 3000:

![](https://ameo.link/u/9wv.png)
