# Hierarchical Hidden Markov Models
#### An approximation of neocortex structures, according to Ray Kurzweil

In [a recent post](https://www.kurzweilai.net/dialogue-a-conversation-on-creating-a-mind), famous futurist Ray Kurzweil mentions that - in his opinion - brain structures in the neocortex are technically similar to hierarchical hidden Markov models (HHMM).
An idea he also explained in more detail in his 2012 book "How to Create a Mind" [1].

Unfortunately though, neither the article nor the book has enough information to understand this machine learning model in detail, let alone implement it.
A pity for any hobby AI scientist interested in the implementation of conscious machines!

So let's use this article then, to try and understand hierarchical hidden Markov models.
We'll have a brief, high-level look at most of the concepts it builds on and prepare the stage for an actual implementation (which will follow in another article due to content length).

Hierarchical hidden Markov models are, as the name implies, based on hidden Markov models, which are in turn based on Markov chains, all of which are stochastic processes.
We'll start with the last concept and work our way backwards.

*NB: This is a light-weight introduction to the HHMM topic. We'll stick to examples and simple concepts and defer the math to the point when we're implementing this.*

## Stochastic Processes, A Brief Intro

As mentioned, all of the things we'll look at in this article are stochastic processes.
I'm sure you will already know what that is, but as it's fundamental to the whole article, let us briefly take a look at it again:

[Wikipedia defines stochastic processes](https://en.wikipedia.org/wiki/Stochastic_process) informally as:

> a sequence of random variables

To build an understanding of what this means, picture a certain type of random event, like the weather or a coin flip, measured or observed in sequence, for example once daily, or for 10 consecutive flips.
When trying to describe this mathematically, we can make use of a stochastic process.
We'd simplify the real-world phenomenon and try to calculate probability distributions for future states of it.

As a difference to [statistical models](https://en.wikipedia.org/wiki/Statistical_model), this will include an element of iteration: Previous states affect the current observations.
Did it rain today? Then based on experience, there is a high probability that it might at least be cloudy tomorrow.

On a side note, I have the feeling this also more closely resembles how we build models mentally: Instead of thinking of smooth function-like relationships of input to output variables (linear or otherwise), we tend to think in if-x-happens-now-then-y-might-happen-later kind of models.

E.g. instead of thinking 

> If I observe that the temperature is ABC and the air pressure is XYZ, changing at the rate of UVW, then within 3 hours and 20 minutes (give or take 10 minutes) I can expect rain to come with an intensity of DEF.

we'd probably rather think 

> It was cloudy all day long and seems to be getting darker and colder, so theres a good chance we'll have rain soon.

And not just because we don't sit down with a thermometer, barometer and stopwatch to predict the weather, but also because it's somehow really hard to build and use these kind of functional models mentally.
I assume this is related to Kahneman's ideas in "Thinking Fast, Thinking Slow" [2], from which I take that we usually prefer using quick pattern-matching over slow, analytical thoughts - but I cannot provide any scientific evidence for this.

In any case, the theory of stochastic processes is a lot richer than the examples might imply.
However, for understanding HHMMs we don't need all of that detail right now.

Based on these ideas it will now be almost trivial to understand Markov chains - promised!
So without further ado, let's jump into this next topic now.

## Markov Models & Chains

A [Markov model](https://en.wikipedia.org/wiki/Markov_model), named after the Russian mathematician [Andrey Markov](https://en.wikipedia.org/wiki/Andrey_Markov) is a stochastic model (or process) with a specific property - called, who would have guessed, the Markov property - that requires a future state of the process to be depending only on the current state.

Let's go back to the weather and model it simplistically with only three states (that we check for each morning): 'Sunny', 'Cloudy' and 'Rainy'.
Now, by writing down our observations over a period of time and counting how often e.g. 'Cloudy' followed after 'Sunny', we can create a model similar to this one:

![Sample markov chain with weather states and transition probabilities](/home/amnesia/projects/hhmm/weather_mc.png)
**State transition probability diagram of a simple weather model. Note that all transitions from one state to another only depend on the current state and their probabilities sum up to 100%. (Graphic by author)**

With this, we defined a simple [Markov chain](https://en.wikipedia.org/wiki/Markov_chain).
That is to say, this model exhibits the Markov property and all states of the system modeled and their transition probabilities are known and included in it.

Let's stick with this concept a little longer and look at another example:  
Simplified text generation.

We can regard text as a sequence of words (and punctuation), where certain combinations of words are more probable than others - i.e. some words are more likely to follow one another than others.
This can be represented by a Markov chain to a certain degree, by simply reading in sample texts and counting word-to-word occurrences. (Here: The start of Moby Dick by Herman Melville)

> **Call** **me** Ishmael.  Some years ago - never mind how long precisely - having little or no money in my purse, and nothing particular to interest **me** on shore, I thought I would sail about a little and see the watery part of the world.

- Call -> me (100%)
- me -> Ishmael (50%)
- me -> on (50%)
- ...

And so forth. This can also be extended to several words, i.e. finding the next word after a combination of two or more words (also known as 2-grams or [n-grams](https://en.wikipedia.org/wiki/N-gram) respectively).

Not only does this allow us to build a stochastic model of the studied text, we can also use it to generate more like it, by starting with a random start word and picking next words with the recorded probabilities.

## Hidden Markov Models

To build a Markov chain as discussed in the earlier section, we need to be able to directly observe the occurrences of the states.
To continue from the last example: In order to create the word-to-word probability model, you need to read-in lots of text, i.e. lots of word-to-word pairs.

But many things in the world are not directly observable, and we can only estimate what is going on by looking at what measurable effects these 'hidden' states have.

To go back to the example with the weather, you could imagine standing indoors looking out through the window and trying to estimate whether it's cold enough to wear coat, hat and mittens, or whether you could still go outside just dressed with a sweater.
If you don't have access to a thermometer and can't try to feel the temperature, you can still reason about the temperature by looking at its effects: E.g. what other people are wearing or if there is snow and frost outside.

Modeling these unobservable processes and their effects is what [hidden Markov models](https://en.wikipedia.org/wiki/Hidden_Markov_model) were made for.
Intuitively, based on what we learned before, they consist of a Markov chain at their core, with states and the probabilistic transitions.
The twist now lies in the fact that these states of the system cannot be observed or measured, giving us no easy way to calculate the transition probabilities directly.
Instead of that though, we can measure other effects the states cause.
These effects are probabilistic of their own account though.

Let's again look at the weather and the potential effects it might have on the world:

![Sample HMM with weather states and effects](/home/amnesia/projects/hhmm/weather_hmm.png)
**State transition diagram of 'hidden' Markov chain with effects and probabilities thereof. Note that one state can potentially have more than one observable effect, which leads to the problem that states cannot be directly inferred from observations. (Graphic by author)**

In this example, we again have three states of underlying weather (a bit simplistic, but oh well), however, this time we also show their observable effects on the world (in this case, what people are wearing) and the probabilities of seeing them.

Note that Markov Chains can be conceptually modeled as HMMs as well: If each state only results in exactly one observation with certainty (100% of the time) then we're back at the simpler Markov chain model.

### Uses

Models like this are usually used in two different ways:

1. Estimating the probability of a certain sequence of observations. E.g. how likely is it to observe people wearing "Sweater" -> "Rain Coat" -> "Sweater" -> "Rain Coat"? I guess it also depends on whether you're in the UK...
2. Based on a sequence of observations, try estimating the sequence of states that it stems from. E.g. if you see "T-Shirt" -> "Sweater" -> "T-Shirt", the states of the world were most likely "Sunny", "Sunny", "Sunny". Nice!

That is, if you already created the model and optimized it.
If that's not done yet, you face another challenge:

3. Given a model structure (i.e. states, transitions and effects) and a bunch of observation sequences, try finding the most likely probabilities for all the 'arrows'. I.e. train the model.

It goes without saying, that all of these problems have been addressed for many years now and formulas are well known. We'll go through them in more detail when we implement them.

## Hierarchical Hidden Markov Models

So far we only looked at an extremely simple HMM.
It's simple, yet illustrates the point of the model quite clearly.
Let's do the same for hierarchical hidden Markov models as described in the 1998 paper [3].

Where HMMs can be understood as a directed graph where each state is reachable, the hierarchical version is, in a sense, more restricted. (Actually, the authors of [3] also mention that an HHMM can be represented as a fully-connected HMM, with the downside of loosing the semantics of the hierarchy)

So what is an HHMM then?
Simply put, as the name implies, a hierarchical HMM adds a tree-like hierarchy to the hidden states.
It starts with a root node representing the first layer, which has state transitions to each of a second layer of states. The second layer in itself is structured like an Markov chain, i.e. the states have transitions with each other.
Each state in that layer however, can be the root node of another HHMM.
This goes down recursively until we reach the leaf nodes, dubbed production states, that behave like 'normal' HMM states and output a single observation or symbol. They, too, are connected.
There is one last special 'end' state in each layer, that, when reached, automatically (with 100% probability) jumps back one level.

In terms of order, state transitions go deeper first, then, when the signal comes back through the lower-level end state, other states on the same level are activated.

With this structure, non-production states don't have a output assigned directly. However, their output is implicitly described by the sequence of outputs of the lower-level production nodes.
To illustrate the point: Think about the production nodes like being letters or characters in a text. Higher-level nodes could then progressively represent syllables, words, word combinations, sentences and so on.
This is the nature of the hierarchy here: It is used to represent abstractions over simpler concepts.

The mentioned paper used the approach to train the model on an English text corpus, which shows the ideas quite nicely [3].
The following graph is an annotated subset of their findings:

![Sample HHMM with outputs](/home/amnesia/projects/hhmm/text_hhmm.png)
**State transitions and example path within a simplified hierarchical hidden Markov model. Here, we show text production going from letters or letter combinations to short words to parts of a sentence. Probabilities are not included to enhance legibility. (Graphic by author)**

### Relation to the Human Brain

According to Kurzweil, these hierarchies of pattern matchers and/or output producers are how we can picture the structure of the human neocortex [1].
This is the part of our brain that takes-on higher-level tasks, such as language understanding and production.

The same way that the example above produces more and more complex text the higher in the hierarchy we look at the output, an equivalent structure in the human brain could e.g. include concepts like humor and irony on the top-level.

This similarity is why I assume Kurzweil suggested these as viable models for a human brain. Or, more specifically, for a conscious mind.

### Uses

A very famous example use case for models like these and HMMs (practically since the 1970's) is speech recognition and speech synthesis.
In these, short sequences of sound, analyzed using a Fourier transform are 'translated' into most likely phonemes and - on a higher level - words from a known vocabulary.

These algorithms have continuously improved since then and enable us now to use voice assistants like Siri or Alexa on our mobile devices.

## Next Steps

As mentioned, I'll continue this topic with a more technical article illustrating an implementation of an HHMM, based on the research presented in [3].
My hope is to reproduce (in a tutorial style article) some of their results on natural language analysis. In the best case, we'll see some nice hierarchy-level dependent abstractions of sample texts.t

---

All finished source documents, notebooks and code related to this is also available on [Github](). Please feel encouraged to leave feedback and suggest improvements.

---

[1] R. Kurzweil, "How to create a mind: The secret of human thought revealed" (2012), Penguin
[2] D. Kahneman, "Thinking, fast and slow" (2011), Macmillan
[3] S. Fine, Y. Singer, and N. Tishby, [The hierarchical hidden Markov model: Analysis and applications](https://link.springer.com/content/pdf/10.1023/A:1007469218079.pdf) (1998), Machine learning 32.1 (pp. 41-62)
