---
title: 'Embeddings.jl: easy access to pretrained word embeddings from julia'
tags:
  - julialang
  - opendata
  - NLP
  - word embeddings
  - machine learning
authors:
 - name: Lyndon White
   orcid: 0000-0003-1386-1646
   affiliation: 1

affiliations:
 - name: The University of Western Australia
   index: 1

date: 30 Aug 2018
bibliography: paper.bib
---

# Summary

Embeddings.jl is a tool help users of the Julia programming language ([@Julia]), make use of pretrained word embeddings for NLP.
Word embeddings are a very important feature representation in natural language processing.
The use of embeddings pretrained on very large corpora can be seen as a form of transfer learning.
It allows knowledge of lexical semantics derived from the distributional hypothesis: that words occurring in similar contexts have similar meaning,
to be injected into models which may have only limited amounts of supervised, task oriented training data.

Generously many creators of word embedding methods have made sets of pretrained word representations publicly available.
Embeddings.jl exposes these as standard matrix of numbers and a matching array of strings.
This makes it easy to use them with julia machine learning packages such as Flux ([@flux]).
In such deep learning packages they are commonly used in an input layer in LSTMs.
Where may be frozen, or used only for initialization then fine tuned on the supervised task.
They also maybe summed to represent a bag of words, and used with other machine learning methods.


Embeddings.jl makes use of DataDeps.jl ([@2018arXiv180801091W]),
to allow for convenient automatic downloading of the data when and if required.
It also uses the DataDeps.jl prompt to ensure the user of the embeddings has full knowledge of the original source of the data, and which papers to cite etc.

It currently provides access to
 - multiple sets of word2vec embeddings ([@word2vec]) for English
 - multiple sets of GLoVE ([@glove]) embeddings for English
 - multiple sets of FastText embeddings ([@fasttext157lang], [@fasttext]) for several hundred languages

It is anticipated that as more pretrained embeddings are made available for more languages and using newer methods,
the Embeddings.jl package will be updated to support them.
	

# References
