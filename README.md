# Example Milvus Vector Embeddings Project

## Overview

This is the companion code repository for [my blog post on using milvus for vector search](https://stephencollins.tech/posts/how-to-use-milvus-to-store-and-query-vector-embeddings).

## Prerequisites

You need to have the following installed:

- Docker Compose
- Python version 3.9

And install the necessary python packages with:

```bash
pip3 install -r requirements.txt
```

## Getting Started

First, spin up the Milvus server running in a docker container with the following:

```bash
sh start.sh
```

Then, run the `app.py`:

```bash
python3 app.py
```
