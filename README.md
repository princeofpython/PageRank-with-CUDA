# PageRank-with-CUDA
Parallelizing Google's PageRank algorithm in C++ with CUDA framework on GPU. Conducted some experiments, tried new ideas. Final Course Project for CS6023: GPU Programming on 2021 Spring, IIT Madras.

- [**Detailed Report of Everything**](https://github.com/princeofpython/PageRank-with-CUDA/blob/main/Report.pdf)
- [Sequential Code (C++)](https://github.com/princeofpython/PageRank-with-CUDA/blob/main/sequential.cpp)
- [Parallel Code (C++ with CUDA)](https://github.com/princeofpython/PageRank-with-CUDA/blob/main/parallel.cu)
- [Demo in Jupyter Notebook](https://github.com/princeofpython/PageRank-with-CUDA/blob/main/pagerank.ipynb)
- [Datasets (File name indicates number of nodes)](https://github.com/princeofpython/PageRank-with-CUDA/tree/main/data)
- [Tex Directory](https://github.com/princeofpython/PageRank-with-CUDA/tree/main/PageRank)

We use the Internet to search for information on a daily basis. We use
search engines to find out the useful information from the vast Internet.
This is possible because the search engines are using a heuristic called
PageRank. It is nothing but a value to a web page that is assigned by a
PageRanking algorithm. This algorithm scans all possible webpages and
then calculates the rank accordingly given by a formula. Search engines
show results according these ranks, which stand for the popularity of the
page. The lower is the rank, more popular the page is. Traditional approaches use multi-CPU architecture and this is not a very good choice
due to the communication overhead and the low processing power of CPU
compared to GPU. Hence, designing a PageRanking algorithm efficiently
modified for parallel GPU-CPU environment that achieves higher accuracy and consumes lesser time to evaluate the PageRank exact rank vector
even for large-scale webgraphs.

|No: of nodes| C++| CUDA C++| speedup|
|---|---|---|---|
|100 |405 |593| 0.68|
|200 |6556| 2152| 3.04|
|400 |3462 |936 |3.69|
|800 |22785 |2356 |9.67|
|1000 |2972898 |104831 |28.35|
|5000 |69266776 |816446| 84.83|
|10000| 68519880| 755148| 90.73|
|16000 |26509166| 626986 |42.28|
|18000 |1030231744| 18294508 |56.31|
