<h1>README.md</h1>
This project is developed for course work of IFT6390 at the University of Montreal. We developed logistic regression on Census data downloaded from UCI. For detailed information of data and the model please check card.pdf and paper.pdf
<h2>Deployment</h2>
To reproduce the result please follow these steps:

1. Fetch: `python run.py fetch`
2. Train: `python run.py train`, please note that training took about 20 minutes on our core i5 laptop, please be patient while model is training. However, we have saved best model weights in evaluation function.
3. Evaluate: `python run.py evaluate` 
4. Build paper: `python run.py build_paper`

<h2>Dependencies</h2>
1. We used Python 3.8.3, and packages: pandas 1.0.5, numpy 1.18.5, requests 2.24.0, matplotlib 3.2.2 \
2. For generating report you need to have latex installed, please check out [latex project website](https://www.latex-project.org/) for more information.