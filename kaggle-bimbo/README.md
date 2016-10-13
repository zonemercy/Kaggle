# Grupo Bimbo Inventory Demand - Maximize sales and minimize returns of bakery goods

My simplified model solution for the Grupo Bimbo kaggle competition https://www.kaggle.com/c/grupo-bimbo-inventory-demand. reach around 0.43785 on Public LB (~26th) / 0.46028 on Private LB (~57th). 

# models
- GBRT with Sframe by [Graphlab Create](https://turi.com/products/create/)
- [ftrl](https://www.kaggle.com/scirpus/grupo-bimbo-inventory-demand/ftlr-use-pypy) with [pypy](http://pypy.org/)

With additional parameters tuning it should be able to avoid overfitting.