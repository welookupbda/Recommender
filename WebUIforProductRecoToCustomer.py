from flask import Flask
from flask import request
from flask import render_template

import ProductRecommendations6months as pr

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template("WebUIforProductRecoToCustomer.html") 

@app.route('/', methods=['POST'])
def my_form_post():
    customerID = request.form['text1']
    #numOfProducts = request.form['text2']

    #content = pr.topRecommendedProducts(customerID, numOfProducts)
  
    content = pr.topRecommendedProducts6months(customerID, 10)

    return (content)

if __name__ == '__main__':
    app.run()
