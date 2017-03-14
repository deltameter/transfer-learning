# Transfer Learning

Performed transfer learning on the CIFAR-100 Dataset, with a 60-40 split of the labels.

Orange: original training on the 60 split labels
Purple: training from scratch on the 40 split labels
Teal: transferred weights from the model trained on the 60 split labels, trained on 40 split labels


![Test Accuracy Chart](https://i.gyazo.com/584b5ad207bb9e1549cac09fbcebb181.png "Test Accuracy Chart")

As you can see, the transferred model trains both quicker and more accurately!
