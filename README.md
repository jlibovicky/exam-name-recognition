# Student Name Recognition on Exam Sheets

This is a set of scripts for automatic recognition of students names on exam
sheets.

## Processing the exam sheet

The input is a list of students which are enrolled (i.e., the list of names
that might occur on the exam sheets) and the scanned exam sheets.

The exam sheet has a Aruco markers in the corners and a pre-made boxes for name
and surname. The script `process_scan.py` contains hard-coded positions of the
markers and. (Although there are probably better and more robust ways to to the
image registration.) The markers are also used to recognize the first page of
the exam where the name is written.

After the scan is aligned with the templated, it cuts off the boxes with
letters and runs character recognition classifier into 27 classes corresponding
to 26 characters of English alphabet and an empty box. Then, using the
probability distributions returned by the classifier, it computes a probability
score for each of the potential student names.

This leads to a score for each names and exam sheet pair. Finally, the
assignment of the sheets and the student names is computed as a maximum
matching in a bipartite graph using the `linear_sum_assignment` algorithm from
SciPy.

## Character Recognition Model

Character recognition is done using a simple convolutional NN trained on the
[A-Z Handwritten
Alphabets](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)
from Kaggle, which looks like the famous MNIST dataset, but contains 26 English
character classes.

The recognition uses modified [ShuffleNet](https://arxiv.org/abs/1707.01083)
and is based on an implementation from a [iPython notebook published on
Kaggle](https://www.kaggle.com/code/taranmarley/modified-shufflenet-99-5/notebook).

After downloading the dataset from Kaggle, script `train_ocr.py` trains the CNN
for character recognition and saves a checkpoint that is then used by the
`process_scan.py` script.

## JavaScript for assign credits in SIS

The script `generate_js.py` takes a list of students names and generates a
JavaScript that can be copied into the JavaScript console in the browser. It
checks checkboxes for the given names in the corresponding page of SIS, the
Charles University information system.
