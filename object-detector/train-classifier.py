# Import the required modules
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import argparse as ap
import glob
import os

# Parse the command line arguments
parser = ap.ArgumentParser()
parser.add_argument('-p', "--posfeat", help="Path to the positive features directory", required=True)
parser.add_argument('-n', "--negfeat", help="Path to the negative features directory", required=True)
args = vars(parser.parse_args())

pos_feat_path =  args["posfeat"]
neg_feat_path = args["negfeat"]

# Classifiers supported
clf_type = ["LIN_SVM", "LOG_REG"]

fds = []
labels = []
# Load the positive features
for feat_path in glob.glob(os.path.join(pos_feat_path,"*.feat")):
    fd = joblib.load(feat_path)
    fds.append(fd)
    labels.append(1)

# Load the negative features
for feat_path in glob.glob(os.path.join(neg_feat_path,"*.feat")):
    fd = joblib.load(feat_path)
    fds.append(fd)
    labels.append(0)

clf = LinearSVC()
clf.fit(fds, labels)
joblib.dump(clf, '../data/svm.model')
