from collections import Counter
import numpy as np
import csv, collections, re, os,io
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from fractions import Fraction

np.set_printoptions(threshold=np.nan)

# classes={
# 	'Opiniones':'0',
# 	'No respondibles':'1',
# 	'Particulares':'2',
# 	'Otros':'3',
# 	'Archivada':'4',
# 	'Web':'5',
# 	'Complejas':'6',
# 	'Web y Archivada':'7'
# }

classes={
	'Opiniones':0,
	'No respondibles':1,
	'Particulares':2,
	'Otros':3,
	'Archivada':4,
	'Web':5,
	'Complejas':6,
	'Web y Archivada':7
}
tags=[]

with io.open('vector.txt', 'w+', encoding='utf-8') as vectorFile:
	with io.open('toBOW.txt', 'w+', encoding='utf-8') as svmFile:	

		with open('text.csv', 'rU') as icsvfile:
			spamreader = csv.reader(icsvfile, delimiter=',')
			bows=[]
			
			for i,row in enumerate(spamreader):	
				# print 'row lenght {}'.format(len(row))
				if i>0:
					idText = row[0].replace('\n', ' ').strip();
					text = row[1].replace('\n', ' ').strip();
					text = ' '.join(x.strip() for x in text.split())
					tag = row[2].replace('\n', ' ').strip();
					# print 'id: {}'.format(idText);
					# print 'text: {}'.format(text);
					# print 'tag: {}'.format(tag);
					# print 'class: {}'.format(classes[tag]);
					bow = collections.Counter(re.findall(r'\w+', text.lower()));
					bows.append(bow)
					tags.append(classes[tag])
					# titleFile.write(unicode(title+'|'+tag+'|'+cat+'\n', errors='ignore'));
			# print bows
			sumbags = sum(bows, collections.Counter())
	    	mostCommon = sumbags.most_common()
	   		 # print 'Dimensions: {}'.format(len(sumbags))
	   		 # print 'Sumbags: {}'.format(sumbags)
	    	# print 'Most common: {}'.format(mostCommon)

	    	dim = len(mostCommon)
	    	# print 'dim: {}'.format(dim)

	    	coords=[]
	    	for i,v in enumerate(bows):

	            dic=dict(v)
	            # print 'dic: {}'.format(dic)
	            fileLine='{}'.format(tags[i])

	            vector=np.zeros(dim)
	            for index, (word,count) in enumerate(mostCommon):
	                # print 'index: {}'.format(index)
	                # print 'word: {}'.format(word)
	                # print 'count: {}'.format(count)

	                if dic.get(word) > 0:
	                    feat = ' {}:{}'.format(index+1,dic[word])
	                    vector[index]=dic[word]
	                    fileLine+=feat
	            svmFile.write(unicode(fileLine+'\n', errors='ignore'));
	            # print '{} {}'.format(i,tags[i])
	            coords.append(vector)
        # print tags
        # tags = np.array(tags)
        # print tags
        print len(coords)
        # db = DBSCAN(eps=100, min_samples=10).fit(coords)
        metricsSelected = ['canberra', 'chebyshev', 'euclidean', 'manhattan']
        with io.open('params.txt', 'w+', encoding='utf-8') as f:

	        for samples in range(2,10):
		        for base in range(1,1000):
		        	dec = Fraction(base,100)
		        	eps = float(dec)
		        	
	        		for metric in metricsSelected:
				        db = DBSCAN(eps=eps, min_samples=samples, metric=metric, algorithm='auto').fit(coords)
				        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
				        # print 'core_samples_mask: {}'.format(core_samples_mask)
				        core_samples_mask[db.core_sample_indices_] = True
				        labels = db.labels_
				        
				        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
				        txt = 'metric: {} clusters: {} eps: {} , samples: {}'.format(metric,n_clusters_,eps,samples)
				       	print txt
				        if n_clusters_ >3:
				        	print 'labels: {}'.format(labels)
				        	# print 'labels: {}'.format(labels)
			
				        	f.write(unicode(txt+'\n', errors='ignore'));
				        	homogeneity_score=metrics.homogeneity_score(tags, labels)
				        	completeness_score=metrics.completeness_score(tags, labels)
				        	v_measure_score=metrics.v_measure_score(tags, labels)
				        	adjusted_rand_score=metrics.adjusted_rand_score(tags, labels)
				        	adjusted_mutual_info_score=metrics.adjusted_mutual_info_score(tags, labels)
				        	# silhouette_score=metrics.silhouette_score(coords, labels)
				        	classification_report=metrics.classification_report(tags, labels)
				        	# accuracy_score =metrics.accuracy_score(coords, labels)
							# auc=metrics.auc(x, y[, reorder])	Compute Area Under the Curve (AUC) using the trapezoidal rule
							# metrics.average_precision_score(y_true, y_score)	Compute average precision (AP) from prediction scores
							
							# metrics.confusion_matrix(y_true, y_pred[, ...])	Compute confusion matrix to evaluate the accuracy of a classification
							# metrics.f1_score(y_true, y_pred[, labels, ...])	Compute the F1 score, also known as balanced F-score or F-measure


							# metrics.precision_recall_curve(y_true, ...)	Compute precision-recall pairs for different probability thresholds
							# metrics.precision_recall_fscore_support(...)	Compute precision, recall, F-measure and support for each class
							# metrics.precision_score(y_true, y_pred[, ...])	Compute the precision
							# metrics.recall_score(y_true, y_pred[, ...])	Compute the recall
							# metrics.roc_auc_score(y_true, y_score[, ...])	Compute Area Under the Curve (AUC) from prediction scores
							# metrics.roc_curve(y_true, y_score[, ...])	Compute Receiver operating characteristic (ROC)
							# metrics.zero_one_loss(y_true, y_pred[, ...])
					        f.write(unicode('Estimated number of clusters: %d\n' % n_clusters_, errors='ignore'));
					        f.write(unicode("Homogeneity: %0.3f\n" % homogeneity_score, errors='ignore'));
					        f.write(unicode("Completeness: %0.3f\n" % completeness_score, errors='ignore'));
					        f.write(unicode("V-measure: %0.3f\n" % v_measure_score, errors='ignore'));
					        f.write(unicode("Adjusted Rand Index: %0.3f\n" % adjusted_rand_score, errors='ignore'));
					        f.write(unicode("Adjusted Mutual Information: %0.3f\n" % adjusted_mutual_info_score, errors='ignore'));
					        # f.write("Silhouette Coefficient: %0.3f\n\n" % silhouette_score)
					        f.write(unicode("Classification Report: \n{}\n".format(classification_report), errors='ignore'));

	        # vectorFile.write(unicode('{}\n'.format(coords), errors='ignore'));

