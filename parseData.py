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
        with io.open('params.txt', 'w+', encoding='utf-8') as f:
	        for samples in range(2,100):
		        for base in range(1,5000):
		        	dec = Fraction(base,1000)
		        	eps = float(dec)
		        	print 'eps {:f}'.format(eps)   	
		        	print 'dec {}'.format(dec)   	
		        	print 'base {:f}'.format(base)   	
	        		print samples
			        db = DBSCAN(eps=eps, min_samples=samples, metric='euclidean', algorithm='auto').fit(coords)
			        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
			        # print 'core_samples_mask: {}'.format(core_samples_mask)
			        core_samples_mask[db.core_sample_indices_] = True
			        labels = db.labels_
			        
			        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
			        if n_clusters_ >1:
			        	print 'labels: {}'.format(labels)
			        	txt = 'eps: {} , samples: {}'.format(eps,samples)
			        	f.write(unicode(txt+'\n', errors='ignore'));
				        print('Estimated number of clusters: %d' % n_clusters_)
				        print("Homogeneity: %0.3f" % metrics.homogeneity_score(tags, labels))
				        print("Completeness: %0.3f" % metrics.completeness_score(tags, labels))
				        print("V-measure: %0.3f" % metrics.v_measure_score(tags, labels))
				        print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(tags, labels))
				        print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(tags, labels))
				        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(coords, labels))

	        # vectorFile.write(unicode('{}\n'.format(coords), errors='ignore'));

