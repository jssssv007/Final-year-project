import networkx as nx
from sage.all import *
from math import ceil
# ------------------------------------------------------------------------------------------------------------------
f1 = open("AMiner-Author.txt", "r").read()
arr = []
for i in f1.split("\n\n"):
	arr.append(i.split("\n"))
arr = arr[:-1]
brr = []
d = {}
for i in range(len(arr)):
	arr[i][0] = arr[i][0].split()
	arr[i][0] = arr[i][0][0][0] + arr[i][0][-1]
	arr[i][1] = arr[i][1][3:]
	if "#hi 0" not in arr[i]:
		brr.append(arr[i])
for i in range(len(brr)):
	for j in range(5, 8):
		brr[i][j] = brr[i][j].split()
		brr[i][j] = brr[i][j][0] + ":" + brr[i][j][1]
# d = {#arr[i][0]:[arr[i][5],arr[i][6],arr[i][7]]}
for i in range(len(brr)):
	d[brr[i][0]] = [brr[i][5], brr[i][6], brr[i][7],brr[i][2]]
for i in d:
	d[i][0] = d[i][0].split(":")
	d[i][1] = d[i][1].split(":")
	d[i][2] = d[i][2].split(":")
	k=d[i][-1].split(" ")
	d[i] = {
		d[i][0][0]: float(d[i][0][1]),
		d[i][1][0]: float(d[i][1][1]),
		d[i][2][0]: float(d[i][2][1]),
		k[0]:' '.join(k[i] for i in range(1,len(k)))
	}
g={}
for i in d:
	g[i]={"attr_sum":ceil((5*d[i]['#hi']+d[i]['#upi']+d[i]["#pi"])/3),'#a':d[i]['#a']}
k=sorted(g.items(),key=lambda x:x[1]['attr_sum'],reverse=True)

##calculating the attributes sum and after that converging these values by the formula (5*hi+pi+upi)/3 and as the numerical values are much favourable to the clustering we created a total of 23 groups which merges the author's indices.  
for i in k:
    i[1]["clu_value"] = (i[1]["attr_sum"] // 100) + 1
    if i[1]["attr_sum"] > 1000:
        i[1]["clu_value"] += (i[1]["attr_sum"] % 1000) // 100
##left out with affiliations -- :)
# ------------------------------------------------------------------------------------------------------------------
f2 = open("AMiner-Paper.txt", "r").read()
crr = []
for i in f2.split("\n\n"):
	crr.append(i.split("\n"))
crr = crr[:-1]
# brr=[]
for i in crr:
	i[2] = i[2].replace('"', " ")
	a, *b = i[2].split(" ")
	i[2] = " ".join(b)
c = []
d = {}
for i in crr:
	i[2] = i[2].split(";")
	for j in i[2]:
		if j not in d:
			d[j] = [i[5]]
		else:
			d[j].append(i[5])

g = {}
krr, drr = [], []
for i in brr:
	try:
		g[i[1]] = d[i[1]]
	except:
		krr.append(i[1])
for i in brr:
	if i[1] not in krr:
		drr.append(i)
# ---------------------------------------------------------------------------------------
def blocks(array):
	return [array[i : i + 3] for i in range(0, len(array), 3)]


f = open("AMiner-Coauthor.txt", "r").readlines()
f = "".join(f)
f = f.split()
lp = blocks(f)
"""lp=blocks(f)
#[123,456,1] and [123,789,2] to {'123':{'456':1},'256':25}}}
d={}
for i in lp:
	if i[0] not in d:
		d[i[0]]={i[1]:i[2]}
	else:
		d[i[0]][i[1]]=i[2]
"""
lpr = []
irr = set([i[0] for i in brr])
for i in lp:
	i[1] = "#" + i[1]
	if i[0] in irr and i[1] in irr:
		lpr.append(i)
dic = {}
for i in brr:
	dic[i[0]] = [i[1]]
lprr = []
for i in lpr:
	lprr.append([dic[i[0]], dic[i[1]], i[-1]])
# ------------------------------------------------------------------------------------------------------------------
# Random Pickings

G = nx.Graph()
# lp=['#522324', '1034146', '1']
# set node attributes for each node from dictionary d
# d={'522324':{'#hi':1, '#pi':1.3333, '#upi':0.8222},'1355779':{'#hi':1, '#pi':0.4000, '#upi':0.0800}}
G.add_nodes_from(d.keys())
nx.set_node_attributes(G, d)
# set edge attributes for each edge from list lp
for i in lprr:
	G.add_edge(i[0], i[1], weight=float(i[2]))
# show the graph with labels and node attributes and edge weights
pos = nx.spring_layout(G)
nx.draw_networkx_nodes_labels(G, pos)
nx.draw_networkx_edge_labels(G, pos)
nx.draw_networkx(G, pos)

# G=nx.Graph()
# # for i in lp[:2]:
# #     G.add_edge(i[0],i[1],int(weight=i[2]))
# # pos=nx.spring_layout(G)
# # nx.draw_networkx_nodes(G,pos,node_size=10)
# # nx.draw_networkx_edges(G,pos)
# # nx.draw_networkx_labels(G,pos,font_size=10,font_family='sans-serif')
# # edge_labels=dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])

'''
The p-index is a measure of the productivity of an academic researcher. The p-index is calculated by counting the number of papers that an author has published that have been cited at least p times, where p is an integer. The higher the p-value, the more selective the measure and the higher the p-index score.

UPI (Unique Productivity Index) is an alternative index to the H-index, which is similar to the P-index, but it considers the number of unique papers that have been cited at least p times. UPI is calculated by counting the number of papers that have been cited at least p times and have at least p authors.

Both the p-index and the UPI are bibliometric measures of the productivity and impact of an academic researcher. These parameters are used to compare the productivity and impact of researchers within a specific field.
'''