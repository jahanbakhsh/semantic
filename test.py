l = [1,-1,0,1,0,-1,0,-1,-1]
print(0==[0])
dic = dict()
for item in set(l):
    dic[item] = l.count(item)

print(dic.popitem()[0])