def HMF(Ah,h,d):
    mh=[]#-1
    if len(h)!=0:
        for j in range(len(h)):
            if h[j][-1]==d:
                Vj=h[j][0]
                Wj=h[j][1]
                v=1   #Sensitivity Parameter
                n=len(Ah)
                op=0
                for i in range(n):
                    op=op+max(0,1-max(0,v*min(1,Ah[i]-Wj[i])))+max(0,1-max(0,v*min(1,Vj[i]-Ah[i])))
                a=op*1/(2*n)
                mh.append([a,j])
        mh=sorted(mh,key=lambda x: (x[0],x[1]),reverse=True)
    return mh
####### Data input ############
import numpy as np
x=np.genfromtxt('irisNormal.csv',delimiter=',')
d=x[:,-1].tolist()
x=x[:,:-1].tolist()
train_x=x[:25]
train_x.extend(x[50:75])
train_x.extend(x[100:125])
test_x=x[25:50]
test_x.extend(x[75:100])
test_x.extend(x[125:])
train_y=d[:25]
train_y.extend(d[50:75])
train_y.extend(d[100:125])
test_y=d[25:50]
test_y.extend(d[75:100])
test_y.extend(d[125:])
x=train_x
d=train_y
th=0.2  #theta for putthing bounds on hyperboxes
n=len(x[0])
fir=1    #will contain all hyperboxes
b=0
############ Training Starts ##############
for i,j in zip(x,d):
    if fir==1:#true if first input
        H=[[i,i,j]]
        fir=0
    else:
        hii=HMF(i,H,j)
        if hii!=[]:
            var=0
            for hx in hii:
                hi=hx[1]
                he=H[hi]
                s=0
###########          Expansion        ################                
                for k in range(n):
                    s=s+max(he[1][k],i[k])-min(he[0][k],i[k])
                if n*th >= s:
                    var=1
                    p=[]
                    q=[]
                    for k in range(n):
                        p.append(min(H[hi][0][k],i[k]))
                        q.append(max(H[hi][1][k],i[k]))
                    H[hi][0]=p
                    H[hi][1]=q
    #############         Overlap        ###############
                    for z in range(len(H)):
                        h=H[z]
                        if h[-1]!=j:
                            delta=-1
                            delold=1
                            delnew=1
                            shu=0
                            for dim in range(n):
                                ##### Case1 #####
                                Vji=p[dim]
                                Vki=h[0][dim]
                                Wji=q[dim]
                                Wki=h[1][dim]
                                if Vji<Vki<Wji<Wki:
                                    delnew=min(delold,Wji-Vki)
                                    shu=shu+1
                                elif Vki<Vji<Wki<Wji:
                                    delnew=min(delold,Wki-Vji)
                                    shu=shu+1
                                elif Vji<Vki<Wki<Wji:
                                    delnew=min(delold,min((Wki-Vji),(Wji-Vki)))
                                    shu=shu+1
                                elif Vki<Vji<Wji<Wki:
                                    delnew=min(delold,min((Wki-Vji),(Wji-Vki)))
                                    shu=shu+1
                                if delold-delnew>0:
                                    delta=dim
                                    delold=delnew
                            if shu!=n:
                                delta=-1
                            
    #############         Contraction      ##############
                            if delta!=-1:
                                Vji=p[delta]
                                Vki=h[0][delta]
                                Wji=q[delta]
                                Wki=h[1][delta]
                                gf=0
                                if Vji<Vki<Wji<Wki:
                                    Vki=Wji=(Vki+Wji)/2
                                    gf=1
                                elif Vki<Vji<Wki<Wji:
                                    gf=1
                                    Vji=Wki=(Vji+Wki)/2
                                elif Vji<Vki<Wki<Wji:
                                    gf=1
                                    if((Wki-Vji)>(Wji-Vki)):
                                        Wji=Vki
                                    elif((Wki-Vji)<=(Wji-Vki)):
                                        Vji=Wki
                                elif Vki<Vji<Wji<Wki:
                                    gf=1
                                    if((Wji-Vki)>(Wki-Vji)):
                                        Wki=Vji
                                    elif((Wji-Vki)<=(Wki-Vji)):
                                        Vki=Wji
                                H[hi][0][delta]=Vji
                                H[z][0][delta]=Vki
                                H[hi][1][delta]=Wji
                                H[z][1][delta]=Wki
                    break;
            if var==0:
                H.append([i,i,j])
        else:
            H.append([i,i,j])

print("\nHyperBoxes:-")
print(H)
print("\nTotal number of Hyperboxes: "+str(len(H)))
x=test_x
d=test_y
def HMF1(Ah,h,d):
    m=0
    mh=-1
    if len(h)!=0:
        for j in range(len(h)):
            Vj=h[j][0]
            Wj=h[j][1]
            v=1   #Sensitivity Parameter
            n=len(Ah)
            op=0
            for i in range(n):
                op=op+max(0,1-max(0,v*min(1,Ah[i]-Wj[i])))+max(0,1-max(0,v*min(1,Vj[i]-Ah[i])))
            a=op*1/(2*n)
            if(a>m):
                m=a
                mh=j
    return mh
cla=[[0,0,0] for i in range(len(H))] 
for i in range(len(x)):
    cc=HMF1(x[i],H,d[i])
    cla[cc][int(d[i])-1]=cla[cc][int(d[i])-1]+1
#print(cla)
H_labels=[]
for i in cla:
    maxi=0
    for j in range(len(i)):
        if i[maxi]<i[j]:
            maxi=j
    H_labels.append(maxi+1)
#print(H_labels)
confusion_matrix=[[0,0,0],[0,0,0],[0,0,0]]
for i in range(1,4):
    for j in range(len(H_labels)):
        if i==H_labels[j]:
            confusion_matrix[i-1]=[(r+e) for r,e in zip(confusion_matrix[i-1],cla[:][j])]
    confusion_matrix[i-1]=[e*4 for e in confusion_matrix[i-1]]
confusion_matrix=np.array(confusion_matrix).T.tolist()
print("\nConfusion Matrix")
print(confusion_matrix)