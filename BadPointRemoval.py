import numpy as np
import scipy
import matplotlib.pyplot as plt
import csv
import math

def save_arrays_to_csv(file_path, column_titles, *arrays):
    # Transpose the arrays to align them by columns
    transposed_arrays = np.array(list(zip(*arrays)))
    

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        #Writes the column titles.
        writer.writerow(column_titles)
        
        #Writes in data rows.
        for row in transposed_arrays:
            writer.writerow(row)


data = np.genfromtxt(f'GridDataF6N3.csv', delimiter=',', dtype=float, skip_header=0)
#Index 0 -> Temperature; index 1 -> Sigma; index 2 -> V


Ts = range(15,1000,15)
sigmas = range(0,1000,15); step = 15

print(len(sigmas))
print(len(Ts)*len(sigmas))
print(data.shape)
flags = []

for i, T in enumerate(Ts):
    V = data[i*len(sigmas):(i+1)*len(sigmas),2]


    dVdx = np.diff(V,prepend=[0])
    d2Vdx2 = np.diff(dVdx,append=[0])
    
    for j in range(1,len(sigmas)-2):
        skip = False
        if skip is False:
            #If d2V/dx2 is less than 1% of the mid value of V
            if abs(d2Vdx2[j])>abs(max(V)-min(V))/100:
                if d2Vdx2[j+2]<0 and d2Vdx2[j]<0:
                    #y=mx+c
                    m = (V[j+1]-V[j])/(sigmas[j+1]-sigmas[j])
                    c = -m*sigmas[j]+V[j]
                    
                    y = m*sigmas[j+2] + c

                    if V[j+2]>y:
                        flags.append([T, sigmas[j+2],V[j+2]])
                        #if abs(T-Ts[51])<0.1:

                        #    plt.plot(sigmas,m*sigmas+c)
                        #    plt.scatter(sigmas[j+2],V[j+2],color='blue')
                        #    print(f'V={V[j+2]} and y={y}')
                        skip = True
                        
                        
                if d2Vdx2[j+2]>0 and d2Vdx2[j]>0:
                    m = (V[j+2]-V[j])/(sigmas[j+2]-sigmas[j])
                    c = -m*sigmas[j]+V[j]
                    
                    y = m*sigmas[j+1]+ c

                    if V[j+1]>y:
                        flags.append([T,sigmas[j+1],V[j+1]])
                        #if T==Ts[51]:
                        #    plt.plot(sigmas,m*sigmas+c)
                        #    plt.scatter(sigmas[j+2],V[j+2],color='blue')
                        skip = True
        else:
            skip=False



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
print(flags)
flags=np.array(flags)
ax.scatter(flags[:,0],flags[:,1],flags[:,2],color='red')
ax.plot_trisurf(data[:,0],data[:,1],data[:,2], linewidth=0, antialiased=False)
plt.show()

deletions = []; _deletions=[]
for flag in flags:
    T=flag[0]; Ti = int(np.round(T/step))-1
    sigma=flag[1]; sigmai=int(np.round(sigma/step))
    

    plt.plot(sigmas,data[Ti*len(sigmas):(Ti+1)*len(sigmas),2])
    print(rf'T={T}, $\sigma$={sigma}')
    print(Ts[Ti])
    
    plt.scatter(flag[1],flag[2])
    plt.show()
    
    validInput=False
    while not validInput:
        toRemove = input ("Enter 'r' to remove or 'k' to keep:")
        print(f'You have selected {toRemove}')
        if str(toRemove.strip()) == 'r':
            deletions.append([Ti,sigmai])
            _deletions.append([Ti*len(sigmas)+sigmai])
            print('Point added to removal list.')
            validInput=True
        elif str(toRemove.strip()) == 'k':
            print('Point kept.')
            validInput=True
        else:
            print('Input character not valid. Try again please')
    
print(_deletions)
data_new = np.delete(data, _deletions,axis=0)
print(data[_deletions,:])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_trisurf(data_new[:,0],data_new[:,1],data_new[:,2], linewidth=0, antialiased=False)
plt.show()

save_arrays_to_csv(f'GridDataF6N3Corrected.csv', ['T',r'$\sigma$','V'], data_new[:,0],data_new[:,1],data_new[:,2])