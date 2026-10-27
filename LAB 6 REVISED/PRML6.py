# %%
import math
import matplotlib.pyplot as plt

N = 1000                 # points per class, per covariance experiment
K = 3                    # Gaussian components per class
SEED = 42
weights1 = [0.40, 0.35, 0.25]
weights2 = [0.30, 0.40, 0.30]
means1 = [[-2, -3], [0, 0], [1, 4]]
means2 = [[6, -4], [7, 0], [8, 4]]
assert abs(sum(weights1)-1) < 1e-12 and abs(sum(weights2)-1) < 1e-12
print('Means:', means1, means2, '| samples per class:', N)


# %%
class LCG:
    def __init__(self, seed=42):
        self.state = seed
        self.a = 1103515245
        self.c = 12345
        self.m = 2**31
    def uniform(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m  # 0 <= U < 1
    def randbelow(self, limit):
        return min(int(self.uniform() * limit), limit - 1)

def box_muller(rng):
    u1 = rng.uniform()
    while u1 == 0.0:  # guard against log(0)
        u1 = rng.uniform()
    u2 = rng.uniform()
    radius = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return radius * math.cos(theta), radius * math.sin(theta)

def select_component(weights, rng):
    y = rng.uniform()
    if y < weights[0]:
        return 0
    elif y < weights[0] + weights[1]:
        return 1
    else:
        return 2

rng_demo = LCG(SEED)
print('First uniform:', rng_demo.uniform())
print('Selected Class 1 component index:', select_component(weights1, rng_demo))
print('Box-Muller pair:', box_muller(rng_demo))


# %%
COVARIANCES = {
    'Isotropic': (
        [[[1,0],[0,1]], [[1.5,0],[0,1.5]], [[1.2,0],[0,1.2]]],
        [[[1.2,0],[0,1.2]], [[1,0],[0,1]], [[1.5,0],[0,1.5]]]),
    'Diagonal': (
        [[[2,0],[0,0.8]], [[0.8,0],[0,2]], [[1.5,0],[0,0.6]]],
        [[[1,0],[0,2]], [[2,0],[0,1]], [[1.5,0],[0,0.8]]]),
    'Full': (
        [[[2,0.8],[0.8,1.5]], [[1.5,-0.6],[-0.6,2]], [[2,1],[1,1.5]]],
        [[[1.5,-0.7],[-0.7,2]], [[2,0.9],[0.9,1.5]], [[1.5,-0.5],[-0.5,1]]])
}

def transform_gaussian(z, mean, cov):
    a, b, d = cov[0][0], cov[0][1], cov[1][1]
    if cov[1][0] != b or a <= 0 or a*d - b*b <= 0:
        raise ValueError('Covariance must be symmetric positive definite')
    l11 = math.sqrt(a)
    l21 = b / l11
    l22 = math.sqrt(d - l21*l21)
    return [mean[0] + l11*z[0], mean[1] + l21*z[0] + l22*z[1]]

def generate_gmm(n, weights, means, covariances, rng):
    points, components = [], []
    for _ in range(n):
        k = select_component(weights, rng)
        z = box_muller(rng)
        points.append(transform_gaussian(z, means[k], covariances[k]))
        components.append(k)
    return points, components

datasets = {}
for experiment_idx, (kind, (cov1, cov2)) in enumerate(COVARIANCES.items()):
    # Independent, repeatable sequences per experiment and per class
    points1, labels1 = generate_gmm(N, weights1, means1, cov1, LCG(SEED + experiment_idx*100 + 1))
    points2, labels2 = generate_gmm(N, weights2, means2, cov2, LCG(SEED + experiment_idx*100 + 2))
    datasets[kind] = {'all1':points1, 'all2':points2, 'components1':labels1, 'components2':labels2}
    print(kind, 'counts C1:', [labels1.count(k) for k in range(K)], 'C2:', [labels2.count(k) for k in range(K)])


# %%
fig, axes = plt.subplots(1,3,figsize=(16,4.8))
for ax,(kind,d) in zip(axes,datasets.items()):
    ax.scatter([p[0] for p in d['all1']], [p[1] for p in d['all1']],s=7,alpha=.55,label='Class 1')
    ax.scatter([p[0] for p in d['all2']], [p[1] for p in d['all2']],s=7,alpha=.55,label='Class 2')
    ax.set(title=kind,xlabel='X1',ylabel='X2');ax.legend();ax.grid(alpha=.2);ax.set_aspect('equal',adjustable='box')
plt.suptitle('Generated 2D GMM classes');plt.tight_layout();plt.show()

fig,axes=plt.subplots(3,2,figsize=(11,13))
for r,(kind,d) in enumerate(datasets.items()):
    for col in (1,2):
        ax=axes[r,col-1]
        for k,color in enumerate(['tab:blue','tab:green','tab:orange']):
            pts=[p for p,label in zip(d[f'all{col}'],d[f'components{col}']) if label==k]
            ax.scatter([p[0] for p in pts],[p[1] for p in pts],s=8,alpha=.55,c=color,label=f'Generated component {k+1}')
        ax.set(title=f'{kind}: Class {col}',xlabel='X1',ylabel='X2');ax.legend(fontsize=7);ax.grid(alpha=.2);ax.set_aspect('equal',adjustable='box')
plt.tight_layout();plt.show()


# %%
def split_data(points, rng, fraction=0.8):
    shuffled = [p[:] for p in points]
    for i in range(len(shuffled)-1,0,-1):
        j = rng.randbelow(i+1)
        shuffled[i],shuffled[j]=shuffled[j],shuffled[i]
    cutoff=int(len(shuffled)*fraction)
    return shuffled[:cutoff],shuffled[cutoff:]

def squared_distance(x,y):
    return (x[0]-y[0])**2 + (x[1]-y[1])**2

def closest(point, centers):
    return min(range(len(centers)),key=lambda j:squared_distance(point,centers[j]))

def kmeans(points, k, rng, max_iter=100, tol=1e-6):
    available=list(range(len(points)))
    centers=[]
    for _ in range(k):
        position=rng.randbelow(len(available))
        index=available.pop(position)
        centers.append(points[index][:])
    for iteration in range(1,max_iter+1):
        groups=[[] for _ in range(k)]
        for p in points:
            groups[closest(p,centers)].append(p)
        updated=[]
        for j,group in enumerate(groups):
            if group:
                updated.append([sum(p[0] for p in group)/len(group),sum(p[1] for p in group)/len(group)])
            else:
                updated.append(centers[j][:])
        shift=max(squared_distance(centers[j],updated[j]) for j in range(k))
        centers=updated
        if shift < tol*tol:
            break
    labels=[closest(p,centers) for p in points]
    sse=sum(squared_distance(p,centers[j]) for p,j in zip(points,labels))
    return centers,labels,sse,iteration

for i,(kind,d) in enumerate(datasets.items()):
    for cls in (1,2):
        train,test=split_data(d[f'all{cls}'],LCG(5000+i*100+cls))
        centers,labels,sse,steps=kmeans(train,K,LCG(9000+i*100+cls))
        d[f'train{cls}'],d[f'test{cls}']=train,test
        d[f'centers{cls}'],d[f'cluster_labels{cls}']=centers,labels
        d[f'sse{cls}'],d[f'iterations{cls}']=sse,steps
        assert len(train)==800 and len(test)==200
        print(kind,'Class',cls,'iterations',steps,'SSE',round(sse,2))


# %%
def nearest_grid(data, centers, grid_size=90):
    xlo=min(p[0] for p in data)-1.5; xhi=max(p[0] for p in data)+1.5
    ylo=min(p[1] for p in data)-1.5; yhi=max(p[1] for p in data)+1.5
    xs=[xlo+(xhi-xlo)*j/(grid_size-1) for j in range(grid_size)]
    ys=[ylo+(yhi-ylo)*i/(grid_size-1) for i in range(grid_size)]
    grid=[[closest([x,y],centers) for x in xs] for y in ys]
    return xs,ys,grid

fig,axes=plt.subplots(3,2,figsize=(12,14))
for r,(kind,d) in enumerate(datasets.items()):
    for col in (1,2):
        ax=axes[r,col-1];train=d[f'train{col}'];centers=d[f'centers{col}']
        xs,ys,labels=nearest_grid(train,centers)
        ax.contourf(xs,ys,labels,levels=[-.5,.5,1.5,2.5],colors=['#cce4ff','#d7f4dc','#ffdfc3'],alpha=.65)
        # Boundary display from discrete assignments is an approximate grid contour
        ax.contour(xs,ys,labels,levels=[.5,1.5],colors='black',linewidths=1)
        for k,color in enumerate(['tab:blue','tab:green','tab:orange']):
            pts=[p for p in train if closest(p,centers)==k]
            ax.scatter([p[0] for p in pts],[p[1] for p in pts],s=7,c=color,alpha=.55,label=f'Cluster {k+1}')
        ax.scatter([c[0] for c in centers],[c[1] for c in centers],marker='X',s=150,c='black',edgecolor='white',label='Centroids')
        ax.set(title=f'{kind} — Class {col}',xlabel='X1',ylabel='X2');ax.grid(alpha=.15);ax.legend(fontsize=7);ax.set_aspect('equal',adjustable='box')
plt.suptitle('K-Means training clusters and within-class boundaries');plt.tight_layout();plt.show()


# %%
def class_distortion(p, centers):
    return min(squared_distance(p,c) for c in centers)

def classify(p,centers1,centers2):
    d1=class_distortion(p,centers1)
    d2=class_distortion(p,centers2)
    return (1 if d1<=d2 else 2),d1,d2

for kind,d in datasets.items():
    actual=[1]*len(d['test1'])+[2]*len(d['test2'])
    predictions=[classify(p,d['centers1'],d['centers2'])[0] for p in d['test1']+d['test2']]
    confusion=[[0,0],[0,0]]
    for truth,pred in zip(actual,predictions):confusion[truth-1][pred-1]+=1
    correct=sum(a==b for a,b in zip(actual,predictions))
    d['confusion'],d['correct'],d['accuracy']=confusion,correct,100*correct/len(actual)
    print(kind,'accuracy',f"{d['accuracy']:.2f}%",'confusion',confusion)


# %%
fig,axes=plt.subplots(1,3,figsize=(17,5))
for ax,(kind,d) in zip(axes,datasets.items()):
    test1,test2=d['test1'],d['test2']; allp=test1+test2
    xs=[min(p[0] for p in allp)-2 + j*(max(p[0] for p in allp)-min(p[0] for p in allp)+4)/199 for j in range(200)]
    ys=[min(p[1] for p in allp)-2 + i*(max(p[1] for p in allp)-min(p[1] for p in allp)+4)/199 for i in range(200)]
    diff=[[class_distortion([x,y],d['centers1'])-class_distortion([x,y],d['centers2']) for x in xs] for y in ys]
    ax.contour(xs,ys,diff,levels=[0],colors='black',linewidths=2)
    ax.scatter([p[0] for p in test1],[p[1] for p in test1],s=9,alpha=.6,label='Class 1')
    ax.scatter([p[0] for p in test2],[p[1] for p in test2],s=9,alpha=.6,label='Class 2')
    ax.set(title=f'{kind} — {d["accuracy"]:.2f}%',xlabel='X1',ylabel='X2');ax.grid(alpha=.15);ax.legend(fontsize=8)
plt.suptitle('Held-out test data and equal-distortion two-class boundary');plt.tight_layout();plt.show()


# %%
print(f"{'Covariance':<12} {'Accuracy':>10} {'Correct':>9} {'C1 SSE':>12} {'C2 SSE':>12} {'Iterations':>12}")
for kind,d in datasets.items():
    print(f"{kind:<12} {d['accuracy']:>9.2f}% {d['correct']:>7}/400 {d['sse1']:>12.2f} {d['sse2']:>12.2f} {str(d['iterations1'])+'/'+str(d['iterations2']):>12}")
print('\nObservations:')
print('- Isotropic: roughly circular components; diagonal: axis-aligned ellipses; full: possibly tilted ellipses.')
print('- Within-class K-Means boundaries are nearest-centroid (Voronoi) regions.')
print('- Final class boundary uses equal minimum distortion, not a Gaussian density equality.')
print('- Well-separated means usually support high test accuracy; observed outcomes depend on sampling and initialization.')
print('- Compare measured accuracies above rather than presuming one covariance is always best.')


