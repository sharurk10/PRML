# Import libraries for data generation, MLE, evaluation, and plotting.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Set experiment size, train/test ratio, and random seed.

N_PER_CLASS=10000
TRAIN_RATIO=0.80
RANDOM_SEED=42
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER=os.path.join(BASE_DIR,"19-20261005T170052Z-1-001","19")
TRAIN_FILE=os.path.join(DATA_FOLDER,"trian.txt")
DEV_FILE=os.path.join(DATA_FOLDER,"dev.txt")
USE_SUPPLIED_TRAIN_DEV_SPLIT=False
OUTPUT_FOLDER=os.path.join(BASE_DIR,"assignment5_outputs")

os.makedirs(OUTPUT_FOLDER,exist_ok=True)
# Compute a matrix square root for Gaussian data generation.
def covariance_square_root(S):
    eigenvalues,eigenvectors=np.linalg.eigh(S)
    eigenvalues=np.maximum(eigenvalues,0)
    return eigenvectors@np.diag(np.sqrt(eigenvalues))@eigenvectors.T
# Generate samples from a 2-D Gaussian distribution.

def generate_gaussian_class(mu,Sigma,n,seed):
    rng=np.random.default_rng(seed)
    Z=rng.normal(0,1,size=(n,2))
    return mu+Z@covariance_square_root(Sigma).T
# Split each class into 80% training and 20% testing data.

def stratified_split(X,y,train_ratio=0.80,seed=42):
    rng=np.random.default_rng(seed)
    train_indices=[]
    test_indices=[]
    for c in np.unique(y):
        indices=np.where(y==c)[0]
        rng.shuffle(indices)
        n_train=int(len(indices)*train_ratio)
        train_indices.extend(indices[:n_train])
        test_indices.extend(indices[n_train:])
    train_indices=np.array(train_indices)
    test_indices=np.array(test_indices)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return X[train_indices],X[test_indices],y[train_indices],y[test_indices]

# Estimate means, covariance matrices, and priors using maximum likelihood.
def estimate_mle(X_train,y_train,shared_covariance=False,covariance_type="full"):
    classes=np.unique(y_train)
    means={}
    covariances={}
    priors={}
    total=len(X_train)
    for c in classes:
        Xc=X_train[y_train==c]
        mu=np.mean(Xc,axis=0)
        D=Xc-mu
        means[c]=mu
        covariances[c]=(D.T@D)/len(Xc)
        priors[c]=len(Xc)/total
    if shared_covariance:
        pooled=np.zeros((X_train.shape[1],X_train.shape[1]))
        for c in classes:
            D=X_train[y_train==c]-means[c]
            pooled+=D.T@D
        pooled/=total
        for c in classes:
            covariances[c]=pooled.copy()
    for c in classes:
        S=covariances[c]
        if covariance_type=="isotropic":
            sigma2=np.trace(S)/S.shape[0]
            covariances[c]=sigma2*np.eye(S.shape[0])
        elif covariance_type=="diagonal":
            covariances[c]=np.diag(np.diag(S))
        elif covariance_type=="full":
            covariances[c]=S
        else:
            raise ValueError("Unknown covariance type.")
    return {"classes":classes,"means":means,"covariances":covariances,"priors":priors}

# Compute Gaussian discriminant scores for every class.
def discriminant_scores(X,model):
    classes=model["classes"]
    scores=np.zeros((len(X),len(classes)))
    for j,c in enumerate(classes):
        mu=model["means"][c]
        Sigma=model["covariances"][c]+1e-9*np.eye(model["covariances"][c].shape[0])
        sign,logdet=np.linalg.slogdet(Sigma)
        if sign<=0:
            raise ValueError(f"Covariance matrix for class {c} is not positive definite.")
        D=X-mu
        solved=np.linalg.solve(Sigma,D.T).T
        mahalanobis=np.sum(D*solved,axis=1)
        scores[:,j]=-0.5*logdet-0.5*mahalanobis+np.log(model["priors"][c])
    return scores
# Assign each sample to the class with the highest score.
def classify_mle(X,model):
    scores=discriminant_scores(X,model)
    return model["classes"][np.argmax(scores,axis=1)]

# Build the multiclass confusion matrix.
def make_confusion_matrix(y_true,y_pred,classes):
    matrix=np.zeros((len(classes),len(classes)),dtype=int)
    class_to_index={c:i for i,c in enumerate(classes)}
    for actual,predicted in zip(y_true,y_pred):
        matrix[class_to_index[actual],class_to_index[predicted]]+=1
    return matrix
# Calculate classification accuracy.

def accuracy_score(y_true,y_pred):
    return np.mean(y_true==y_pred)
# Calculate one-vs-rest ROC points and AUC for one class.

def calculate_roc(y_true,scores,positive_class):
    y_binary=(y_true==positive_class).astype(int)
    order=np.argsort(scores)[::-1]
    y_binary=y_binary[order]
    scores=scores[order]
    positives=np.sum(y_binary==1)
    negatives=np.sum(y_binary==0)
    tpr=[0.0]
    fpr=[0.0]
    tp=fp=0
    previous_score=None
    for label,score in zip(y_binary,scores):
        if previous_score is not None and score!=previous_score:
            tpr.append(tp/positives if positives else 0.0)
            fpr.append(fp/negatives if negatives else 0.0)
        if label==1:
            tp+=1
        else:
            fp+=1
        previous_score=score
    tpr.append(tp/positives if positives else 0.0)
    fpr.append(fp/negatives if negatives else 0.0)
    fpr=np.array(fpr)
    tpr=np.array(tpr)
    order=np.argsort(fpr)
    fpr=fpr[order]
    tpr=tpr[order]
    auc=np.trapezoid(tpr,fpr) if hasattr(np,"trapezoid") else np.trapz(tpr,fpr)
    return fpr,tpr,auc
# Save one-vs-rest multiclass ROC curves and AUC values.

def save_roc_curve(X_test,y_test,model,title,folder):
    scores=discriminant_scores(X_test,model)
    classes=model["classes"]
    fig,ax=plt.subplots(figsize=(8,7))
    auc_rows=[]
    macro_fpr=np.linspace(0,1,500)
    macro_tpr=np.zeros_like(macro_fpr)
    for j,c in enumerate(classes):
        fpr,tpr,auc=calculate_roc(y_test,scores[:,j],c)
        macro_tpr+=np.interp(macro_fpr,fpr,tpr)
        auc_rows.append({"Class":c,"AUC":auc})
        ax.plot(fpr,tpr,linewidth=2,label=f"Class {c} vs Rest (AUC = {auc:.4f})")
    macro_tpr/=len(classes)
    macro_auc=np.trapezoid(macro_tpr,macro_fpr) if hasattr(np,"trapezoid") else np.trapz(macro_tpr,macro_fpr)
    auc_rows.append({"Class":"Macro Average","AUC":macro_auc})
    ax.plot(macro_fpr,macro_tpr,linewidth=2,linestyle="--",label=f"Macro Average (AUC = {macro_auc:.4f})")
    ax.plot([0,1],[0,1],linestyle=":",linewidth=1.5,label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title+" - ROC Curve")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(folder,"roc_curve.png"),dpi=300,bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(auc_rows).to_csv(os.path.join(folder,"roc_auc.csv"),index=False)

# Save the confusion matrix as a PNG image and CSV table.
def save_confusion_matrix(cm,classes,title,folder):
    fig,ax=plt.subplots(figsize=(7,6))
    image=ax.imshow(cm)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels([f"Class {c}" for c in classes])
    ax.set_yticklabels([f"Class {c}" for c in classes])
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    fig.colorbar(image,ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(folder,"confusion_matrix.png"),dpi=300,bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(cm,index=[f"Actual {c}" for c in classes],columns=[f"Predicted {c}" for c in classes]).to_csv(os.path.join(folder,"confusion_matrix.csv"))

# Print and save the MLE parameters for each class.
def print_model_parameters(model,title,folder):
    print("\n"+"="*70)
    print(title)
    print("="*70)
    output=[title,"="*len(title)]
    for c in model["classes"]:
        prior=model["priors"][c]
        mean=np.round(model["means"][c],6)
        cov=np.round(model["covariances"][c],6)
        print(f"\nClass {c}")
        print("Prior probability =",round(prior,6))
        print("Estimated Mean =",mean)
        print("Estimated Covariance =")
        print(cov)
        output.extend([f"\nClass {c}",f"Prior probability = {prior:.6f}","Estimated Mean =",np.array2string(mean),"Estimated Covariance =",np.array2string(cov)])
    with open(os.path.join(folder,"mle_parameters.txt"),"w",encoding="utf-8") as f:
        f.write("\n".join(output))
# Print and save test classification results.

def print_results(y_true,y_pred,classes,title,folder):
    cm=make_confusion_matrix(y_true,y_pred,classes)
    acc=accuracy_score(y_true,y_pred)
    table=pd.DataFrame(cm,index=[f"Actual {c}" for c in classes],columns=[f"Predicted {c}" for c in classes])
    print("\n"+"-"*70)
    print(title)
    print("-"*70)
    print("\nConfusion Matrix")
    print(table)
    print("\nAccuracy =",round(acc*100,4),"%")
    save_confusion_matrix(cm,classes,title,folder)
    with open(os.path.join(folder,"results.txt"),"w",encoding="utf-8") as f:
        f.write(title+"\n"+"="*len(title)+"\n\nConfusion Matrix\n"+table.to_string()+f"\n\nAccuracy = {acc*100:.4f}%\n")
    return cm,acc
# Plot decision regions, boundaries, and train/test samples.

def plot_decision_boundary(X_train,y_train,X_test,y_test,model,title,folder):
    all_X=np.vstack((X_train,X_test))
    x_min,x_max=all_X[:,0].min(),all_X[:,0].max()
    y_min,y_max=all_X[:,1].min(),all_X[:,1].max()
    x_range=max(x_max-x_min,1e-6)
    y_range=max(y_max-y_min,1e-6)
    x=np.linspace(x_min-0.08*x_range,x_max+0.08*x_range,450)
    y=np.linspace(y_min-0.08*y_range,y_max+0.08*y_range,450)
    xx,yy=np.meshgrid(x,y)
    grid=np.column_stack((xx.ravel(),yy.ravel()))
    grid_prediction=classify_mle(grid,model)
    classes=model["classes"]
    class_to_index={c:i for i,c in enumerate(classes)}
    Z=np.array([class_to_index[c] for c in grid_prediction]).reshape(xx.shape)
    fig,ax=plt.subplots(figsize=(9,7))
    ax.contourf(xx,yy,Z,levels=np.arange(len(classes)+1)-0.5,alpha=0.18)
    ax.contour(xx,yy,Z,levels=np.arange(len(classes)-1)+0.5,linewidths=2)
    for c in classes:
        train_points=X_train[y_train==c]
        test_points=X_test[y_test==c]
        ax.scatter(train_points[:,0],train_points[:,1],s=10,alpha=0.08,label=f"Class {c} Train")
        ax.scatter(test_points[:,0],test_points[:,1],s=32,alpha=0.70,marker="x",label=f"Class {c} Test")
        mu=model["means"][c]
        ax.scatter(mu[0],mu[1],marker="*",s=180,edgecolors="black",linewidths=1.2,label=f"Class {c} Mean")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(folder,"decision_boundary.png"),dpi=300,bbox_inches="tight")
    plt.close(fig)
# Save concise observations based on accuracy and class-wise errors.
def write_observation(folder,title,accuracy,cm,classes):
    total=np.sum(cm)
    correct=np.trace(cm)
    errors=total-correct
    class_errors={c:np.sum(cm[i])-cm[i,i] for i,c in enumerate(classes)}
    worst_class=max(class_errors,key=class_errors.get)
    lines=[title,"="*len(title),"",f"Test accuracy = {accuracy*100:.4f}%",f"Correct classifications = {correct} / {total}",f"Misclassified samples = {errors}",f"Class {worst_class} has the largest number of misclassified samples."]
    with open(os.path.join(folder,"observations.txt"),"w",encoding="utf-8") as f:
        f.write("\n".join(lines))
# Run one synthetic-data covariance experiment.

def run_part1_experiment(experiment_name,means,true_covariances,covariance_type,shared_covariance):
    print("\n"+"#"*80)
    print("PART 1:",experiment_name)
    print("#"*80)
    classes=np.array([1,2,3])
    X_parts=[]
    y_parts=[]
    for i,c in enumerate(classes):
        X_parts.append(generate_gaussian_class(means[c],true_covariances[c],N_PER_CLASS,RANDOM_SEED+i+1))
        y_parts.append(np.full(N_PER_CLASS,c))
    X=np.vstack(X_parts)
    y=np.concatenate(y_parts)
    X_train,X_test,y_train,y_test=stratified_split(X,y,TRAIN_RATIO,RANDOM_SEED)
    print("Total samples =",len(X))
    print("Training samples =",len(X_train))
    print("Testing samples =",len(X_test))
    safe_name=experiment_name.lower().replace(" ","_").replace("/","_")
    output_folder=os.path.join(OUTPUT_FOLDER,"part1_"+safe_name)
    os.makedirs(output_folder,exist_ok=True)
    model=estimate_mle(X_train,y_train,shared_covariance,covariance_type)
    print_model_parameters(model,experiment_name+" - MLE Parameters",output_folder)
    y_pred=classify_mle(X_test,model)
    cm,acc=print_results(y_test,y_pred,model["classes"],experiment_name+" - Test Results",output_folder)
    save_roc_curve(X_test,y_test,model,experiment_name,output_folder)
    plot_decision_boundary(X_train,y_train,X_test,y_test,model,"Part 1 - "+experiment_name,output_folder)
    write_observation(output_folder,"Part 1 - "+experiment_name+" Observation",acc,cm,model["classes"])
    return {"name":experiment_name,"X_train":X_train,"X_test":X_test,"y_train":y_train,"y_test":y_test,"model":model,"confusion_matrix":cm,"accuracy":acc}
# Run all three covariance types with both shared and class-specific covariance.

def run_part1():
    means={1:np.array([5.0,5.0]),2:np.array([9.0,9.0]),3:np.array([5.0,9.0])}
    experiments=[
        ("Same Isotropic Covariance",{1:np.eye(2),2:np.eye(2),3:np.eye(2)},"isotropic",True),
        ("Same Diagonal Covariance",{1:np.array([[4.0,0.0],[0.0,1.0]]),2:np.array([[4.0,0.0],[0.0,1.0]]),3:np.array([[4.0,0.0],[0.0,1.0]])},"diagonal",True),
        ("Same Full Covariance",{1:np.array([[4.0,1.5],[1.5,2.0]]),2:np.array([[4.0,1.5],[1.5,2.0]]),3:np.array([[4.0,1.5],[1.5,2.0]])},"full",True),
        ("Different Isotropic Covariance",{1:1.0*np.eye(2),2:2.0*np.eye(2),3:3.0*np.eye(2)},"isotropic",False),
        ("Different Diagonal Covariance",{1:np.array([[2.0,0.0],[0.0,1.0]]),2:np.array([[4.0,0.0],[0.0,2.0]]),3:np.array([[1.0,0.0],[0.0,3.0]])},"diagonal",False),
        ("Different Full Covariance",{1:np.array([[1.5,0.0],[0.0,1.0]]),2:np.array([[4.0,1.2],[1.2,2.0]]),3:np.array([[2.0,-0.8],[-0.8,3.0]])},"full",False)
    ]
    return [run_part1_experiment(name,means,covs,cov_type,shared) for name,covs,cov_type,shared in experiments]
# Load the real Team 19 data.

def load_team19_file(filename):
    data=pd.read_csv(filename,header=None,names=["x1","x2","class"]).dropna()
    X=data[["x1","x2"]].to_numpy(dtype=float)
    y=data["class"].to_numpy()
    return X,y
# Run the real-data MLE experiment for Team 19.
def run_part2():
    print("\n"+"#"*80)
    print("PART 2 - TEAM 19 REAL DATA")
    print("#"*80)
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Could not find {TRAIN_FILE}. Check the dataset folder.")
    if not os.path.exists(DEV_FILE):
        raise FileNotFoundError(f"Could not find {DEV_FILE}. Check the dataset folder.")
    X_train_original,y_train_original=load_team19_file(TRAIN_FILE)
    X_dev,y_dev=load_team19_file(DEV_FILE)
    print("\ntrian.txt samples =",len(X_train_original))
    print("dev.txt samples =",len(X_dev))
    print("Total samples =",len(X_train_original)+len(X_dev))
    print("\nClass counts in trian.txt:")
    print(pd.Series(y_train_original).value_counts().sort_index())
    print("\nClass counts in dev.txt:")
    print(pd.Series(y_dev).value_counts().sort_index())
    if USE_SUPPLIED_TRAIN_DEV_SPLIT:
        X_train,y_train=X_train_original,y_train_original
        X_test,y_test=X_dev,y_dev
    else:
        X_all=np.vstack((X_train_original,X_dev))
        y_all=np.concatenate((y_train_original,y_dev))
        X_train,X_test,y_train,y_test=stratified_split(X_all,y_all,TRAIN_RATIO,RANDOM_SEED)
    print("\nActual Part 2 split:")
    print("Training samples =",len(X_train))
    print("Testing samples =",len(X_test))
    total=len(X_train)+len(X_test)
    print("Training percentage =",round(100*len(X_train)/total,2),"%")
    print("Testing percentage =",round(100*len(X_test)/total,2),"%")
    output_folder=os.path.join(OUTPUT_FOLDER,"part2_team19")
    os.makedirs(output_folder,exist_ok=True)
    model=estimate_mle(X_train,y_train,shared_covariance=False,covariance_type="full")
    print_model_parameters(model,"PART 2 - Team 19 Full-Covariance MLE",output_folder)
    y_pred=classify_mle(X_test,model)
    cm,acc=print_results(y_test,y_pred,model["classes"],"PART 2 - Team 19 Test Results",output_folder)
    save_roc_curve(X_test,y_test,model,"PART 2 - Team 19",output_folder)
    plot_decision_boundary(X_train,y_train,X_test,y_test,model,"Part 2 - Team 19 MLE Decision Boundary",output_folder)
    write_observation(output_folder,"Part 2 - Team 19 Observation",acc,cm,model["classes"])
    pd.DataFrame({"Actual_Class":y_test,"Predicted_Class":y_pred}).to_csv(os.path.join(output_folder,"test_predictions.csv"),index=False)
    return {"X_train":X_train,"X_test":X_test,"y_train":y_train,"y_test":y_test,"model":model,"confusion_matrix":cm,"accuracy":acc}
# Save the final accuracy summary.
def save_final_summary(part1_results,part2_result):
    rows=[]
    for i,result in enumerate(part1_results,1):
        rows.append({"Part":"Part 1","Experiment":f"Experiment {i}: {result['name']}","Training Samples":len(result["X_train"]),"Testing Samples":len(result["X_test"]),"Accuracy (%)":result["accuracy"]*100})
    rows.append({"Part":"Part 2","Experiment":"Team 19 Full-Covariance MLE","Training Samples":len(part2_result["X_train"]),"Testing Samples":len(part2_result["X_test"]),"Accuracy (%)":part2_result["accuracy"]*100})
    summary=pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUTPUT_FOLDER,"final_summary.csv"),index=False)
    return summary

# Run Part 1, Part 2, and save the final summary.
if __name__=="__main__":
    print("="*80)
    print("AI5707 - ASSIGNMENT 5")
    print("MAXIMUM LIKELIHOOD PARAMETER ESTIMATION")
    print("="*80)
    print("\nRandom seed =",RANDOM_SEED)
    print("Required train/test ratio =",TRAIN_RATIO)
    print("Samples per synthetic class =",N_PER_CLASS)
    part1_results=run_part1()
    part2_result=run_part2()
    summary=save_final_summary(part1_results,part2_result)
    print("\n"+"="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))
    print("\nAll required outputs have been saved in:",os.path.abspath(OUTPUT_FOLDER))
    print("\nProgram completed successfully.")
