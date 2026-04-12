function model = SVMTraining(images, labels)


% first we check if the problem is binary classification or multiclass
if max(labels)<2
    %binary classification
    model.type='binary';
    
    %SVM software requires labels -1 or 1 for the binary problem
    labels(labels==0)=-1;

    %Initilaise and setup SVM parameters
    %using parameters from hyperparamter optimization in fitcsvm
    %fullImage = 909.1755, 1/C, 16.1631, but -> (Inf, 1e-20, 10) is better
    %hog = 24.3811, 1/C, 5.9788
    C = 909.1755;
    lambda = 1/C;  
    sigmakernel= 16.1631;
    K=svmkernel(images,'gaussian',sigmakernel); 
    kerneloption.matrix=K;
	kernel='numerical';
  

    % Calculate the support vectors
    [xsup,w,w0,pos,tps,alpha] = svmclass(images,labels,C,lambda,kernel,kerneloption,0); 

    % create a structure encapsulating all teh variables composing the model
    model.xsup = xsup;
    model.w = w;
    model.w0 = w0;

    model.param.sigmakernel=sigmakernel;
    model.param.kernel=kernel;
    
    
else
    %multiple class classification
     model.type='multiclass';
    
    %SVM software requires labels from 1 to N for the multi-class problem
    labels = labels+1;
    nbclass=max(labels);
    
    %Initilaise and setup SVM parameters
    lambda = 1e-20;  
    C = 100000;
    kerneloption= 5;
    kernel='gaussian';
    
    % Calculate the support vectors
    [xsup,w,b,nbsv]=svmmulticlassoneagainstall(images,labels,nbclass,C,lambda,kernel,kerneloption,1);
    
    % create a structure encapsulating all teh variables composing the model
    model.xsup = xsup;
    model.w = w;
    model.b = b;
    model.nbsv = nbsv;

    model.param.kerneloption=kerneloption;
    model.param.kernel=kernel;
    
end



end