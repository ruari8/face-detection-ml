function [prediction, pred]= SVMTesting(image,model)

if strcmp(model.type,'binary')
    
    kerneloption.matrix=svmkernel(image,'gaussian',model.param.sigmakernel,model.xsup);
    pred = svmval(image,model.xsup,model.w,model.w0,model.param.kernel,kerneloption);
 
    if pred>0
        prediction = 1;
    else
        prediction = -1;
    end
    
end
    
end