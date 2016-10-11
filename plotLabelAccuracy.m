% -------------------------------------------------------------------------
% This function plots the label wise accuracy 
% Author : Sukrit Shankar 
% -------------------------------------------------------------------------
function plotLabelAccuracy(tags,styleString,...
    labelWiseAccuracy,totalAccuracy,rootFolderName)   

% Configuration 
labelsPerFigure = 20; 

% Plot the label accuracy 
numberOfFigures = ceil(length(labelWiseAccuracy) / labelsPerFigure);
for i = 1:1:numberOfFigures
    typeAccTemp = labelWiseAccuracy(labelsPerFigure*(i-1)+1:min(labelsPerFigure*i,...
        length(labelWiseAccuracy))); 
    tagsTemp = tags(labelsPerFigure*(i-1)+1:min(labelsPerFigure*i,...
        length(labelWiseAccuracy))); 
    
    figure; 
    b = bar(typeAccTemp * 100,0.9);
    b.LineWidth = 1.5;
    b.EdgeColor = [0 .9 .9];
    b.FaceColor = [0 .5 .5];
    yh = ylabel(['Accuracy']);
    set(gca,...
              'linewidth',1,...
              'xcolor',[0.3,0.3,0.3],...
              'fontsize',16,...
              'fontname','arial',...
              'fontweight','bold');
    set(yh,...
              'fontweight','bold',...
              'fontsize',14,...
              'color',[0,0,0]);
    ylim ([-30, 150]); 
    xlim ([0, length(typeAccTemp)+1]); 

    title (char(strcat('',{' '},styleString, {' '},...
        'Total Accuracy = ',num2str(totalAccuracy * 100)))); 
    for k = 1:1:length(tagsTemp)
      t = text(k,-20,tagsTemp{k},'rotation',90); 
      t.FontSize = 10;
      t.FontWeight = 'normal';  
      t.Color = [0.3, 0.3, 0.3];
      t = text(k,100,num2str(typeAccTemp(k)*100),'rotation',90); 
      t.FontSize = 10;
      t.FontWeight = 'bold';  
      t.Color = [0.3, 0.3, 0.3];
    end

    % Save the image 
    set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 30 20])
    print('-dpng', strcat(rootFolderName,'/accuracyPlot_',...
        num2str(i), '.png'), '-r300');
    clf; close all; 
    
    % Clear the variables
    clear typeAccTemp tagsTemp; 
end
