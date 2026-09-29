import os
import torch
from utils import make_cuda
import torch.nn.functional as F
import torch.nn as nn
import param
import torch.optim as optim
from utils import save_model
import pandas as pd
import numpy as np
import pickle
import subprocess
import ast
import re
import matplotlib.pyplot as plt


def pretrain(args, encoder, classifier, data_loader):
    """Train classifier for source domain."""

    # setup criterion and optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                           lr=param.c_learning_rate)
    CELoss = nn.CrossEntropyLoss()

    encoder.train()
    classifier.train()

    for epoch in range(args.pre_epochs):
        for step, (reviews, mask, labels) in enumerate(data_loader): 
            reviews = make_cuda(reviews)
            mask = make_cuda(mask)
            labels = make_cuda(labels)

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for discriminator
            feat = encoder(reviews, mask)
            cls_loss = 0
            t = 0
            
            for i in range(param.num_exits):
                preds = classifier[i](feat[i])
                loss = CELoss(preds, labels)
                cls_loss += i * loss   # weight the loss by the exit index
                t += i
            cls_loss = cls_loss / t

            # optimize source classifier
            cls_loss.backward()
            optimizer.step()

            # print step info
            if (step + 1) % args.pre_log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: cls_loss=%.4f"
                      % (epoch + 1,
                         args.pre_epochs,
                         step + 1,
                         len(data_loader),
                         cls_loss.item()))

    # save final model
    save_model(args, encoder, param.src_encoder_path)
    save_model(args, classifier, param.src_classifier_path)
    
    return encoder, classifier

def adapt(args, src_encoder, discriminator,
          src_classifier, src_data_loader, tgt_data_train_loader, tgt_data_all_loader):
    """Train encoder for target domain."""

    src_encoder.eval()
    src_classifier.eval()
    discriminator.train()

    # setup criterion and optimizer
    BCELoss = nn.BCELoss()
    CELoss = nn.CrossEntropyLoss()
    optimizer_D = optim.Adam(discriminator.parameters(), lr=param.d_learning_rate)
    len_data_loader = min(len(src_data_loader), len(tgt_data_train_loader))

    for epoch in range(args.num_epochs):
        data_zip = enumerate(zip(src_data_loader, tgt_data_train_loader))
        for step, ((reviews_src, src_mask, label_src), (reviews_tgt, tgt_mask, _)) in data_zip:
            reviews_src = make_cuda(reviews_src)
            src_mask = make_cuda(src_mask)
            
            label_src = label_src.to(dtype=torch.float32)
            label_src = make_cuda(label_src).unsqueeze(1)
            
            reviews_tgt = make_cuda(reviews_tgt)
            tgt_mask = make_cuda(tgt_mask)
            
            feat_src = src_encoder(reviews_src, src_mask)

            optimizer_D.zero_grad()
            dis_loss = 0
            t = 0

            for i in range(param.num_exits):
                pred_concat = discriminator[i](feat_src[i])
                
                dis_loss += BCELoss(pred_concat, label_src)  # weight the loss by the exit index
                
                t += i
            dis_loss = dis_loss
            dis_loss.backward()
            optimizer_D.step()

            pred_cls = torch.squeeze(torch.round(pred_concat))
 
            acc = (pred_cls == label_src).float().mean()

            if (step + 1) % args.log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: "
                      "acc=%.4f d_loss=%.4f"
                      % (epoch + 1,
                         args.num_epochs,
                         step + 1,
                         len_data_loader,
                         acc.item(),
                         dis_loss.item()))

    return discriminator

def evaluate(args,encoder, classifier, data_loader):# add discriminator 
    """Evaluation for target encoder by source classifier on target dataset."""
    encoder.eval()
    classifier.eval() 

    all_labels = []  
    all_probs = []  

    with torch.no_grad():
        all_true_labels=[]
        
        for (reviews, mask, labels) in data_loader:
            reviews = make_cuda(reviews)
            mask = make_cuda(mask)
            labels = make_cuda(labels)
            
            all_true_labels.extend(labels.cpu().numpy())
            
            # Forward pass through the encoder
            feat = encoder(reviews, mask)

            batch_labels = []
            batch_probs = []

            for i in range(param.num_exits):
                preds = classifier[i](feat[i])
                probs = torch.nn.functional.softmax(preds, dim=1)
                
                predicted_labels = torch.argmax(preds, dim=1).cpu().numpy()
                predicted_probs = probs.cpu().numpy()
                batch_labels.append(predicted_labels)
                batch_probs.append(np.round(predicted_probs, decimals=4))

            all_labels.append(batch_labels)
            all_probs.append(batch_probs)
            
        all_true_labels = np.array(all_true_labels)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        all_labels = all_labels.reshape(len(all_true_labels), 12, 1)
        all_probs = all_probs.reshape(len(all_true_labels), 12, 2, -1)

        src_y=all_true_labels 

        probability_of_true_label=[]

        for i in range(len(src_y)):
            if src_y[i]==0:
                h=[]
                for j in range(len(all_labels[i])):
                    h.append(all_probs[i][j][0])
                probability_of_true_label.append(h)
            else:
                h=[]
                for j in range(len(all_labels[i])):
                    h.append(all_probs[i][j][1])
                probability_of_true_label.append(h)

        probability_of_true_label_tensor = torch.tensor(probability_of_true_label)

        probability_of_true_label_var = probability_of_true_label_tensor.var(dim=1, unbiased=False)
        mean_of_trueclass_prob = probability_of_true_label_tensor.mean(dim=1)

####################################################################
        # Define alpha and beta
        alpha = args.alpha  
        beta = args.beta    
#####################################################################

        # Define conditions
        condition1 = (mean_of_trueclass_prob < alpha) & (probability_of_true_label_var <= beta)
        condition2 = (mean_of_trueclass_prob >= alpha) & (probability_of_true_label_var > beta)
        condition3 = (mean_of_trueclass_prob < alpha) & (probability_of_true_label_var > beta)
        condition4 = (mean_of_trueclass_prob >= alpha) & (probability_of_true_label_var <= beta)

        colors = ['red' if cond1 else 'orange' if cond2 else 'blue' if cond3 else 'green' for cond1, cond2, cond3, cond4 in zip(condition1, condition2, condition3, condition4)]
        
        count1 = sum(condition1)
        count2 = sum(condition2)
        count3 = sum(condition3)
        count4 = sum(condition4)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(probability_of_true_label_var, mean_of_trueclass_prob, c=colors, alpha=0.5)
        plt.title('Scatter Plot of Variance of Predictions vs. mean_of_trueclass_prob')
        plt.xlabel('Variance of Predictions')
        plt.ylabel('mean_of_trueclass_prob')
        plt.grid(True)

        legend_labels = ['Hard labels', 'Ambiguous', 'Hard and Ambiguous', 'Easy']
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in ['red', 'orange', 'blue', 'green']]
        plt.legend(handles, legend_labels)
        
        xlim = plt.xlim()  
        ylim = plt.ylim()  

        plt.text(xlim[1] * 0.6, ylim[0] + (ylim[1] - ylim[0]) * 0.05, f'Hard labels: {count1}', fontsize=12, verticalalignment='bottom', color='red', weight='bold')
        plt.text(xlim[1] * 0.6, ylim[0] + (ylim[1] - ylim[0]) * 0.10, f'Ambiguous: {count2}', fontsize=12, verticalalignment='bottom', color='orange', weight='bold')
        plt.text(xlim[1] * 0.6, ylim[0] + (ylim[1] - ylim[0]) * 0.15, f'Hard and Ambiguous: {count3}', fontsize=12, verticalalignment='bottom', color='blue', weight='bold')
        plt.text(xlim[1] * 0.6, ylim[0] + (ylim[1] - ylim[0]) * 0.20, f'Easy: {count4}', fontsize=12, verticalalignment='bottom', color='green', weight='bold')

        
        plt.savefig('scatter_plot.png')

        all_labels_list = all_labels.tolist()
        all_probs_list = all_probs.tolist()

        results_df = pd.DataFrame({
          'predicted_class': all_labels_list,
          'probability': all_probs_list,
          'probability_of_true_label':probability_of_true_label,
          'variance_of_prob': probability_of_true_label_var.tolist(),
          'mean_of_trueclass_prob(confidence)': mean_of_trueclass_prob.tolist(),
          'true_labels': src_y.tolist()
        })
        
        results_df['mean_of_trueclass_prob(confidence)'] = results_df['mean_of_trueclass_prob(confidence)'].apply(lambda x: x[0] if isinstance(x, list) else x)
        results_df['variance_of_prob'] = results_df['variance_of_prob'].apply(lambda x: x[0] if isinstance(x, list) else x)

        results_df['deferal_label'] = 0

        # Loop through each row and set the deferal_label
        for index, row in results_df.iterrows():
            if row['mean_of_trueclass_prob(confidence)'] >= alpha and row['variance_of_prob'] <= beta:
                results_df.at[index, 'deferal_label'] = 1
            else:
                results_df.at[index, 'deferal_label'] = 0

        results_df['deferal_label'] = results_df['deferal_label'].apply(lambda x: 0 if x == 1 else 1)

        print("number of hard sample",sum(results_df['deferal_label']))


        # Initialize a list to store accuracy for each layer
        num_layers = len(results_df['predicted_class'].iloc[0])  
        layer_accuracies = []

        # Iterate over each layer
        for layer_idx in range(num_layers):
            predicted_for_layer = results_df['predicted_class'].apply(lambda x: x[layer_idx][0])
            
            correct_predictions = predicted_for_layer == results_df['true_labels']
            
            accuracy = correct_predictions.sum() / len(results_df)
            layer_accuracies.append(f"Accuracy for Layer {layer_idx + 1}: {accuracy:.4f}\n")

        file_path = 'layer_accuracies.txt'
        with open(file_path, 'w') as f:
            f.writelines(layer_accuracies)

        print(f"Layer accuracies have been saved to {file_path}")

        results_df.to_csv('deferal_label.csv', index=False)
              
        print("csv saved")
        
    # set eval state 
    encoder.eval()
    classifier.eval()

    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()
    e = 0
    ne = 0
    cost = 0
    
    # evaluate network
    for (reviews, mask, labels) in data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        labels = make_cuda(labels)
        with torch.no_grad():
            feat = encoder(reviews, mask)
            prev_dis_value = None   
            
            for i in range(param.num_exits):
                preds = classifier[i](feat[i])
                prob_vector = torch.sigmoid(preds)  
                confidence = prob_vector.max().item()  
      
                if confidence > 0.95:
                    cost += (i+1)
                    e += 1
                    break
                elif i == param.num_exits - 1:
                    ne += 1
                    cost += param.num_exits

        loss += criterion(preds, labels).item()
        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum().item()
        
    loss /= len(data_loader)
    acc /= len(data_loader.dataset)
    print("The speedup is", (param.num_exits * len(data_loader)) / cost)
    print("Avg Loss = %.4f, Avg Accuracy = %.4f" % (loss, acc))

    return acc


def evaluate_disc(encoder, classifier, data_loader,discriminator,h,k):
    """Evaluation for target encoder by source classifier on target dataset."""

    encoder.eval()
    classifier.eval()
    discriminator.eval() 

    # init loss and accuracy
    loss = 0
    acc = 0
    acc_1 = 0

    # list for def_confidence
    def_confidence = [[] for i in range(12)]

    # set loss function
    criterion = nn.CrossEntropyLoss()
    e = 0
    exit_lis = []
    ne = 0
    cost = 0
    not_def = 0.001
    defer = 0 
    # evaluate network
    for (reviews, mask, labels) in data_loader:   
        patience = 0  
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        labels = make_cuda(labels)
        pred_cls_1 = None

        with torch.no_grad():
            feat = encoder(reviews, mask)
            prev_dis_value = None   
            
            for i in range(param.num_exits):
                preds = classifier[i](feat[i])
                prob_vector = torch.sigmoid(preds)  # Apply sigmoid to get probabilities
                confidence = prob_vector.max().item()  # Take the maximum probability as confidence

                # Evaluate discriminator
                dis_value = discriminator[i](feat[i])
                
                dis_value = dis_value.item()
                
                if dis_value > h: 
                    pred_cls = labels
                    defer+=1
                    exit_lis.append(i+1)
                    break

                if confidence > k:
                    preds = classifier[i](feat[i])
                    pred_cls = preds.data.max(1)[1]
                    pred_cls_1 = preds.data.max(1)[1]
                    not_def+=1
                    exit_lis.append(i+1)
                    break
                    
                elif i==11:
                   exit_lis.append(i+1)

                   preds = classifier[i](feat[i])
                   pred_cls = preds.data.max(1)[1]
                   not_def+=1 
                
        if pred_cls_1 is not None:
          acc_1+=pred_cls_1.eq(labels.data).cpu().sum().item()
        acc += pred_cls.eq(labels.data).cpu().sum().item()
        print("Not deferred samples are:", not_def,"deferred are", defer, "accuracy is", acc/len(data_loader.dataset), (1-acc_1)/not_def)

    acc /= len(data_loader.dataset)   
    acc_1/=not_def
    risk = 1-acc_1
    speedup = 12*len(data_loader.dataset)/sum(exit_lis)


    df = pd.DataFrame({
        "Def_Confidence": def_confidence
    })

    # Save to CSV
    df.to_csv('def_confidence.csv', index=False)

    print("Avg Loss = %.4f, Avg Accuracy = %.4f, Avg. risk = %.4f"  % (loss, acc, risk))

    return acc, (defer/len(data_loader.dataset))*100, speedup, risk, not_def/len(data_loader.dataset)
