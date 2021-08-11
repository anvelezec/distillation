import time
import copy
import torch


def train_model(student, teacher, criterion, optimizer, scheduler, device, dataloaders, num_epochs, dataset_sizes):
            since = time.time()

            teacher.eval()

            best_model_wts = copy.deepcopy(student.state_dict())
            best_acc = 0.0

            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        student.train()  # Set model to training mode
                    else:
                        student.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0
                    running_corrects_teacher = 0

                    # Iterate over data.
                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            t_outputs = teacher(inputs)
                            s_outputs = student(inputs)
                            _, preds = torch.max(s_outputs, 1)
                            _, preds_teacher = torch.max(t_outputs, 1)
                            loss = criterion(s_outputs, t_outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        running_corrects_teacher += torch.sum(preds_teacher == labels.data)
                    if phase == 'train':
                        scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]
                    epoch_acc_teacher = running_corrects_teacher.double() / dataset_sizes[phase]

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))
                    
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc_teacher))

                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(student.state_dict())

                print()

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))

            # load best model weights
            student.load_state_dict(best_model_wts)
            return student