import torch
from IPython.display import display
from tqdm.auto import tqdm
import matplotlib_inline
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

matplotlib_inline.backend_inline.set_matplotlib_formats("retina")


def CIFAR10_dataloaders(batch_size=64):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Normalize to [-1, 1]
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return trainloader, testloader, classes


def training_monitor(
    device,
    model,
    optimizer,
    criterion,
    num_epochs,
    trainloader,
    testloader,
    plot_interval=1,
):

    train_losses = []
    val_losses = []
    val_accuracies = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    pbar = tqdm(total=num_epochs, leave=True)

    hdisplay_img = display(display_id=True)
    hdisplay_pbar = display(display_id=True)
    title = f"|{'Epoch':^20}|{'Train loss':^20}|{'Validation loss':^20}|{'Validation accuracy, %':^25}|"
    print(title)
    print("_" * len(title))

    for epoch in range(num_epochs):
        # Step 3
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)

        # Step 4
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(testloader)
        val_losses.append(avg_val_loss)
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)

        # tqdm progress bar
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
        # pbar.set_postfix_str(
        #     f"Train Loss: {train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.2f}%"
        # )
        pbar.update()

        # Plotting updates every `plot_interval` epochs
        if (epoch + 1) % plot_interval == 0:
            # Clear the previous plots
            ax1.clear()
            ax2.clear()

            # Plot training and validation loss
            ax1.plot(range(1, epoch + 2), train_losses, "b-", label="Train Loss")
            ax1.plot(range(1, epoch + 2), val_losses, "r-", label="Val Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Loss")
            ax1.legend()
            ax1.grid(linestyle="--", linewidth=0.5, alpha=0.5)

            # Plot validation accuracy
            ax2.plot(range(1, epoch + 2), val_accuracies, "g-", label="Val Accuracy")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy (%)")
            ax2.set_title("Validation Accuracy")
            ax2.legend()
            ax2.grid(linestyle="--", linewidth=0.5, alpha=0.5)
            hdisplay_img.update(fig)
            hdisplay_pbar.update(pbar.container)

        print(
            f"|{epoch+1:^20}|{train_loss:^20.4f}|{avg_val_loss:^20.4f}|{accuracy:^25.4f}|"
        )
    plt.close()
    return model

def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params = {total_params:e}\nTrainable params = {trainable_params:e}")
